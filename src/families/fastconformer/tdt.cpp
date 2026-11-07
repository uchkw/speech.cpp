#include "tdt.h"

#include <memory>
#include <stdexcept>

#include "error.h"
#include "decoder-step.h"
#include "keyword-prefix.h"

namespace fastconformer {

namespace {

using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

Allocator new_allocator(ggml_backend_t backend) {
    Allocator a(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), &ggml_gallocr_free);
    if (!a) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    return a;
}

}  // namespace

TdtDecoder::TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint)
    : prediction_(prediction),
      joint_(joint),
      blank_((int) m.u32("fastconformer.decoder.blank_id")),
      durations_(m.i32_array("fastconformer.decoder.tdt.durations")),
      max_symbols_(m.u32("fastconformer.decoder.tdt.max_symbols")) {}

Decoding TdtDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const {
    return run(projected, backend, progress, nullptr);
}

Decoding TdtDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress,
                            const Boost & boost) const {
    if (boost.tree.vocab_size() != blank_) throw std::logic_error("the boosting tree's vocabulary is not the model's");
    return run(projected, backend, progress, &boost);
}

Decoding TdtDecoder::record_keyword_prefix(const std::vector<float> & projected, ggml_backend_t backend,
                                          const DecodingProgress & progress, KeywordPrefix & prefix) const {
    return run(projected, backend, progress, nullptr, &prefix);
}

Decoding TdtDecoder::decode_keyword_prefix(const std::vector<float> & projected, ggml_backend_t backend,
                                          const DecodingProgress & progress, const Boost & boost, const KeywordPrefix & prefix) const {
    if (boost.tree.vocab_size() != blank_) throw std::logic_error("the boosting tree's vocabulary is not the model's");
    return run(projected, backend, progress, &boost, nullptr, &prefix);
}

Decoding TdtDecoder::run(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress,
                         const Boost * boost, KeywordPrefix * record, const KeywordPrefix * reuse) const {
    if (reuse && (reuse->decoder_ != this || reuse->backend_ != backend)) {
        throw std::logic_error("the keyword prefix belongs to another decoder or backend");
    }
    if (record) {
        record->entries_.clear();
        record->decoder_ = this;
        record->backend_ = backend;
        record->entries_.reserve(KeywordPrefix::kCapacity);
    }
    const int hidden = joint_.hidden();
    const int64_t frames = (int64_t) (projected.size() / (size_t) hidden);
    auto frame = [&](int64_t t) {
        return std::vector<float>(projected.begin() + t * hidden, projected.begin() + (t + 1) * hidden);
    };
    // The step's graph and the joint's alone keep an allocator each, so that neither is planned again when the
    // other runs.
    DecoderStep step(prediction_, joint_, backend, 1);
    const Allocator joint_allocator = new_allocator(backend);

    // The tree's bonus of each token from its state, which the tokens emitted so far reached, and the state each leads to.
    std::vector<float> bonus, boosted((size_t) blank_);
    std::vector<int32_t> reached;
    int32_t tree_state = BoostingTree::kStart;
    if (boost) boost->tree.advance(tree_state, bonus, reached);
    // The argmax of the tokens and the blank; where it is a token and words are boosted, the argmax of the tokens'
    // logits plus alpha times their bonus, rounded as NeMo's float tensors round the product and then the sum, which a
    // contraction of the two into one fused multiply-add would not.
    auto label_of = [&](const float * out) {
        const int32_t label = first_argmax(out, blank_ + 1);
        if (!boost || label == blank_) return label;
        for (int i = 0; i < blank_; i++) {
            const float scaled = bonus[(size_t) i] * boost->alpha;
            boosted[(size_t) i] = out[i] + scaled;
        }
        return (int32_t) first_argmax(boosted.data(), blank_);
    };

    Decoding d;
    const auto joint_at = [&](int64_t t, const std::vector<float> & prediction) {
        Graph j(64);
        ggml_tensor * next = joint_.build(j.ctx(), j.input(frame(t), hidden), j.input(prediction, hidden));
        j.output(next);
        j.compute(backend, joint_allocator.get());
        d.graphs++;
        return Graph::read(next);
    };
    PredictionState state = prediction_.initial_state();
    int32_t label = blank_;
    int64_t t = 0, last_token_frame = -1;
    uint32_t tokens_on_frame = 0;
    size_t position = 0;
    bool matching = reuse != nullptr;
    while (t < frames) {
        if (progress && !progress((double) t / (double) frames)) break;
        const KeywordPrefix::Entry * cached = nullptr;
        if (matching) {
            // The label a step is fed is the token the step before it emitted, so a step whose fed label equals the
            // recorded one can only be reached through the whole token prefix of the recording; the first step, fed the
            // blank, matches the same way. One label that differs ends the reuse for good, even where later labels
            // coincide again.
            if (position < reuse->entries_.size() && reuse->entries_[position].label == label) cached = &reuse->entries_[position];
            else matching = false;
        }
        std::vector<float> prediction, out;
        if (cached) {
            state = cached->state;
            prediction = cached->prediction;
            out = joint_at(t, prediction);
        } else {
            step.compute(label, state, frame(t));
            d.graphs++;
            state = step.state();
            prediction = step.prediction();
            step.logits(out);
            if (record && record->entries_.size() < KeywordPrefix::kCapacity) {
                record->entries_.push_back({label, state, prediction});
            }
        }
        ++position;

        int64_t label_frame;
        int32_t duration;
        for (;;) {
            label = label_of(out.data());
            duration = durations_[(size_t) first_argmax(out.data() + blank_ + 1, (int) durations_.size())];
            if (label == blank_ && duration == 0) duration = 1;
            label_frame = t;
            t += duration;
            if (label != blank_ || t >= frames) break;
            out = joint_at(t, prediction);
        }
        if (label == blank_) break;
        d.ids.push_back(label);
        if (boost) {
            tree_state = reached[(size_t) label];
            boost->tree.advance(tree_state, bonus, reached);
        }
        d.frames.push_back(label_frame);
        d.durations.push_back(duration);
        tokens_on_frame = label_frame == last_token_frame ? tokens_on_frame + 1 : 1;
        last_token_frame = label_frame;
        if (t < frames && t == label_frame && tokens_on_frame >= max_symbols_) t++;
    }
    return d;
}

}  // namespace fastconformer
