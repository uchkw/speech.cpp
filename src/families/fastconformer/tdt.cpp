#include "tdt.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>

#include "error.h"
#include "decoder-step.h"

namespace fastconformer {

namespace {

using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;

bool joint_reuse_requested() {
    const char * value = std::getenv("SPEECH_HAAS_TDT_JOINT_REUSE");
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw std::runtime_error("SPEECH_HAAS_TDT_JOINT_REUSE must be 0 or 1");
}

Allocator new_allocator(ggml_backend_t backend) {
    Allocator a(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), &ggml_gallocr_free);
    if (!a) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    return a;
}

}  // namespace

TdtDecoder::TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint)
    : TdtDecoder(m, prediction, joint, joint_reuse_requested()) {}

TdtDecoder::TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint, bool reuse_joint)
    : prediction_(prediction),
      joint_(joint),
      blank_((int) m.u32("fastconformer.decoder.blank_id")),
      durations_(m.i32_array("fastconformer.decoder.tdt.durations")),
      max_symbols_(m.u32("fastconformer.decoder.tdt.max_symbols")),
      reuse_joint_(reuse_joint) {}

Decoding TdtDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const {
    return run(projected, backend, progress, nullptr);
}

Decoding TdtDecoder::decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress,
                            const Boost & boost) const {
    if (boost.tree.vocab_size() != blank_) throw std::logic_error("the boosting tree's vocabulary is not the model's");
    return run(projected, backend, progress, &boost);
}

Decoding TdtDecoder::run(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress,
                         const Boost * boost) const {
    const int hidden = joint_.hidden();
    const int64_t frames = (int64_t) (projected.size() / (size_t) hidden);
    auto frame = [&](int64_t t) {
        return std::vector<float>(projected.begin() + t * hidden, projected.begin() + (t + 1) * hidden);
    };
    // The step's graph and the joint's alone keep an allocator each, so that neither is planned again when the
    // other runs.
    DecoderStep step(prediction_, joint_, backend, 1);
    Allocator joint_allocator(nullptr, ggml_gallocr_free);
    if (!reuse_joint_) joint_allocator = new_allocator(backend);
    std::unique_ptr<JointStep> blank_step;

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
    PredictionState state = prediction_.initial_state();
    int32_t label = blank_;
    int64_t t = 0, last_token_frame = -1;
    uint32_t tokens_on_frame = 0;
    while (t < frames) {
        if (progress && !progress((double) t / (double) frames)) break;
        // The prediction network takes the last label, and the joint is evaluated at the current frame with its
        // output, in one graph.
        step.compute(label, state, frame(t));
        d.graphs++;
        state = step.state();
        const std::vector<float> prediction = step.prediction();
        std::vector<float> out;
        step.logits(out);

        int64_t label_frame;
        int32_t duration;
        for (;;) {
            label = label_of(out.data());
            duration = durations_[(size_t) first_argmax(out.data() + blank_ + 1, (int) durations_.size())];
            if (label == blank_ && duration == 0) duration = 1;
            label_frame = t;
            t += duration;
            if (label != blank_ || t >= frames) break;
            if (reuse_joint_) {
                if (!blank_step) blank_step = std::make_unique<JointStep>(joint_, backend);
                blank_step->compute(frame(t), prediction);
                out = blank_step->logits();
            } else {
                Graph j(64);
                ggml_tensor * next = joint_.build(j.ctx(), j.input(frame(t), hidden), j.input(prediction, hidden));
                j.output(next);
                j.compute(backend, joint_allocator.get());
                out = Graph::read(next);
            }
            d.graphs++;
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
