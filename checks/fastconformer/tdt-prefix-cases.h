// What the TDT prediction prefix check runs beyond the model's dumps: the exact comparison every array of the check is
// held to, deterministic synthetic decoder-only weights the check writes into the work directory it is given, and the
// controlled decoding cases on them. The synthetic weights are the check's own data, chosen so that projected frames
// select labels and durations; they are not a model, and nothing measured on them is evidence of what NeMo's or
// flash-linear-attention's code computes. The real-model differential runs stay in fastconformer-tdt-prefix-check.cpp.
//
// Every case runs the actual TdtDecoder, plain and with the prefix cache, and asserts the expected tokens, frames,
// durations, graphs, progress reports, hits and recorded entries. The case table is fixed, a case that throws counts
// as failed, and the summary names how many cases ran, so a mandatory case cannot go missing in silence.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "compare.h"
#include "model-file.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/tdt.h"
#include "fastconformer/tdt-prefix.h"

namespace tdt_prefix_check {

using namespace fastconformer;

/**
 * The one rule this check holds every array to: the same number of float values, all of them finite on both sides, and
 * every byte equal. A difference the argmax agrees with is still a difference, and no tolerance is borrowed from any
 * other check, for no rounding between the joint-only and the combined graph has been measured; where a real one is,
 * its owner records it here. The signal-to-noise ratio, the largest difference and the argmaxes are printed only to
 * explain a failure, as diagnostics. `labels` and `classes`, where the array is a step's logits, add the argmaxes to
 * those diagnostics.
 */
inline bool exactly_equal(const std::vector<float> & got, const std::vector<float> & want, const std::string & what,
                          int labels = 0, int classes = 0) {
    if (got.size() != want.size()) {
        std::printf("    %s: %zu values against %zu\n", what.c_str(), got.size(), want.size());
        return false;
    }
    for (size_t i = 0; i < got.size(); i++) {
        if (!std::isfinite(got[i]) || !std::isfinite(want[i])) {
            std::printf("    %s: value %zu is not finite (%g against %g)\n", what.c_str(), i, got[i], want[i]);
            return false;
        }
    }
    if (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0) return true;
    size_t differing = 0, first = got.size();
    for (size_t i = 0; i < got.size(); i++) {
        if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
            if (first == got.size()) first = i;
            differing++;
        }
    }
    std::printf("    %s: %zu of %zu values differ, the first at %zu (%g against %g)\n", what.c_str(), differing,
                got.size(), first, got[first], want[first]);
    const Diff d = compare(got, want);
    std::printf("    %s: diagnostics: SNR %6.1f dB, max |diff| %.2e", what.c_str(), d.snr_db, d.max_abs);
    if (labels > 0 && classes > labels) {
        std::printf(", label argmax %d against %d, duration argmax %d against %d", first_argmax(got.data(), labels),
                    first_argmax(want.data(), labels), first_argmax(got.data() + labels, classes - labels),
                    first_argmax(want.data() + labels, classes - labels));
    }
    std::printf("\n");
    return false;
}

/** Answers every progress report until it has answered `stop_after` of them, recording what the decoding reported, in
 * order, so that a case can cancel at a chosen point and compare the reports of two runs. */
struct Progress {
    size_t stop_after = std::numeric_limits<size_t>::max();
    std::vector<double> seen;

    bool step(double done) {
        seen.push_back(done);
        return seen.size() < stop_after;
    }
};

/** The cases' tally: every expectation counts, a failed one the check has already described counts silently, and a
 * failed one given a name prints it. */
struct Tally {
    size_t failed = 0;

    void expect(bool ok, const std::string & what) {
        if (ok) return;
        failed++;
        std::printf("    FAIL: %s\n", what.c_str());
    }

    /** Counts an expectation whose detail the comparison that failed has already printed. */
    void count(bool ok) {
        if (!ok) failed++;
    }

    /** Asserts a decoding's every part: the token ids, the frame of each, the duration of each and the graphs
     * computed, printing the parts that differ. */
    void decoding_is(const Decoding & d, const std::vector<int32_t> & ids, const std::vector<int64_t> & frames,
                     const std::vector<int32_t> & durations, size_t graphs, const std::string & what) {
        if (d.ids == ids && d.frames == frames && d.durations == durations && d.graphs == graphs) return;
        std::string why;
        if (d.ids != ids) why += " the ids differ (" + text_of(d.ids) + " against" + text_of(ids) + ")";
        if (d.frames != frames) why += " the frames differ (" + text_of(d.frames) + " against" + text_of(frames) + ")";
        if (d.durations != durations)
            why += " the durations differ (" + text_of(d.durations) + " against" + text_of(durations) + ")";
        if (d.graphs != graphs) why += " " + std::to_string(d.graphs) + " graphs against " + std::to_string(graphs);
        expect(false, what + ":" + why);
    }

private:
    template <typename T>
    static std::string text_of(const std::vector<T> & v) {
        std::string s;
        for (const T & value : v) s += " " + std::to_string(value);
        return s.empty() ? " none" : s;
    }
};

/** Whether two decodings agree in every part. */
inline bool same(const Decoding & a, const Decoding & b) {
    return a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs;
}

// The dimensions of the synthetic weights, Haas's own, so that the state each entry carries is the size the recorded
// figures are of. The durations are 0 to 4 in order, so duration slot s asks for the duration s.
constexpr int64_t kHidden = 640, kHeads = 5, kHead = 64, kWidth = kHeads * kHead, kTaps = 4, kLayers = 2;
constexpr int kBlank = 16, kDurations = 5, kOutputs = kBlank + 1 + kDurations;
const Storage kFloat = float32_storage();
const char * const kFixtureName = "synthetic-kda-tdt.gguf";

/** A decoder-only F32 fixture with two KDA layers and the complete recurrent and convolution state. */
inline std::vector<TensorSpec> tensors() {
    std::vector<TensorSpec> out = {
        {"pred.embed.weight", {kHidden, kBlank + 1}, kFloat},
        {"pred.norm.weight", {kHidden}, kFloat}, {"pred.norm.bias", {kHidden}, kFloat},
        {"joint.pred.weight", {kHidden, kHidden}, kFloat}, {"joint.pred.bias", {kHidden}, kFloat},
        {"joint.out.weight", {kHidden, kOutputs}, kFloat}, {"joint.out.bias", {kOutputs}, kFloat},
    };
    for (int l = 0; l < kLayers; l++) {
        const std::string p = "pred.kda." + std::to_string(l) + ".";
        const std::vector<TensorSpec> layer = {
            {"norm.weight", {kHidden}, kFloat}, {"norm.bias", {kHidden}, kFloat},
            {"q.weight", {kHidden, kWidth}, kFloat}, {"k.weight", {kHidden, kWidth}, kFloat},
            {"v.weight", {kHidden, kWidth}, kFloat},
            {"q_conv.weight", {kTaps, kWidth}, kFloat}, {"k_conv.weight", {kTaps, kWidth}, kFloat},
            {"v_conv.weight", {kTaps, kWidth}, kFloat},
            {"f_a.weight", {kHidden, kHead}, kFloat}, {"f_b.weight", {kHead, kWidth}, kFloat},
            {"b.weight", {kHidden, kHeads}, kFloat}, {"a_log", {kHeads}, kFloat}, {"dt_bias", {kWidth}, kFloat},
            {"g_a.weight", {kHidden, kHead}, kFloat}, {"g_b.weight", {kHead, kWidth}, kFloat},
            {"g_b.bias", {kWidth}, kFloat}, {"o_norm.weight", {kHead}, kFloat},
            {"o.weight", {kWidth, kHidden}, kFloat},
        };
        for (const TensorSpec & spec : layer) out.push_back({p + spec.name, spec.shape, spec.storage});
    }
    return out;
}

const Layout kLayout = {"fastconformer-tdt-prefix-synthetic", 1,
                        "it is the check's own synthetic file, which only the check reads; run the check again to write it",
                        [](const ModelFile &) { return tensors(); }};

/**
 * Writes the deterministic synthetic weights: a KDA prediction network of small values, and a joint whose prediction
 * projection scales by a tenth and whose output layer is the identity, so that a step's logits are the projected
 * frame's values through a ReLU, perturbed by the prediction alone, and a frame chooses its label and its duration
 * wherever the values it is given stand far enough above the rest.
 */
inline void write_fixture(const std::string & path) {
    const auto specs = tensors();
    size_t bytes = 1024;
    for (const TensorSpec & spec : specs) {
        size_t count = 1;
        for (int64_t n : spec.shape.ne) count *= (size_t) n;
        bytes += ggml_tensor_overhead() + count * sizeof(float) + 64;
    }
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("cannot create the synthetic TDT fixture");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    gguf_set_val_str(file.get(), "general.architecture", kLayout.architecture);
    gguf_set_val_str(file.get(), "general.name", "synthetic-kda-tdt");
    gguf_set_val_u32(file.get(), "general.file_type", 0);
    gguf_set_val_u32(file.get(), "speech.layout", 1);
    gguf_set_val_str(file.get(), "speech.requires", "0.9.4");
    gguf_set_val_str(file.get(), "fastconformer.decoder.kind", "tdt");
    gguf_set_val_str(file.get(), "fastconformer.decoder.prediction", "kda");
    gguf_set_val_u32(file.get(), "fastconformer.decoder.prediction_layers", kLayers);
    gguf_set_val_u32(file.get(), "fastconformer.decoder.kda.num_heads", kHeads);
    gguf_set_val_f32(file.get(), "fastconformer.decoder.kda.norm_eps", 1e-5f);
    gguf_set_val_f32(file.get(), "fastconformer.decoder.kda.gate_norm_eps", 1e-5f);
    gguf_set_val_f32(file.get(), "fastconformer.decoder.kda.qk_norm_eps", 1e-6f);
    gguf_set_val_bool(file.get(), "fastconformer.decoder.kda.final_norm", true);
    gguf_set_val_u32(file.get(), "fastconformer.decoder.blank_id", kBlank);
    gguf_set_val_u32(file.get(), "fastconformer.decoder.tdt.max_symbols", 2);
    const int32_t durations[] = {0, 1, 2, 3, 4};
    gguf_set_arr_data(file.get(), "fastconformer.decoder.tdt.durations", GGUF_TYPE_INT32, durations, kDurations);
    for (size_t s = 0; s < specs.size(); s++) {
        const auto & spec = specs[s];
        ggml_tensor * t = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, GGML_MAX_DIMS, spec.shape.ne);
        ggml_set_name(t, spec.name.c_str());
        float * values = static_cast<float *>(t->data);
        for (int64_t i = 0; i < ggml_nelements(t); i++) {
            float value = std::sin(0.017f * (float) (i + 23 * s)) * 0.025f;
            if (spec.name.find("norm.weight") != std::string::npos) value = 1.0f;
            if (spec.name.find("norm.bias") != std::string::npos) value = 0.0f;
            if (spec.name.find("a_log") != std::string::npos || spec.name.find("dt_bias") != std::string::npos) value = -1.0f;
            if (spec.name == "pred.embed.weight" && i >= kBlank * kHidden) value = 0.0f;
            if (spec.name.rfind("joint.", 0) == 0) value = 0.0f;
            if (spec.name == "joint.pred.weight" && i % kHidden == i / kHidden) value = 0.1f;
            if (spec.name == "joint.out.weight" && i % kHidden == i / kHidden) value = 1.0f;
            values[i] = value;
        }
        gguf_add_tensor(file.get(), t);
    }
    if (!gguf_write_to_file(file.get(), path.c_str(), false)) throw std::runtime_error("cannot write " + path);
}

// The strengths a synthetic projected frame chooses with: a label far above the prediction's contribution, a duration
// beside it, and a value negative enough that the joint's ReLU zeroes every logit of the frame.
constexpr float kLabel = 100.0f, kDuration = 50.0f, kSilenced = -100.0f;

/** One projected frame that asks for `label` and duration slot `slot`; every other value is zero. */
inline std::vector<float> choice(int label, int slot) {
    std::vector<float> row((size_t) kHidden, 0.0f);
    row[label] = kLabel;
    row[kBlank + 1 + slot] = kDuration;
    return row;
}

/** One projected frame every value of which the joint's ReLU turns to a zero logit, so that the label argmax ties over
 * the tokens and the blank, the duration argmax over every duration, and the first of each wins. */
inline std::vector<float> tied() {
    return std::vector<float>((size_t) kHidden, kSilenced);
}

/** Stacks frames into the [T, hidden] rows the decoder takes. */
inline std::vector<float> stack(std::initializer_list<std::vector<float>> frames) {
    std::vector<float> rows;
    for (const std::vector<float> & frame : frames) rows.insert(rows.end(), frame.begin(), frame.end());
    return rows;
}

/** The frames of `rows` repeated `times` over, for an input that reaches the entry cap. */
inline std::vector<float> tiled(const std::vector<float> & rows, int64_t times) {
    std::vector<float> out;
    out.reserve(rows.size() * (size_t) times);
    for (int64_t r = 0; r < times; r++) out.insert(out.end(), rows.begin(), rows.end());
    return out;
}

/** What every case runs on: the synthetic model's decoder and the parts it is made of, on the check's backend. */
struct World {
    const TdtDecoder & decoder;
    const PredictionNetwork & prediction;
    const Joint & joint;
    ggml_backend_t backend;
    Tally & tally;

    /** The audio length the cache is told for `frames` frames, in samples, as the engine's requests count them. */
    static size_t samples_of(int64_t frames) { return (size_t) frames * 320; }

    /** Runs `rows` plainly, telling `progress` how far the decoding comes. */
    Decoding plain(const std::vector<float> & rows, Progress & progress) const {
        return decoder.decode(rows, backend, [&](double done) { return progress.step(done); });
    }

    /** The same with the candidate `reading` recording what it computes, cold or warm on what a cache holds. */
    Decoding cached(const std::vector<float> & rows, TdtPrefixCache::Reading & reading, Progress & progress) const {
        return decoder.decode(rows, backend, [&](double done) { return progress.step(done); }, reading);
    }
};

/** The initial blank every TDT decoding feeds first, published as the one entry a single blank frame produces and
 * reused whole by the next request. */
inline void initial_blank(const World & w) {
    const std::vector<float> rows = stack({choice(kBlank, 1)});
    Progress plain_p, cold_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    w.tally.decoding_is(plain, {}, {}, {}, 1, "the initial blank: the plain decoding");
    w.tally.expect(plain_p.seen.size() == 1, "the initial blank: one progress report");

    TdtPrefixCache cache;
    TdtPrefixCache::Reading cold_r = cache.begin(World::samples_of(1));
    const Decoding cold = w.cached(rows, cold_r, cold_p);
    w.tally.expect(same(plain, cold) && plain_p.seen == cold_p.seen, "the initial blank: the cold run agrees");
    w.tally.expect(cold_r.complete() && cold_r.hits() == 0, "the initial blank: the cold run records without reusing");
    cache.publish(std::move(cold_r));
    w.tally.expect(cache.entries().size() == 1 && cache.entries().front().label == kBlank &&
                       cache.entries().front().next_label == kBlank,
                   "the initial blank: one entry, the blank fed and the blank chosen to end");

    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(1));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.decoding_is(warm, {}, {}, {}, 1, "the initial blank: the warm decoding");
    w.tally.expect(plain_p.seen == warm_p.seen, "the initial blank: the warm run reports the plain one's progress");
    w.tally.expect(warm_r.complete() && warm_r.hits() == 1, "the initial blank: the warm run reuses the blank entry");
}

/** A blank chain: the joint's blank of duration 0 is corrected to advance one frame, the chain evaluates the joint on
 * each later frame with the same prediction, and the token the third frame chooses is emitted with its frame. */
inline void blank_chain(const World & w) {
    const std::vector<float> rows = stack({choice(kBlank, 0), choice(kBlank, 0), choice(3, 1)});
    Progress plain_p, cold_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    w.tally.decoding_is(plain, {3}, {2}, {1}, 3, "the blank chain: the plain decoding");
    w.tally.expect(plain_p.seen.size() == 1, "the blank chain: one prediction step, the blank chain inside it");

    TdtPrefixCache cache;
    TdtPrefixCache::Reading cold_r = cache.begin(World::samples_of(3));
    const Decoding cold = w.cached(rows, cold_r, cold_p);
    w.tally.expect(same(plain, cold) && plain_p.seen == cold_p.seen, "the blank chain: the cold run agrees");
    cache.publish(std::move(cold_r));
    w.tally.expect(cache.entries().size() == 1 && cache.entries().front().label == kBlank &&
                       cache.entries().front().next_label == 3,
                   "the blank chain: the chain publishes the one step it computed");

    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(3));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.expect(same(plain, warm) && plain_p.seen == warm_p.seen, "the blank chain: the warm run agrees");
    w.tally.expect(warm_r.hits() == 1, "the blank chain: the warm run reuses the step and chains on its own");
}

/**
 * A frame every logit of which is zero: the label argmax ties over the tokens and the blank and the duration argmax
 * over every duration, and the first of each wins, so the decoding emits token 0 with duration 0 until the
 * max-symbols rule advances the frame. Two frames give four steps, two on each frame.
 */
inline void ties_and_max_symbols(const World & w) {
    const std::vector<float> rows = stack({tied(), tied()});
    Progress plain_p, cold_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    w.tally.decoding_is(plain, {0, 0, 0, 0}, {0, 0, 1, 1}, {0, 0, 0, 0}, 4, "the ties: the plain decoding");
    w.tally.expect(plain_p.seen.size() == 4, "the ties: four prediction steps");

    TdtPrefixCache cache;
    TdtPrefixCache::Reading cold_r = cache.begin(World::samples_of(2));
    const Decoding cold = w.cached(rows, cold_r, cold_p);
    w.tally.expect(same(plain, cold) && plain_p.seen == cold_p.seen, "the ties: the cold run agrees");
    cache.publish(std::move(cold_r));
    w.tally.expect(cache.entries().size() == 4 && cache.entries().front().label == kBlank &&
                       cache.entries().back().next_label == 0,
                   "the ties: the four steps of the two frames are recorded");

    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(2));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.expect(same(plain, warm) && plain_p.seen == warm_p.seen, "the ties: the warm run agrees");
    w.tally.expect(warm_r.hits() == 4, "the ties: the warm run reuses every step");
}

/**
 * The last token a shorter input emitted but never fed: the shorter audio's prefix serves the longer one up to that
 * step, no entry feeds the token, the longer input computes its prediction, and every recorded entry replies with the
 * prediction network's own state and projected prediction, byte for byte.
 */
inline void last_token_never_fed(const World & w) {
    const std::vector<float> short_rows = stack({choice(5, 4)});
    Progress short_p, short_cold_p;
    const Decoding short_plain = w.plain(short_rows, short_p);
    w.tally.decoding_is(short_plain, {5}, {0}, {4}, 1, "the last one: the shorter input's plain decoding");

    TdtPrefixCache cache;
    TdtPrefixCache::Reading short_r = cache.begin(World::samples_of(1));
    const Decoding short_cold = w.cached(short_rows, short_r, short_cold_p);
    w.tally.expect(same(short_plain, short_cold), "the last one: the shorter input's cold run agrees");
    cache.publish(std::move(short_r));
    w.tally.expect(cache.entries().size() == 1 && cache.entries().front().next_label == 5,
                   "the last one: the token that ended the audio is a next label, never fed");

    const std::vector<float> rows =
        stack({choice(5, 4), choice(3, 1), choice(3, 1), choice(3, 1), choice(2, 1)});
    Progress plain_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    w.tally.decoding_is(plain, {5, 2}, {0, 4}, {4, 1}, 2, "the last one: the longer input's plain decoding");

    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(5));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.expect(same(plain, warm) && plain_p.seen == warm_p.seen, "the last one: the longer input agrees");
    w.tally.expect(warm_r.complete() && warm_r.hits() == 1,
                   "the last one: the longer audio reuses the shorter one's step and computes the token it never fed");
    cache.publish(std::move(warm_r));
    w.tally.expect(cache.entries().size() == 2 && cache.entries().back().label == 5 &&
                       cache.entries().back().next_label == 2,
                   "the last one: the never-fed token's own prediction is recorded");

    // The entries reply with the prediction network's own state and projected prediction, replayed from the zero
    // state over the fed labels alone, each at the frame its step ran on.
    DecoderStep step(w.prediction, w.joint, w.backend, 1);
    PredictionState state = w.prediction.initial_state();
    const int64_t at[] = {0, 4};
    auto frame = [&](int64_t t) {
        return std::vector<float>(rows.begin() + t * kHidden, rows.begin() + (t + 1) * kHidden);
    };
    for (size_t k = 0; k < cache.entries().size(); k++) {
        step.compute(cache.entries()[k].label, state, frame(at[k]));
        w.tally.count(exactly_equal(step.state(), cache.entries()[k].state,
                                    "the last one: entry " + std::to_string(k) + " replays its state"));
        w.tally.count(exactly_equal(step.prediction(), cache.entries()[k].prediction,
                                    "the last one: entry " + std::to_string(k) + " replays its prediction"));
        state = step.state();
    }
}

/**
 * The first divergence no later agreement undoes: the joint's choice at the first step differs from the published
 * entry's, the labels of every later step are the published ones again, and none of them is reused.
 */
inline void divergence_is_final(const World & w) {
    const std::vector<float> recorded = stack({choice(3, 1), choice(4, 1), choice(2, 1)});
    TdtPrefixCache cache;
    {
        Progress p;
        TdtPrefixCache::Reading reading = cache.begin(World::samples_of(3));
        const Decoding plain = w.cached(recorded, reading, p);
        w.tally.decoding_is(plain, {3, 4, 2}, {0, 1, 2}, {1, 1, 1}, 3, "the divergence: the recorded decoding");
        cache.publish(std::move(reading));
    }
    w.tally.expect(cache.entries().size() == 3, "the divergence: three steps recorded");

    // The first frame chooses another token than the entry recorded; the later frames choose the recorded labels.
    const std::vector<float> rows = stack({choice(5, 1), choice(4, 1), choice(2, 1)});
    Progress plain_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    w.tally.decoding_is(plain, {5, 4, 2}, {0, 1, 2}, {1, 1, 1}, 3, "the divergence: the divergent decoding");

    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(3));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.expect(same(plain, warm) && plain_p.seen == warm_p.seen, "the divergence: the warm run agrees");
    w.tally.expect(warm_r.hits() == 1,
                   "the divergence: only the first step is reused, the later matching labels are not");
}

/** Changed frames alter the joint's decisions while the prediction comes from the cache: the same fed-label prefix
 * with different audio reuses the first prediction and decodes the new audio's own tokens. */
inline void changed_frames(const World & w) {
    const std::vector<float> recorded = stack({choice(1, 1), choice(1, 1)});
    TdtPrefixCache cache;
    {
        Progress p;
        TdtPrefixCache::Reading reading = cache.begin(World::samples_of(2));
        const Decoding plain = w.cached(recorded, reading, p);
        w.tally.decoding_is(plain, {1, 1}, {0, 1}, {1, 1}, 2, "the changed frames: the recorded decoding");
        cache.publish(std::move(reading));
    }

    const std::vector<float> rows = stack({choice(7, 1), choice(9, 3)});
    Progress plain_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    w.tally.decoding_is(plain, {7, 9}, {0, 1}, {1, 3}, 2, "the changed frames: the new decoding");

    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(2));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.expect(same(plain, warm) && plain_p.seen == warm_p.seen, "the changed frames: the warm run agrees");
    w.tally.expect(warm_r.hits() == 1,
                   "the changed frames: the reused prediction meets the new frames' own joint decisions");
}

/** The entry cap's boundary: 63 predictions, one below the cap, are all recorded; 64, the cap itself, fill it; 65 and
 * more are computed but only the cap's worth is kept and reused. Every frame of the inputs asks for token 2 with
 * duration 0, which the max-symbols rule turns into two predictions on each frame. */
inline void entry_cap(const World & w) {
    const auto run = [&](const std::vector<float> & rows, size_t tokens, size_t graphs, const std::string & what) {
        Progress plain_p, cold_p, warm_p;
        const Decoding plain = w.plain(rows, plain_p);
        w.tally.expect(plain.ids.size() == tokens && plain.graphs == graphs && plain_p.seen.size() == graphs,
                       what + ": the plain decoding makes " + std::to_string(tokens) + " predictions");
        w.tally.expect(std::all_of(plain.ids.begin(), plain.ids.end(), [](int32_t id) { return id == 2; }),
                       what + ": every prediction emits token 2");

        TdtPrefixCache cache;
        TdtPrefixCache::Reading cold_r = cache.begin(World::samples_of((int64_t) tokens));
        const Decoding cold = w.cached(rows, cold_r, cold_p);
        w.tally.expect(same(plain, cold) && plain_p.seen == cold_p.seen, what + ": the cold run agrees");
        w.tally.expect(cold_r.complete(), what + ": the cold run finishes");
        cache.publish(std::move(cold_r));

        TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of((int64_t) tokens));
        const Decoding warm = w.cached(rows, warm_r, warm_p);
        w.tally.expect(same(plain, warm) && plain_p.seen == warm_p.seen, what + ": the warm run agrees");
        return std::pair<size_t, size_t>(cache.entries().size(), warm_r.hits());
    };

    // Thirty-one frames of two predictions each, then one whose duration 1 ends the audio before a second prediction
    // on it: sixty-three predictions, one below the cap.
    std::vector<float> below = tiled(choice(2, 0), 31);
    const std::vector<float> ends = choice(2, 1);
    below.insert(below.end(), ends.begin(), ends.end());
    const auto below_kept = run(below, 63, 63, "the cap at 63");
    w.tally.expect(below_kept.first == 63 && below_kept.second == 63,
                   "the cap at 63: every prediction below the cap is recorded and reused");

    const std::vector<float> at = tiled(choice(2, 0), 32);
    const auto at_kept = run(at, 64, 64, "the cap at 64");
    w.tally.expect(at_kept.first == TdtPrefixCache::kEntries && at_kept.second == TdtPrefixCache::kEntries,
                   "the cap at 64: the cap's worth is recorded and reused");

    const std::vector<float> past = tiled(choice(2, 0), 33);
    const auto past_kept = run(past, 66, 66, "the cap past 64");
    w.tally.expect(past_kept.first == TdtPrefixCache::kEntries && past_kept.second == TdtPrefixCache::kEntries,
                   "the cap past 64: only the cap's worth is recorded and reused");
}

/**
 * A decoding cancelled before its first prediction and after some: the candidate publishes nothing, the published
 * prefix survives whole, and the next request reuses all of it. The cancelled runs with and without the cache stop at
 * the same points with the same tokens.
 */
inline void cancellation(const World & w) {
    const std::vector<float> rows = stack({tied(), tied()});
    TdtPrefixCache cache;
    {
        Progress p;
        TdtPrefixCache::Reading reading = cache.begin(World::samples_of(2));
        w.cached(rows, reading, p);
        cache.publish(std::move(reading));
    }
    w.tally.expect(cache.entries().size() == 4, "the cancellation: four entries published first");

    for (const size_t stop : {size_t(1), size_t(3)}) {
        Progress plain_stop, cached_stop;
        plain_stop.stop_after = cached_stop.stop_after = stop;
        const Decoding plain = w.plain(rows, plain_stop);
        TdtPrefixCache::Reading reading = cache.begin(World::samples_of(2));
        const Decoding cached = w.cached(rows, reading, cached_stop);
        w.tally.expect(same(plain, cached) && plain_stop.seen.size() == stop && cached_stop.seen.size() == stop,
                       "the cancellation at " + std::to_string(stop) + ": both runs stop at the same point");
        w.tally.expect(!reading.complete(), "the cancellation at " + std::to_string(stop) + ": the candidate is incomplete");
        bool refused = false;
        try {
            cache.publish(std::move(reading));
        } catch (const std::logic_error &) {
            refused = true;
        }
        w.tally.expect(refused, "the cancellation at " + std::to_string(stop) + ": the candidate is refused");
        w.tally.expect(cache.entries().size() == 4,
                       "the cancellation at " + std::to_string(stop) + ": the published prefix survives");
    }

    Progress plain_p, warm_p;
    const Decoding plain = w.plain(rows, plain_p);
    TdtPrefixCache::Reading warm_r = cache.begin(World::samples_of(2));
    const Decoding warm = w.cached(rows, warm_r, warm_p);
    w.tally.expect(same(plain, warm) && warm_r.hits() == 4,
                   "the cancellation: the next request reuses the whole surviving prefix");
}

/** Empty input publishes an empty prefix, and a request that announces shorter audio drops the published prefix whole
 * and decodes as if no cache existed. */
inline void empty_and_reset(const World & w) {
    const std::vector<float> rows = stack({tied(), tied()});
    TdtPrefixCache cache;
    {
        Progress p;
        TdtPrefixCache::Reading reading = cache.begin(World::samples_of(2));
        w.cached(rows, reading, p);
        cache.publish(std::move(reading));
    }
    w.tally.expect(cache.entries().size() == 4, "the reset: four entries published first");

    Progress empty_p;
    TdtPrefixCache::Reading empty_r = cache.begin(0);
    const Decoding empty = w.cached(std::vector<float>(), empty_r, empty_p);
    w.tally.decoding_is(empty, {}, {}, {}, 0, "the reset: the empty input's decoding");
    w.tally.expect(empty_p.seen.empty() && empty_r.complete() && empty_r.hits() == 0,
                   "the reset: the empty input reports nothing and reuses nothing");
    cache.publish(std::move(empty_r));
    w.tally.expect(cache.entries().empty(), "the reset: the empty input publishes no entry");

    {
        Progress p;
        TdtPrefixCache::Reading reading = cache.begin(World::samples_of(2));
        w.cached(rows, reading, p);
        cache.publish(std::move(reading));
    }
    Progress plain_p, dropped_p;
    const Decoding plain = w.plain(rows, plain_p);
    TdtPrefixCache::Reading dropped = cache.begin(World::samples_of(2) - 1);
    w.tally.expect(cache.entries().empty(), "the reset: shorter announced audio drops the published prefix");
    const Decoding after = w.cached(rows, dropped, dropped_p);
    w.tally.expect(same(plain, after) && dropped.hits() == 0,
                   "the reset: the decoding after the drop reuses nothing and agrees");
}

/**
 * Runs every controlled case on the synthetic weights written into `work_dir`, on `backend`. The cases are a fixed
 * table; one that throws counts as failed, and the summary names how many of the table ran, so a case cannot go
 * missing in silence.
 */
inline bool run_cases(const std::string & work_dir, ggml_backend_t backend) {
    std::filesystem::create_directories(std::filesystem::u8path(work_dir));
    const std::string file = (std::filesystem::u8path(work_dir) / kFixtureName).u8string();
    write_fixture(file);
    ModelFile model(file, backend, kLayout);
    PredictionNetwork prediction(model);
    Joint joint(model);
    const TdtDecoder decoder(model, prediction, joint);

    Tally tally;
    const World world{decoder, prediction, joint, backend, tally};
    const std::vector<std::pair<const char *, void (*)(const World &)>> cases = {
        {"the initial blank", initial_blank},
        {"the blank chain with the duration-0 correction", blank_chain},
        {"first-argmax ties and the max-symbols frame advance", ties_and_max_symbols},
        {"the last token emitted and never fed", last_token_never_fed},
        {"the first divergence that later agreement does not undo", divergence_is_final},
        {"changed frames altering the fresh joint decisions", changed_frames},
        {"the entry cap at 63, 64 and more predictions", entry_cap},
        {"cancellation before the first prediction and after some", cancellation},
        {"empty input and the reset of a new request", empty_and_reset},
    };
    size_t ran = 0;
    for (const auto & [name, body] : cases) {
        const size_t before = tally.failed;
        bool threw = false;
        try {
            body(world);
        } catch (const std::exception & e) {
            threw = true;
            std::printf("    FAIL: %s: %s\n", name, e.what());
            tally.count(false);
        }
        ran++;
        std::printf("  %s: %s\n", name, threw || tally.failed != before ? "FAIL" : "ok");
    }
    std::printf("synthetic cases: %zu of %zu ran, %zu failed\n", ran, cases.size(), tally.failed);
    return ran == cases.size() && tally.failed == 0;
}

}  // namespace tdt_prefix_check
