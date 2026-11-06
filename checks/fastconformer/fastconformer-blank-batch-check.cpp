// Compares scalar and four-frame blank joints, then their complete TDT paths, progress and cancellation points.
// Strict logits equality and decision equality are reported independently; either failure makes the check fail.
// Synthetic KDA weights exercise duration jumps, skipped frames and prediction changes without official references.
//
// usage: fastconformer-blank-batch-check <model.gguf> [cpu|gpu|device name] [threads]
//        fastconformer-blank-batch-check --synthetic <work dir> [cpu|gpu|device name] [threads]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "fastconformer/tdt.h"
#include "ggml-cpu.h"
#include "state-fixture.h"

using namespace fastconformer;

namespace {

struct Checks {
    size_t total = 0, failed = 0, logits_cases = 0, logits_differ = 0, argmax_differ = 0;

    void expect(bool ok, const std::string & what) {
        total++;
        if (!ok) {
            failed++;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    void logits(const std::vector<float> & got, const std::vector<float> & want, int outputs, int blank,
                int durations, const std::string & name) {
        const bool finite = std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(want.begin(), want.end(), [](float x) { return std::isfinite(x); });
        const bool shape = got.size() == want.size();
        const bool exact = finite && shape && std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0;
        bool decisions = finite && shape;
        for (size_t at = 0; decisions && at < got.size(); at += (size_t) outputs) {
            decisions = first_argmax(got.data() + at, blank + 1) == first_argmax(want.data() + at, blank + 1) &&
                        first_argmax(got.data() + at + blank + 1, durations) ==
                            first_argmax(want.data() + at + blank + 1, durations);
        }
        logits_cases++;
        logits_differ += !exact;
        argmax_differ += !decisions;
        expect(exact, name + " strict logits equality");
        expect(decisions, name + " label and duration argmax");
        std::printf("%s: strict_logits=%s, argmax=%s\n", name.c_str(), exact ? "equal" : "DIFFER", decisions ? "equal" : "DIFFER");
        print_diff("  scalar/batched logits", compare(got, want));
    }
};

std::vector<float> frames(int hidden, int count, int seed) {
    std::vector<float> out((size_t) hidden * count);
    for (size_t i = 0; i < out.size(); i++) out[i] = std::sin(0.031f * (float) (i + 37 * seed));
    return out;
}

/** Checks the batch sizes of a full block and every shorter tail with predictions from successive KDA states. */
void joint_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                 ggml_backend_t backend, Checks & checks) {
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    if (!allocator) throw std::runtime_error("cannot create the joint-check allocator");
    const int hidden = joint.hidden(), outputs = joint.outputs();
    const int blank = (int) model.u32("fastconformer.decoder.blank_id");
    const int durations = (int) model.i32_array("fastconformer.decoder.tdt.durations").size();
    DecoderStep step(prediction, joint, backend, 1);
    PredictionState state = prediction.initial_state();
    for (int iteration = 0; iteration < 4; iteration++) {
        step.compute(iteration ? (7 * iteration) % blank : blank, state, frames(hidden, 1, iteration));
        state = step.state();
        const auto predicted = step.prediction();
        for (int count : {1, 2, 3, 4}) {
            const auto projected = frames(hidden, count, 41 * iteration + count);
            Graph batch(64);
            auto * out = joint.build(batch.ctx(), batch.input(projected, hidden, count), batch.input(predicted, hidden));
            batch.output(out);
            batch.compute(backend, allocator.get());
            const auto got = Graph::read(out);
            std::vector<float> want;
            for (int i = 0; i < count; i++) {
                const std::vector<float> frame(projected.begin() + (size_t) i * hidden, projected.begin() + (size_t) (i + 1) * hidden);
                Graph scalar(64);
                auto * single = joint.build(scalar.ctx(), scalar.input(frame, hidden), scalar.input(predicted, hidden));
                scalar.output(single);
                scalar.compute(backend, allocator.get());
                const auto values = Graph::read(single);
                want.insert(want.end(), values.begin(), values.end());
            }
            checks.logits(got, want, outputs, blank, durations,
                          "prediction " + std::to_string(iteration) + ", batch " + std::to_string(count));
        }
    }
}

/** Compares the emitted path and the callback sequence even when the callback stops a request part way through. */
Decoding decode_pair(const TdtDecoder & scalar, const TdtDecoder & batched, ggml_backend_t backend,
                     const std::vector<float> & projected, size_t allowed, const std::string & name, Checks & checks,
                     const Boost * boost = nullptr) {
    std::vector<double> a_progress, b_progress;
    auto a_callback = [&](double done) { a_progress.push_back(done); return a_progress.size() <= allowed; };
    auto b_callback = [&](double done) { b_progress.push_back(done); return b_progress.size() <= allowed; };
    const auto a = boost ? scalar.decode(projected, backend, a_callback, *boost) : scalar.decode(projected, backend, a_callback);
    const auto b = boost ? batched.decode(projected, backend, b_callback, *boost) : batched.decode(projected, backend, b_callback);
    checks.expect(a.ids == b.ids && a.frames == b.frames && a.durations == b.durations, name + " token IDs, frames and durations");
    checks.expect(a_progress == b_progress, name + " progress and cancellation points");
    checks.expect(a.blank_rows_computed == a.blank_rows_used && b.blank_rows_used == a.blank_rows_used &&
                      b.blank_rows_computed >= b.blank_rows_used && b.graphs <= a.graphs,
                  name + " consumed rows and graph counts");
    std::printf("%s: scalar graphs=%zu blank_rows=%zu/%zu, batched graphs=%zu blank_rows=%zu/%zu unused=%zu\n",
                name.c_str(), a.graphs, a.blank_rows_used, a.blank_rows_computed,
                b.graphs, b.blank_rows_used, b.blank_rows_computed, b.blank_rows_computed - b.blank_rows_used);
    return b;
}

/** Makes the fixture's identity joint choose each specified label and duration with a margin over the prediction. */
std::vector<float> controlled(const std::vector<int> & labels, const std::vector<int> & durations, int hidden) {
    if (labels.size() != durations.size()) throw std::logic_error("controlled labels and durations differ in length");
    std::vector<float> projected(labels.size() * (size_t) hidden, 0.0f);
    for (size_t i = 0; i < labels.size(); i++) {
        projected[i * hidden + labels[i]] = 100.0f;
        projected[i * hidden + state_check::kBlank + 1 + durations[i]] = 50.0f;
    }
    return projected;
}

void decoding_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                    ggml_backend_t backend, bool synthetic, Checks & checks) {
    const TdtDecoder scalar(model, prediction, joint, false), batched(model, prediction, joint, true);
    const size_t unlimited = std::numeric_limits<size_t>::max();
    decode_pair(scalar, batched, backend, {}, unlimited, "empty projected input", checks);
    for (int count : {1, 2, 3, 4, 5, 9, 19}) {
        const auto projected = frames(joint.hidden(), count, count);
        for (size_t allowed : {size_t{0}, size_t{1}, size_t{2}, unlimited}) {
            decode_pair(scalar, batched, backend, projected, allowed,
                        "frames " + std::to_string(count) + ", callbacks " + std::to_string(allowed), checks);
        }
        decode_pair(scalar, batched, backend, projected, unlimited, "request after cancellation", checks);
    }
    const BoostingTree tree({{3, 4}, {5, 6}}, scalar.blank(), 10.0, 1.0, 0.0f);
    const Boost boost{tree, 20.0f};
    decode_pair(scalar, batched, backend, frames(joint.hidden(), 9, 23), unlimited, "boosted random frames", checks, &boost);
    if (!synthetic) return;

    const int blank = state_check::kBlank;
    for (int count : {1, 2, 3, 4, 5, 6, 10}) {
        const auto projected = controlled(std::vector<int>((size_t) count, blank), std::vector<int>((size_t) count, 0), joint.hidden());
        const auto decoded = decode_pair(scalar, batched, backend, projected, unlimited,
                                         "zero-duration blank chain " + std::to_string(count), checks);
        checks.expect(decoded.ids.empty() && decoded.graphs == 1 + (size_t) (count - 1 + 3) / 4 &&
                          decoded.blank_rows_computed == (size_t) count - 1 && decoded.blank_rows_used == (size_t) count - 1,
                      "blank duration zero advances one and the final batch contains only the remaining rows");
    }

    const auto within = controlled({1, blank, blank, 9, 2, blank}, {0, 0, 2, 1, 1, 4}, joint.hidden());
    const auto within_result = decode_pair(scalar, batched, backend, within, unlimited, "duration skips within a batch", checks);
    checks.expect(within_result.ids == std::vector<int32_t>({1, 1, 2}) &&
                      within_result.frames == std::vector<int64_t>({0, 0, 4}) && within_result.durations == std::vector<int32_t>({0, 0, 1}) &&
                      within_result.graphs == 5 && within_result.blank_rows_computed == 4 && within_result.blank_rows_used == 2,
                  "a skipped future token is not consumed");

    const auto forced = controlled({blank, 1, 2, blank, 3, blank}, {1, 0, 1, 1, 1, 1}, joint.hidden());
    const auto stopped = decode_pair(scalar, batched, backend, forced, 2, "cancel after a forced advance", checks);
    checks.expect(stopped.ids == std::vector<int32_t>({1, 1}) && stopped.frames == std::vector<int64_t>({1, 1}) &&
                      stopped.graphs == 3 && stopped.blank_rows_computed == 4 && stopped.blank_rows_used == 1,
                  "cancellation stops before the prediction after max_symbols");
    const auto advanced = decode_pair(scalar, batched, backend, forced, unlimited, "forced advance discards future logits", checks);
    checks.expect(advanced.ids == std::vector<int32_t>({1, 1, 2, 3}) && advanced.frames == std::vector<int64_t>({1, 1, 2, 4}) &&
                      advanced.durations == std::vector<int32_t>({0, 0, 1, 1}) && advanced.graphs == 7 &&
                      advanced.blank_rows_computed == 6 && advanced.blank_rows_used == 2,
                  "a nonblank invalidates the batch before duration zero and max_symbols change the prediction");

    const auto beyond = controlled({blank, blank, 9, 9, 9, 2, blank, blank, 3, blank}, {1, 4, 0, 0, 0, 1, 0, 1, 4, 1}, joint.hidden());
    const auto jumped = decode_pair(scalar, batched, backend, beyond, unlimited, "duration skips beyond a batch", checks);
    checks.expect(jumped.ids == std::vector<int32_t>({2, 3}) && jumped.frames == std::vector<int64_t>({5, 8}) &&
                      jumped.durations == std::vector<int32_t>({1, 4}) && jumped.graphs == 5 &&
                      jumped.blank_rows_computed == 11 && jumped.blank_rows_used == 4,
                  "a duration beyond the cached range starts a new batch at the reached frame");
    const auto boosted = decode_pair(scalar, batched, backend, beyond, unlimited, "boost follows only consumed labels", checks, &boost);
    checks.expect(boosted.ids != jumped.ids && boosted.frames == jumped.frames && boosted.durations == jumped.durations,
                  "boost changes emitted tokens without changing the unboosted blank decisions");
    const auto reset = decode_pair(scalar, batched, backend, beyond, unlimited, "request after boosted decoding", checks);
    checks.expect(reset.ids == jumped.ids && reset.frames == jumped.frames && reset.durations == jumped.durations,
                  "speculative rows and keyword state do not cross requests");

    const auto end = decode_pair(scalar, batched, backend, controlled({3}, {4}, joint.hidden()), unlimited, "final token", checks);
    checks.expect(end.ids == std::vector<int32_t>({3}) && end.frames == std::vector<int64_t>({0}) &&
                      end.durations == std::vector<int32_t>({4}) && end.graphs == 1 && end.blank_rows_computed == 0,
                  "a token whose duration passes the last frame is emitted without speculative work");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    const bool synthetic = args.size() > 1 && args[1] == "--synthetic";
    const size_t device_at = synthetic ? 3 : 2;
    if (args.size() < device_at || args.size() > device_at + 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [device] [threads]\n       %s --synthetic <work dir> [device] [threads]\n",
                     args[0].c_str(), args[0].c_str());
        return 2;
    }
    try {
        const std::string device = args.size() > device_at ? args[device_at] : "cpu";
        const int threads = args.size() > device_at + 1 ? std::stoi(args[device_at + 1]) : 4;
        if (threads < 1) throw std::runtime_error("threads must be positive");
        std::string path = args[1];
        if (synthetic) {
            const auto work = std::filesystem::u8path(args[2]);
            std::filesystem::create_directories(work);
            path = (work / "synthetic-kda-state-F32.gguf").u8string();
            state_check::write_fixture(path);
        }
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(nullptr, ggml_threadpool_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(device), threads), ggml_backend_free);
        if (ggml_backend_is_cpu(backend.get())) {
            auto params = ggml_threadpool_params_default(threads);
            pool.reset(ggml_threadpool_new(&params));
            if (!pool) throw std::runtime_error("cannot create the check's CPU thread pool");
            ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        }
        const Layout & selected = synthetic ? state_check::kLayout : layout;
        const ModelFile model(path, backend.get(), selected, decoding_tensor);
        if (model.str("fastconformer.decoder.prediction") != "kda" || model.str("fastconformer.decoder.kind") != "tdt") {
            throw std::runtime_error("the blank-batch check needs a KDA TDT model");
        }
        const PredictionNetwork prediction(model);
        const Joint joint(model);
        std::map<ggml_type, size_t> types;
        for (const auto & spec : selected.tensors(model)) {
            if (decoding_tensor(spec.name)) types[model.tensor(spec.name)->type]++;
        }
        std::printf("backend: %s, threads: %d, synthetic: %s, blank batch: 4\n", ggml_backend_name(backend.get()), threads, synthetic ? "yes" : "no");
        for (const auto & entry : types) std::printf("decoder weights: %s, %zu tensors\n", tensor_type_text(entry.first).c_str(), entry.second);
        std::printf("Official NeMo/FLA reference: not checked\n");
        Checks checks;
        joint_cases(model, prediction, joint, backend.get(), checks);
        decoding_cases(model, prediction, joint, backend.get(), synthetic, checks);
        std::printf("blank batching: %zu checks, %zu failed; strict logits differ in %zu/%zu cases; argmax differs in %zu/%zu cases\n",
                    checks.total, checks.failed, checks.logits_differ, checks.logits_cases, checks.argmax_differ, checks.logits_cases);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
