// Compares request-owned KDA state with the host-state path on one backend, bit for bit after each prediction step.
// Interleaved requests, full recurrent and convolution histories, repeated decodings and cancellation check state
// transport and lifetime. The synthetic F32 fixture controls TDT blanks and durations without official weights or
// dumps; it is a differential regression check, not a check against NeMo or FLA.
//
// usage: fastconformer-state-check <model.gguf> [cpu|gpu|device name] [threads]
//        fastconformer-state-check --synthetic <work dir> [cpu|gpu|device name] [threads]

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
#include "fastconformer/rnnt-greedy.h"
#include "fastconformer/tdt.h"
#include "ggml-cpu.h"
#include "state-fixture.h"

using namespace fastconformer;

namespace {

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & what) {
        total++;
        if (!ok) {
            failed++;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    void equal(const std::vector<float> & got, const std::vector<float> & want, const std::string & what) {
        const bool finite = std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(want.begin(), want.end(), [](float x) { return std::isfinite(x); });
        const bool same = got.size() == want.size() && (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
        expect(finite && same, what);
        if (!finite || !same) print_diff(what, compare(got, want));
    }
};

std::vector<float> frames(int hidden, int count, int seed) {
    std::vector<float> out((size_t) hidden * count);
    for (size_t i = 0; i < out.size(); i++) out[i] = std::sin(0.031f * (float) (i + 37 * seed));
    return out;
}

/** Two independent step graphs, whose resident state is read only for the comparison. */
struct Steps {
    DecoderState stored;
    DecoderStep host, resident;
    PredictionState state;
    std::vector<float> prediction_output, joint_logits;
    int hidden;
    size_t changed = 0;

    Steps(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend, Checks & checks)
        : stored(prediction, backend), host(prediction, joint, backend, 1),
          resident(prediction, joint, backend, 1, &stored), state(prediction.initial_state()), hidden(joint.hidden()) {
        checks.equal(Graph::read(stored.tensor()), state, "new request state is zero");
    }

    void advance(int32_t label, int seed, Checks & checks) {
        const auto frame = frames(hidden, 1, seed);
        host.compute(label, state, frame);
        resident.compute(label, frame);
        const auto next = host.state();
        checks.equal(resident.state(), next, "prediction next state");
        checks.equal(Graph::read(stored.tensor()), next, "committed resident state, all elements");
        prediction_output = host.prediction();
        checks.equal(resident.prediction(), prediction_output, "prediction output after state commit");
        std::vector<float> a, b;
        host.logits(a);
        resident.logits(b);
        checks.equal(b, a, "joint logits after state commit");
        joint_logits = a;
        changed += std::memcmp(next.data(), state.data(), next.size() * sizeof(float)) != 0;
        state = next;
    }
};

void active_parts(const ModelFile & model, const PredictionState & state, Checks & checks) {
    const int64_t head = model.width("pred.kda.0.o_norm.weight", 0);
    const int64_t width = head * model.u32("fastconformer.decoder.kda.num_heads");
    const int64_t recurrent = width * head, history = width * (model.width("pred.kda.0.q_conv.weight", 0) - 1);
    const int layers = (int) model.u32("fastconformer.decoder.prediction_layers");
    size_t offset = 0;
    for (int layer = 0; layer < layers; layer++) {
        for (int64_t count : {recurrent, history, history, history}) {
            if (count) {
                const bool active = std::any_of(state.begin() + offset, state.begin() + offset + count, [](float x) { return x != 0.0f; });
                checks.expect(active, "recurrent and each convolution history are exercised");
            }
            offset += (size_t) count;
        }
    }
    checks.expect(offset == state.size(), "all state slices are covered");
}

/** Compares the complete token path and the exact points at which cancellation can be observed. */
Decoding decode_pair(const TdtDecoder & host, const TdtDecoder & resident, ggml_backend_t backend,
                     const std::vector<float> & projected, size_t allowed, Checks & checks, const Boost * boost = nullptr) {
    std::vector<double> a_progress, b_progress;
    auto a_callback = [&](double done) { a_progress.push_back(done); return a_progress.size() <= allowed; };
    auto b_callback = [&](double done) { b_progress.push_back(done); return b_progress.size() <= allowed; };
    const auto a = boost ? host.decode(projected, backend, a_callback, *boost) : host.decode(projected, backend, a_callback);
    const auto b = boost ? resident.decode(projected, backend, b_callback, *boost) : resident.decode(projected, backend, b_callback);
    checks.expect(a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs,
                  "TDT tokens, frames, durations and graph count");
    checks.expect(a_progress == b_progress, "TDT progress and cancellation points");
    return b;
}

void decoding_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                    ggml_backend_t backend, bool synthetic, Checks & checks) {
    const TdtDecoder host(model, prediction, joint, false), resident(model, prediction, joint, true);
    const size_t unlimited = std::numeric_limits<size_t>::max();
    decode_pair(host, resident, backend, {}, unlimited, checks);
    for (int count : {1, 7, 19}) {
        const auto projected = frames(joint.hidden(), count, count);
        decode_pair(host, resident, backend, projected, 0, checks);
        decode_pair(host, resident, backend, projected, 1, checks);
        decode_pair(host, resident, backend, projected, unlimited, checks);
        decode_pair(host, resident, backend, projected, unlimited, checks);
    }
    const BoostingTree tree({{1, 2}, {3, 4}}, host.blank(), 10.0, 1.0, 0.0f);
    const Boost boost{tree, 20.0f};
    decode_pair(host, resident, backend, frames(joint.hidden(), 7, 23), unlimited, checks, &boost);
    if (!synthetic) return;

    const int labels[] = {1, state_check::kBlank, state_check::kBlank, 9, 2, state_check::kBlank};
    const int durations[] = {0, 0, 2, 1, 1, 4};
    std::vector<float> projected((size_t) 6 * joint.hidden(), 0.0f);
    for (size_t t = 0; t < 6; t++) {
        projected[t * joint.hidden() + labels[t]] = 100.0f;
        projected[t * joint.hidden() + state_check::kBlank + 1 + durations[t]] = 50.0f;
    }
    const auto stopped = decode_pair(host, resident, backend, projected, 2, checks);
    checks.expect(stopped.ids == std::vector<int32_t>({1, 1}) && stopped.graphs == 2, "cancellation stops before the next prediction");
    const auto complete = decode_pair(host, resident, backend, projected, unlimited, checks);
    checks.expect(complete.ids == std::vector<int32_t>({1, 1, 2}) && complete.frames == std::vector<int64_t>({0, 0, 4}) &&
                  complete.durations == std::vector<int32_t>({0, 0, 1}) && complete.graphs == 6,
                  "TDT zero duration, blank chain, max symbols and final blank");
    decode_pair(host, resident, backend, projected, unlimited, checks, &boost);
    const auto repeated = decode_pair(host, resident, backend, projected, unlimited, checks);
    checks.expect(repeated.ids == complete.ids && repeated.frames == complete.frames, "request after boosted or cancelled decoding keeps the token path");

    std::vector<float> last((size_t) joint.hidden(), 0.0f);
    last[3] = 100.0f;
    last[state_check::kBlank + 1 + 4] = 50.0f;
    const auto end = decode_pair(host, resident, backend, last, unlimited, checks);
    checks.expect(end.ids == std::vector<int32_t>({3}) && end.durations == std::vector<int32_t>({4}), "token whose duration passes the final frame");

    const RnntGreedyDecoder rnnt(model, prediction, joint);
    const auto r = rnnt.decode(projected, backend, {});
    checks.expect(r.ids == std::vector<int32_t>({1, 1, 9, 9, 2, 2}) &&
                  r.frames == std::vector<int64_t>({0, 0, 3, 3, 4, 4}) && r.durations.empty(), "RNN-T retains its host-state path");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    const bool synthetic = args.size() > 1 && args[1] == "--synthetic";
    const size_t device_at = synthetic ? 3 : 2;
    if (args.size() < device_at || args.size() > device_at + 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [device] [threads]\n       %s --synthetic <work dir> [device] [threads]\n", args[0].c_str(), args[0].c_str());
        return 2;
    }
    try {
        const std::string device = args.size() > device_at ? args[device_at] : "cpu";
        const int threads = args.size() > device_at + 1 ? std::stoi(args[device_at + 1]) : 1;
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
        if (model.str("fastconformer.decoder.prediction") != "kda") throw std::runtime_error("this check needs KDA prediction weights");
        const PredictionNetwork prediction(model);
        const Joint joint(model);
        std::map<ggml_type, size_t> types;
        for (const auto & spec : selected.tensors(model)) {
            if (decoding_tensor(spec.name)) types[model.tensor(spec.name)->type]++;
        }
        std::printf("backend: %s, threads: %d, state: %lld F32 values, synthetic: %s\n", ggml_backend_name(backend.get()), threads,
                    (long long) prediction.state_size(), synthetic ? "yes" : "no");
        for (const auto & entry : types) std::printf("decoder weights: %s, %zu tensors\n", tensor_type_text(entry.first).c_str(), entry.second);
        std::printf("CPU F32 differential: %s; official NeMo/FLA reference: not checked\n",
                    ggml_backend_is_cpu(backend.get()) && types.size() == 1 && types.begin()->first == GGML_TYPE_F32 ? "yes" : "no");
        Checks checks;
        {
            Steps a(prediction, joint, backend.get(), checks), b(prediction, joint, backend.get(), checks);
            const int blank = (int) model.u32("fastconformer.decoder.blank_id");
            const int count = std::max(16, (int) model.width("pred.kda.0.q_conv.weight", 0) + 4);
            for (int i = 0; i < count; i++) {
                a.advance(i == 0 ? blank : (7 * i) % blank, i, checks);
                b.advance(i == 0 ? blank : (11 * i + 1) % blank, i + 29, checks);
            }
            checks.expect(a.changed > 1 && b.changed > 1, "both interleaved request states evolve");
            checks.expect(a.state != b.state, "different label histories leave different states");
            if (synthetic) {
                a.advance(1, 55, checks);
                b.advance(1, 55, checks);
                checks.expect(a.prediction_output != b.prediction_output, "prediction depends on history with the same label");
                checks.expect(a.joint_logits != b.joint_logits, "joint logits depend on history with the same label and frame");
            }
            active_parts(model, a.state, checks);
            active_parts(model, b.state, checks);
        }
        {
            Steps reset(prediction, joint, backend.get(), checks);
            reset.advance(1, 0, checks);
        }
        decoding_cases(model, prediction, joint, backend.get(), synthetic, checks);
        std::printf("state transport and lifecycle: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
