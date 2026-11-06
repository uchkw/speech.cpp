// Compares retained CPU execution plans with the ordinary graph path after every prediction step. Interleaved
// requests and the CPU abort callback check plan state and lifetime. TDT cases check cancellation and reset.
// The synthetic F32 fixture is a differential check, not an official NeMo or FLA reference.
//
// usage: fastconformer-cpu-plan-check <model.gguf> [cpu|gpu|device name] [threads]
//        fastconformer-cpu-plan-check --synthetic <work dir> [cpu|gpu|device name] [threads]

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "error.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "tdt-cache-cases.h"

using namespace fastconformer;
using namespace cache_check;

namespace {

struct Steps {
    DecoderStep ordinary, cached;
    PredictionState state;
    int hidden;

    Steps(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend)
        : ordinary(prediction, joint, backend, 1),
          cached(prediction, joint, backend, 1, ggml_backend_is_cpu(backend)),
          state(prediction.initial_state()), hidden(joint.hidden()) {}

    void advance(int32_t label, int seed, Checks & checks) {
        const auto frame = frames(hidden, 1, seed);
        ordinary.compute(label, state, frame);
        cached.compute(label, state, frame);
        const auto next = ordinary.state();
        checks.equal(cached.state(), next, "prediction state with CPU plan reuse");
        checks.equal(cached.prediction(), ordinary.prediction(), "prediction output with CPU plan reuse");
        std::vector<float> a, b;
        ordinary.logits(a);
        cached.logits(b);
        checks.equal(b, a, "joint logits with CPU plan reuse");
        state = next;
    }
};

void abort_case(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend, Checks & checks) {
    bool abort = false;
    ggml_backend_cpu_set_abort_callback(backend, [](void * data) { return *static_cast<bool *>(data); }, &abort);
    {
        Steps steps(prediction, joint, backend);
        steps.advance(0, 17, checks);
        abort = true;
        for (DecoderStep * step : {&steps.ordinary, &steps.cached}) {
            bool stopped = false;
            try {
                step->compute(1, steps.state, frames(joint.hidden(), 1, 18));
            } catch (const Error & e) {
                stopped = e.fault() == Fault::Device;
            }
            checks.expect(stopped, "CPU abort callback stops both execution paths");
        }
        abort = false;
        steps.advance(1, 19, checks);
    }
    ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
    Steps next_request(prediction, joint, backend);
    next_request.advance(2, 20, checks);
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
        const bool cpu = ggml_backend_is_cpu(backend.get());
        if (cpu) {
            auto params = ggml_threadpool_params_default(threads);
            pool.reset(ggml_threadpool_new(&params));
            if (!pool) throw std::runtime_error("cannot create the check's CPU thread pool");
            ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        }
        const Layout & selected = synthetic ? state_check::kLayout : layout;
        const ModelFile model(path, backend.get(), selected, decoding_tensor);
        const PredictionNetwork prediction(model);
        const Joint joint(model);
        std::map<ggml_type, size_t> types;
        for (const auto & spec : selected.tensors(model)) {
            if (decoding_tensor(spec.name)) types[model.tensor(spec.name)->type]++;
        }
        std::printf("backend: %s, threads: %d, CPU plan: %s, synthetic: %s\n", ggml_backend_name(backend.get()), threads,
                    cpu ? "requested" : "not applicable", synthetic ? "yes" : "no");
        for (const auto & entry : types) std::printf("decoder weights: %s, %zu tensors\n", tensor_type_text(entry.first).c_str(), entry.second);
        std::printf("Official NeMo/FLA reference: not checked\n");
        Checks checks;
        {
            Steps a(prediction, joint, backend.get()), b(prediction, joint, backend.get());
            const int blank = (int) model.u32("fastconformer.decoder.blank_id");
            for (int i = 0; i < 16; i++) {
                a.advance(i ? (7 * i) % blank : blank, i, checks);
                b.advance(i ? (11 * i + 1) % blank : blank, i + 29, checks);
            }
            if (synthetic) {
                a.advance(1, 55, checks);
                b.advance(1, 55, checks);
                checks.expect(a.ordinary.prediction() != b.ordinary.prediction(), "prediction depends on history with the same label");
                std::vector<float> a_logits, b_logits;
                a.ordinary.logits(a_logits);
                b.ordinary.logits(b_logits);
                checks.expect(a_logits != b_logits, "joint logits depend on history with the same label and frame");
            }
        }
        if (cpu) abort_case(prediction, joint, backend.get(), checks);
        else {
            bool refused = false;
            try {
                DecoderStep invalid(prediction, joint, backend.get(), 1, true);
            } catch (const std::logic_error &) {
                refused = true;
            }
            checks.expect(refused, "an explicitly requested CPU plan refuses a non-CPU backend");
        }
        decoding_cases(model, prediction, joint, backend.get(), synthetic, checks);
        std::printf("CPU plan reuse: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
