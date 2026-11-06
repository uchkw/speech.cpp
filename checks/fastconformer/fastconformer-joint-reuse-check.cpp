// Compares a request's reused blank-joint graph with fresh graphs, including interleaved prediction steps and
// changing prediction vectors. Controlled TDT frames check duration zero, blank chains, cancellation and reset.
// The synthetic F32 fixture is a differential check, not an official NeMo or FLA reference.
//
// usage: fastconformer-joint-reuse-check <model.gguf> [cpu|gpu|device name] [threads]
//        fastconformer-joint-reuse-check --synthetic <work dir> [cpu|gpu|device name] [threads]

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "tdt-cache-cases.h"

using namespace fastconformer;
using namespace cache_check;

namespace {

void joint_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                 ggml_backend_t backend, Checks & checks) {
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> fresh_allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    if (!fresh_allocator) throw std::runtime_error("cannot create the fresh-joint allocator");
    DecoderStep prediction_a(prediction, joint, backend, 1), prediction_b(prediction, joint, backend, 1);
    JointStep reused_a(joint, backend), reused_b(joint, backend);
    PredictionState state_a = prediction.initial_state(), state_b = prediction.initial_state();
    const int blank = (int) model.u32("fastconformer.decoder.blank_id");
    const int duration_count = (int) model.i32_array("fastconformer.decoder.tdt.durations").size();
    size_t products = 0;
    for (int i = 0; i < 16; i++) {
        prediction_a.compute(i ? (7 * i) % blank : blank, state_a, frames(joint.hidden(), 1, i));
        prediction_b.compute(i ? (11 * i + 1) % blank : blank, state_b, frames(joint.hidden(), 1, i + 29));
        state_a = prediction_a.state();
        state_b = prediction_b.state();
        const std::vector<float> predicted[] = {prediction_a.prediction(), prediction_b.prediction()};
        JointStep * reused[] = {&reused_a, &reused_b};
        for (int frame = 0; frame < 4; frame++) {
            for (int request = 0; request < 2; request++) {
                const auto projected = frames(joint.hidden(), 1, 41 * i + frame + 19 * request);
                Graph fresh(64);
                ggml_tensor * logits = joint.build(fresh.ctx(), fresh.input(projected, joint.hidden()),
                                                   fresh.input(predicted[request], joint.hidden()));
                fresh.output(logits);
                fresh.compute(backend, fresh_allocator.get());
                const auto want = Graph::read(logits);
                reused[request]->compute(projected, predicted[request]);
                const auto got = reused[request]->logits();
                checks.equal(got, want, "all reused blank-joint logits");
                checks.expect(first_argmax(got.data(), blank + 1) == first_argmax(want.data(), blank + 1) &&
                              first_argmax(got.data() + blank + 1, duration_count) ==
                              first_argmax(want.data() + blank + 1, duration_count), "joint label and duration argmax");
                products++;
            }
        }
    }
    std::printf("fresh/reused joint evaluations: %zu, interleaved requests: 2\n", products);
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
        if (ggml_backend_is_cpu(backend.get())) {
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
        std::printf("backend: %s, threads: %d, synthetic: %s\n", ggml_backend_name(backend.get()), threads, synthetic ? "yes" : "no");
        for (const auto & entry : types) std::printf("decoder weights: %s, %zu tensors\n", tensor_type_text(entry.first).c_str(), entry.second);
        std::printf("Official NeMo/FLA reference: not checked\n");
        Checks checks;
        joint_cases(model, prediction, joint, backend.get(), checks);
        decoding_cases(model, prediction, joint, backend.get(), synthetic, checks);
        std::printf("joint graph reuse: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
