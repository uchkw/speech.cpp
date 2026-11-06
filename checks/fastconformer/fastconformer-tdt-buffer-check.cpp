// Compares pointer inputs and reusable host outputs with the vector path, bit for bit, including Graph's immediate
// ownership, size/type rejection, KDA states, all joint logits, plain/boosted TDT paths, cancellation and request reset.
// Controlled synthetic F32 cases check blank chains and duration zero. This is a C++ differential check, not an
// official NeMo/FLA reference check. Optional joint_encoded.npy inputs exercise recorded encoder projections.
//
// usage: fastconformer-tdt-buffer-check <model.gguf> <cpu|gpu|device name> <threads> [joint_encoded.npy...]
//        fastconformer-tdt-buffer-check --synthetic <work dir> <cpu|gpu|device name> <threads>
//        fastconformer-tdt-buffer-check --graph-only <cpu|gpu|device name> <threads>

#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "graph-buffer-cases.h"
#include "npy.h"

using namespace fastconformer;
using namespace buffer_check;

namespace {

/** Reads the same steps through both interfaces after overwriting every reusable output with unrelated data. */
void step_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                ggml_backend_t backend, Checks & checks) {
    DecoderStep vector_step(prediction, joint, backend, 1), pointer_step(prediction, joint, backend, 1);
    PredictionState vector_state = prediction.initial_state(), pointer_state = prediction.initial_state();
    std::vector<float> reused_prediction, reused_logits;
    const int blank = (int) model.u32("fastconformer.decoder.blank_id");
    const int duration_count = (int) model.i32_array("fastconformer.decoder.tdt.durations").size();
    const BoostingTree tree({{1, 2}, {3, 4}}, blank, 10.0, 1.0, 0.0f);
    std::vector<float> bonus;
    std::vector<int32_t> reached;
    tree.advance(BoostingTree::kStart, bonus, reached);
    for (int i = 0; i < 16; i++) {
        const auto projected = frames(joint.hidden(), 1, i + 11);
        const int32_t label = i ? (7 * i) % blank : blank;
        vector_step.compute(label, vector_state, projected);
        pointer_step.compute(label, pointer_state, projected.data(), projected.size());
        vector_state = vector_step.state();
        pointer_state = pointer_step.state();
        checks.equal(pointer_state, vector_state, "complete KDA state is unchanged");
        const auto want_prediction = vector_step.prediction();
        std::vector<float> want_logits;
        vector_step.logits(want_logits);
        const float * prediction_memory = reused_prediction.data();
        const float * logits_memory = reused_logits.data();
        pointer_step.prediction(reused_prediction);
        pointer_step.logits(reused_logits);
        checks.equal(reused_prediction, want_prediction, "all projected prediction values are overwritten");
        checks.equal(reused_logits, want_logits, "all token, blank and duration logits are overwritten");
        if (i) checks.expect(reused_prediction.data() == prediction_memory && reused_logits.data() == logits_memory,
                             "prediction and logits keep their allocated memory");
        checks.expect(first_argmax(reused_logits.data(), blank + 1) == first_argmax(want_logits.data(), blank + 1) &&
                      first_argmax(reused_logits.data() + blank + 1, duration_count) ==
                      first_argmax(want_logits.data() + blank + 1, duration_count), "label and duration argmax are unchanged");
        for (int token = 0; token < blank; token++) reused_logits[(size_t) token] += 20.0f * bonus[(size_t) token];
        std::fill(reused_logits.begin() + blank, reused_logits.end(), -999.0f);
        std::fill(reused_prediction.begin(), reused_prediction.end(), 999.0f);
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    const bool synthetic = args.size() > 1 && args[1] == "--synthetic";
    const bool graph_only = args.size() > 1 && args[1] == "--graph-only";
    const size_t device_at = synthetic ? 3 : 2;
    if (args.size() < device_at + 2 || ((synthetic || graph_only) && args.size() != device_at + 2)) {
        std::fprintf(stderr, "usage: %s <model.gguf> <device> <threads> [joint_encoded.npy...]\n"
                             "       %s --synthetic <work dir> <device> <threads>\n"
                             "       %s --graph-only <device> <threads>\n", args[0].c_str(), args[0].c_str(), args[0].c_str());
        return 2;
    }
    try {
        const int threads = std::stoi(args[device_at + 1]);
        if (threads < 1) throw std::runtime_error("threads must be positive");
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(nullptr, ggml_threadpool_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            start_device(find_device(args[device_at]), threads), ggml_backend_free);
        if (ggml_backend_is_cpu(backend.get())) {
            auto params = ggml_threadpool_params_default(threads);
            pool.reset(ggml_threadpool_new(&params));
            if (!pool) throw std::runtime_error("cannot create the check's CPU thread pool");
            ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        }
        std::printf("backend: %s, threads: %d, synthetic: %s\n", ggml_backend_name(backend.get()), threads, synthetic ? "yes" : "no");
        std::printf("Official NeMo/FLA reference: not checked\n");
        Checks checks;
        graph_cases(backend.get(), checks);
        if (!graph_only) {
            std::string path = args[1];
            if (synthetic) {
                const auto work = std::filesystem::u8path(args[2]);
                std::filesystem::create_directories(work);
                path = (work / "synthetic-kda-state-F32.gguf").u8string();
                state_check::write_fixture(path);
            }
            const Layout & selected = synthetic ? state_check::kLayout : layout;
            const ModelFile model(path, backend.get(), selected, decoding_tensor);
            if (model.str("fastconformer.decoder.prediction") != "kda") throw std::runtime_error("the buffer experiment requires KDA");
            const PredictionNetwork prediction(model);
            const Joint joint(model);
            std::map<ggml_type, size_t> types;
            for (const auto & spec : selected.tensors(model)) {
                if (decoding_tensor(spec.name)) types[model.tensor(spec.name)->type]++;
            }
            for (const auto & entry : types) std::printf("decoder weights: %s, %zu tensors\n", tensor_type_text(entry.first).c_str(), entry.second);
            step_cases(model, prediction, joint, backend.get(), checks);
            decoding_cases(model, prediction, joint, backend.get(), synthetic, checks);
            const TdtDecoder original(model, prediction, joint, false), reused(model, prediction, joint, true);
            const BoostingTree tree({{1, 2}, {3, 4}}, original.blank(), 10.0, 1.0, 0.0f);
            const Boost boost{tree, 20.0f};
            for (size_t i = device_at + 2; i < args.size(); i++) {
                const auto projected = read_npy(args[i]);
                if (projected.shape.size() != 2 || projected.shape[1] != joint.hidden() || projected.f32.empty()) {
                    throw std::runtime_error(args[i] + " must contain nonempty float32 projected frames [T, joint hidden]");
                }
                std::printf("projected input: %s\n", args[i].c_str());
                for (size_t allowed : {size_t(0), size_t(2), std::numeric_limits<size_t>::max()}) {
                    decode_pair(original, reused, backend.get(), projected.f32, allowed, checks);
                    decode_pair(original, reused, backend.get(), projected.f32, allowed, checks, &boost);
                }
            }
        }
        std::printf("host-buffer reuse: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
