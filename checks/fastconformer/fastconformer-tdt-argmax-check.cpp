// Compares two device argmax results with host first_argmax, preserving the source logits for independent reads.
// Raw nonfinite/subnormal cases are separate from complete KDA/TDT comparisons on synthetic or recorded inputs.
// Usage: fastconformer-tdt-argmax-check --raw <device> <threads>
//        fastconformer-tdt-argmax-check --synthetic <work dir> <device> <threads>
//        fastconformer-tdt-argmax-check <model.gguf> <device> <threads> [joint_encoded.npy...]

#include <filesystem>
#include <memory>

#include "args.h"
#include "backend.h"
#include "error.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "tdt-argmax-paths.h"

using namespace fastconformer;
using namespace argmax_check;

namespace {

/** Compares each device decision with the exact row it reduced, and every state and logit with the original graph. */
void step_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                ggml_backend_t backend, Checks & checks, const std::vector<float> & projected) {
    const int blank = (int) model.u32("fastconformer.decoder.blank_id");
    const int duration_count = (int) model.i32_array("fastconformer.decoder.tdt.durations").size();
    DecoderStep original(prediction, joint, backend, 1), selected(prediction, joint, backend, 1, blank);
    PredictionState a_state = prediction.initial_state(), b_state = prediction.initial_state();
    int32_t label = blank;
    const size_t count = projected.size() / (size_t) joint.hidden();
    for (size_t i = 0; i < count; ++i) {
        const std::vector<float> frame(projected.begin() + i * joint.hidden(), projected.begin() + (i + 1) * joint.hidden());
        original.compute(label, a_state, frame);
        selected.compute(label, b_state, frame);
        a_state = original.state();
        b_state = selected.state();
        checks.equal(b_state, a_state, "complete KDA state remains exact");
        checks.equal(selected.prediction(), original.prediction(), "projected prediction remains exact");
        std::vector<float> a_logits, b_logits;
        original.logits(a_logits);
        const auto decision = selected.decision();
        selected.logits(b_logits);
        checks.equal(b_logits, a_logits, "all retained logits remain exact");
        checks.expect(decision.label == first_argmax(b_logits.data(), blank + 1) &&
                      decision.duration_index == first_argmax(b_logits.data() + blank + 1, duration_count),
                      "device IDs equal host decisions on the same actual joint row");
        label = first_argmax(a_logits.data(), blank + 1);
    }
    std::printf("projected-input joint probes: %zu rows\n", count);
}

/** Exercises the constructor's selector without computing a graph, including metadata-only reads. */
void selector_cases(const ModelFile & model, const PredictionNetwork & prediction, const Joint & joint,
                    const std::string & path, const Layout & selected, ggml_backend_t backend, Checks & checks) {
    const bool eligible = model.str("general.name") == "Haas-ja-v1.0" && model.str("fastconformer.decoder.prediction") == "kda";
    const auto registry = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    const bool metal = std::strcmp(ggml_backend_reg_name(registry), "MTL") == 0;
    std::vector<const char *> invalid = {"yes", "2", "-1", " 1", "1 "};
#ifndef _WIN32
    invalid.push_back("");
#endif
    for (const char * value : invalid) {
        setting(value);
        const ModelFile metadata(path, selected);
        checks.expect(metadata.str("general.name") == model.str("general.name"), "metadata-only read ignores invalid experiment settings");
        bool refused = false;
        try { const TdtDecoder decoder(model, prediction, joint); }
        catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
        checks.expect(refused == eligible, std::string("selector rejects only eligible model: '") + value + "'");
    }
    for (const char * value : {static_cast<const char *>(nullptr), "0", "1"}) {
        setting(value);
        const ModelFile metadata(path, selected);
        checks.expect(metadata.str("general.name") == model.str("general.name"), "metadata-only model read ignores the selector");
        const TdtDecoder decoder(model, prediction, joint);
        checks.expect(decoder.blank() == (int) model.u32("fastconformer.decoder.blank_id"), "selector preserves the model's blank");
        const bool enabled = eligible && metal && value && std::strcmp(value, "1") == 0;
        checks.expect(decoder.uses_device_argmax(backend) == enabled, "runtime route follows model, selector and decoding backend");
        checks.expect(!decoder.uses_device_argmax(backend, true), "every boosted request retains full logits");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    const bool raw = args.size() > 1 && args[1] == "--raw";
    const bool synthetic = args.size() > 1 && args[1] == "--synthetic";
    const size_t device_at = synthetic ? 3 : 2;
    if (args.size() < device_at + 2 || ((raw || synthetic) && args.size() != device_at + 2)) {
        std::fprintf(stderr, "usage: %s --raw <device> <threads>\n"
                             "       %s --synthetic <work dir> <device> <threads>\n"
                             "       %s <model.gguf> <device> <threads> [joint_encoded.npy...]\n",
                             args[0].c_str(), args[0].c_str(), args[0].c_str());
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
        std::printf("backend: %s, threads: %d\n", ggml_backend_name(backend.get()), threads);
        Checks checks;
        if (raw) {
            contract_cases(backend.get(), checks);
            raw_cases(backend.get(), false, checks);
            std::printf("finite subtotal: %zu checks, %zu failed\n", checks.total, checks.failed);
            const size_t prior_total = checks.total, prior_failed = checks.failed;
            raw_cases(backend.get(), true, checks);
            std::printf("nonfinite/subnormal subtotal: %zu checks, %zu failed\n", checks.total - prior_total, checks.failed - prior_failed);
        } else {
            std::string path = args[1];
            if (synthetic) {
                const auto work = std::filesystem::u8path(args[2]);
                std::filesystem::create_directories(work);
                path = (work / "synthetic-argmax-F32.gguf").u8string();
                state_check::write_fixture(path);
            }
            const Layout & selected = synthetic ? state_check::kLayout : layout;
            const ModelFile model(path, backend.get(), selected, decoding_tensor);
            if (model.str("fastconformer.decoder.prediction") != "kda" || model.str("fastconformer.decoder.kind") != "tdt") {
                throw std::runtime_error("the model comparison requires KDA TDT");
            }
            const PredictionNetwork prediction(model);
            const Joint joint(model);
            selector_cases(model, prediction, joint, path, selected, backend.get(), checks);
            step_cases(model, prediction, joint, backend.get(), checks, frames(joint.hidden(), 16, 11));
            decoding_cases(model, prediction, joint, backend.get(), synthetic, checks);
            setting("0");
            const TdtDecoder original(model, prediction, joint);
            setting("1");
            const TdtDecoder reduced(model, prediction, joint);
            std::printf("plain TDT runtime route: %s\n", reduced.uses_device_argmax(backend.get()) ? "device argmax" : "full logits");
            const BoostingTree tree({{1, 2}, {3, 4}}, original.blank(), 10.0, 1.0, 0.0f);
            const Boost boost{tree, 20.0f};
            for (size_t i = device_at + 2; i < args.size(); ++i) {
                const auto projected = read_npy(args[i]);
                if (projected.shape.size() != 2 || projected.shape[1] != joint.hidden() || projected.f32.empty()) {
                    throw std::runtime_error(args[i] + " must contain nonempty F32 projected frames [T, joint hidden]");
                }
                std::printf("recorded projected input: %s\n", args[i].c_str());
                step_cases(model, prediction, joint, backend.get(), checks, projected.f32);
                for (size_t allowed : {size_t(0), size_t(2), std::numeric_limits<size_t>::max()}) {
                    decode_pair(original, reduced, backend.get(), projected.f32, allowed, checks);
                    decode_pair(original, reduced, backend.get(), projected.f32, allowed, checks, &boost);
                }
            }
            if (synthetic) {
                for (const auto & scope : std::vector<std::pair<std::string, std::string>>{{"Haas-ja-v1.0-other", "kda"}, {"Haas-ja-v1.0", "lstm"}}) {
                    const auto control_path = (std::filesystem::u8path(args[2]) / ("selector-" + scope.second + ".gguf")).u8string();
                    state_check::write_fixture(control_path, scope.first, scope.second);
                    const ModelFile metadata(control_path, state_check::kLayout);
                    selector_cases(metadata, prediction, joint, control_path, state_check::kLayout, backend.get(), checks);
                }
            }
            std::printf("Official NeMo/FLA reference: not checked; these are C++ differential checks\n");
        }
        std::printf("TDT device argmax: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
