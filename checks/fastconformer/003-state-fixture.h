#pragma once

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.h"
#include "model-file.h"

namespace state_check {

constexpr int64_t kHidden = 640, kHeads = 5, kHead = 64, kWidth = kHeads * kHead, kTaps = 4, kLayers = 2;
constexpr int kBlank = 16, kOutputs = kBlank + 1 + 5;
const Storage kFloat = float32_storage();

/** A decoder-only F32 fixture with two KDA layers and the complete recurrent and convolution state. */
std::vector<TensorSpec> tensors() {
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

const Layout kLayout = {"fastconformer-state-check", 1, "make it again with fastconformer-state-check",
                        [](const ModelFile &) { return tensors(); }};

/** Writes deterministic synthetic weights, with a joint that lets projected frames select tokens and durations. */
void write_fixture(const std::string & path) {
    const auto specs = tensors();
    size_t bytes = 1024;
    for (const TensorSpec & spec : specs) {
        size_t count = 1;
        for (int64_t n : spec.shape.ne) count *= (size_t) n;
        bytes += ggml_tensor_overhead() + count * sizeof(float) + 64;
    }
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({bytes, nullptr, false}), ggml_free);
    if (!ctx) throw std::runtime_error("cannot create the synthetic state fixture");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> file(gguf_init_empty(), gguf_free);
    gguf_set_val_str(file.get(), "general.architecture", kLayout.architecture);
    gguf_set_val_str(file.get(), "general.name", "synthetic-kda-state");
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
    gguf_set_val_u32(file.get(), "fastconformer.decoder.rnnt.max_symbols", 2);
    const int32_t durations[] = {0, 1, 2, 3, 4};
    gguf_set_arr_data(file.get(), "fastconformer.decoder.tdt.durations", GGUF_TYPE_INT32, durations, 5);
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

}  // namespace state_check
