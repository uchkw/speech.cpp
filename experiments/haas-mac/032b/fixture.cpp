// The synthetic model file of the 032b check, written at run time and never committed: a layout of the check's own,
// as quantize-check's is, whose tensor list one function names for both the writer and the reader, so the file and
// its layout cannot disagree. The weights are finite nonzero values that differ from tensor to tensor, made by the
// same deterministic sine the check's features are, at run time only.

#include "fixture.h"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf.h"
#include "storage.h"

namespace batch2 {

namespace {

/**
 * How the layout stores a tensor in a file of each weight type, as FastConformer's does: the matrices in the file's
 * type, F32, F16, Q8_0 or Q4_K here, and the convolution kernels, the norms and the biases in F32, which every
 * reader takes.
 */
const Storage kMatrix = quantized_storage({{GGML_TYPE_F32, kFirstLayoutRelease},
                                           {GGML_TYPE_F16, kFirstLayoutRelease},
                                           {GGML_TYPE_Q8_0, "0.8.0"},
                                           {GGML_TYPE_Q4_K, "0.8.0"}});
const Storage kF32 = float32_storage();

/** FNV-1a of a tensor's name, which gives each tensor's values a phase of their own. */
uint32_t name_hash(const std::string & name) {
    uint32_t h = 2166136261u;
    for (char c : name) {
        h ^= (uint8_t) c;
        h *= 16777619u;
    }
    return h;
}

void fill(ggml_tensor * t, uint32_t seed) {
    const double phase = 0.37 * (double) (seed % 7919);
    float * v = (float *) t->data;
    for (int64_t i = 0; i < ggml_nelements(t); i++) {
        // Always in [0.05, 0.45]: finite, never zero, and the joint projections cannot vanish into a constant.
        v[i] = 0.25f + 0.2f * std::sin(0.37 * (double) i + phase);
    }
}

}  // namespace

const std::vector<TensorSpec> & tensors() {
    static const std::vector<TensorSpec> t = [] {
        // The subsampling halves the mel axis three times: 80 mels give a width of 10.
        const int64_t d = kDModel, ffn = kFfn, k = 3, subsampled_mels = 10;
        std::vector<TensorSpec> t = {{"sub.conv.0.weight", {k, k, 1, kChannels}, kF32},
                                     {"sub.conv.0.bias", {kChannels}, kF32},
                                     {"sub.out.weight", {kChannels * subsampled_mels, d}, kMatrix},
                                     {"sub.out.bias", {d}, kF32},
                                     {"joint.enc.weight", {d, kJointHidden}, kMatrix},
                                     {"joint.enc.bias", {kJointHidden}, kF32},
                                     {"joint.out.weight", {kJointHidden, kOutputs}, kMatrix},
                                     {"joint.out.bias", {kOutputs}, kF32}};
        for (int i = 1; i < kSubLayers; i++) {
            add_block(t, "sub.conv." + std::to_string(i) + ".",
                      {{"dw.weight", {k, k, 1, kChannels}, kF32},
                       {"dw.bias", {kChannels}, kF32},
                       {"pw.weight", {kChannels, kChannels}, kMatrix},
                       {"pw.bias", {kChannels}, kF32}});
        }
        add_numbered(t, "blk.", kLayers,
                     {{"ff1_norm.weight", {d}, kF32},          {"ff1_norm.bias", {d}, kF32},
                      {"ff2_norm.weight", {d}, kF32},          {"ff2_norm.bias", {d}, kF32},
                      {"attn_norm.weight", {d}, kF32},         {"attn_norm.bias", {d}, kF32},
                      {"conv_norm.weight", {d}, kF32},         {"conv_norm.bias", {d}, kF32},
                      {"out_norm.weight", {d}, kF32},          {"out_norm.bias", {d}, kF32},
                      {"attn_pos.weight", {d, d}, kMatrix},    {"attn_pos_bias_u", {d / kHeads, kHeads}, kF32},
                      {"attn_pos_bias_v", {d / kHeads, kHeads}, kF32}, {"conv_dw.weight", {kConvKernel, d}, kF32},
                      {"conv_dw.bias", {d}, kF32},             {"ff1_up.weight", {d, ffn}, kMatrix},
                      {"ff1_down.weight", {ffn, d}, kMatrix},  {"ff2_up.weight", {d, ffn}, kMatrix},
                      {"ff2_down.weight", {ffn, d}, kMatrix},  {"attn_q.weight", {d, d}, kMatrix},
                      {"attn_k.weight", {d, d}, kMatrix},      {"attn_v.weight", {d, d}, kMatrix},
                      {"attn_out.weight", {d, d}, kMatrix},    {"conv_pw1_a.weight", {d, d}, kMatrix},
                      {"conv_pw1_gate.weight", {d, d}, kMatrix}, {"conv_pw2.weight", {d, d}, kMatrix}});
        add_numbered(t, "blk.", kLayers,
                     {{"ff1_up.bias", {ffn}, kF32},    {"ff1_down.bias", {d}, kF32},   {"ff2_up.bias", {ffn}, kF32},
                      {"ff2_down.bias", {d}, kF32},    {"attn_q.bias", {d}, kF32},     {"attn_k.bias", {d}, kF32},
                      {"attn_v.bias", {d}, kF32},      {"attn_out.bias", {d}, kF32},   {"conv_pw1_a.bias", {d}, kF32},
                      {"conv_pw1_gate.bias", {d}, kF32}, {"conv_pw2.bias", {d}, kF32}});
        return t;
    }();
    return t;
}

void write_fixture(const std::string & path, bool local_attention) {
    const std::vector<TensorSpec> & specs = tensors();
    int64_t elements = 0;
    for (const TensorSpec & spec : specs) elements += spec.shape.ne[0] * spec.shape.ne[1] * spec.shape.ne[2] * spec.shape.ne[3];
    ggml_init_params params = {
        ggml_tensor_overhead() * (specs.size() + 2) + (size_t) elements * sizeof(float) + 1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    if (!ctx) throw std::runtime_error("cannot create the fixture's tensor context");
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_empty(), gguf_free);

    gguf_set_val_str(g.get(), "general.architecture", "fastconformer-batch2-check");
    gguf_set_val_str(g.get(), "general.name", "batch2-check");
    gguf_set_val_str(g.get(), "general.organization", "speech.cpp");
    gguf_set_val_str(g.get(), "general.basename", "batch2-check");
    gguf_set_val_str(g.get(), "general.size_label", "13M");
    gguf_set_val_str(g.get(), "general.license", "MIT");
    gguf_set_val_str(g.get(), "general.source.url", "https://huggingface.co/speech-cpp/batch2-check/tree/0");
    gguf_set_val_str(g.get(), "general.source.repo_url", "https://huggingface.co/speech-cpp/batch2-check");
    gguf_set_val_u32(g.get(), "general.file_type", 0);
    gguf_set_val_u32(g.get(), "speech.layout", 1);
    gguf_set_val_str(g.get(), "speech.requires", "0.7.0");

    gguf_set_val_u32(g.get(), "fastconformer.frontend.n_mels", (uint32_t) kMels);
    gguf_set_val_u32(g.get(), "fastconformer.encoder.d_model", (uint32_t) kDModel);
    gguf_set_val_u32(g.get(), "fastconformer.encoder.num_layers", (uint32_t) kLayers);
    gguf_set_val_u32(g.get(), "fastconformer.encoder.num_heads", (uint32_t) kHeads);
    gguf_set_val_u32(g.get(), "fastconformer.encoder.conv_kernel", (uint32_t) kConvKernel);
    gguf_set_val_u32(g.get(), "fastconformer.encoder.subsampling_factor", (uint32_t) 1 << kSubLayers);
    gguf_set_val_f32(g.get(), "fastconformer.encoder.norm_eps", 1e-5f);
    gguf_set_val_f32(g.get(), "fastconformer.encoder.pos_base", 10000.0f);
    // sqrt(d_model), so the batched input scaling runs in the check.
    gguf_set_val_f32(g.get(), "fastconformer.encoder.xscale", 16.0f);
    gguf_set_val_f32(g.get(), "fastconformer.encoder.ff_factor", 0.5f);
    gguf_set_val_bool(g.get(), "fastconformer.encoder.use_bias", true);
    gguf_set_val_str(g.get(), "fastconformer.encoder.attention", local_attention ? "rel_pos_local_attn" : "rel_pos");
    if (local_attention) {
        gguf_set_val_u32(g.get(), "fastconformer.encoder.attention_context", 64);
        gguf_set_val_u32(g.get(), "fastconformer.encoder.global_tokens", 8);
    }

    for (const TensorSpec & spec : specs) {
        ggml_tensor * t = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, spec.shape.ne[0], spec.shape.ne[1], spec.shape.ne[2],
                                             spec.shape.ne[3]);
        ggml_set_name(t, spec.name.c_str());
        fill(t, name_hash(spec.name));
        gguf_add_tensor(g.get(), t);
    }
    if (!gguf_write_to_file(g.get(), path.c_str(), false)) throw std::runtime_error("cannot write " + path);
}

}  // namespace batch2

const Layout kBatch2Layout = {"fastconformer-batch2-check", 1, "make it again with the 032b check's fixture writer",
                              [](const ModelFile &) { return batch2::tensors(); }};
