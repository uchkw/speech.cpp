// Checks normal projected encoder outputs and the MUL_MAT+UNARY+MUL gate-epilogue dispatch counts separately from
// retained intermediate outputs.
// Usage: haas-mm-glu-check model.gguf expected-eligible-fusions [output.bin]. The binary supports original/off/on
// comparisons; the GGML_METAL_MM_GLU selector is never set or unset here, only externally, in a fresh process per
// build, and this helper never disables generic fusion either.
// These deterministic feature inputs are differential checks, not official NeMo reference tensors.

#include "args.h"
#include "backend.h"
#include "fastconformer/encoder.h"
#include "fastconformer/layout.h"
#include "fastconformer/transducer.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>

namespace {

void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct Fusion {
    using Get = void * (*)(ggml_backend_dev_t);
    using Initialize = void (*)(void *);
    using Reset = void (*)(void *);
    using Stats = int (*)(void *, const char **, uint64_t *, int);
    void * info;
    Reset reset;
    Stats stats;

    explicit Fusion(ggml_backend_t backend) {
        const auto device = ggml_backend_get_device(backend);
        const auto reg = ggml_backend_dev_backend_reg(device);
        require(std::strcmp(ggml_backend_reg_name(reg), "MTL") == 0, "this check requires the Metal backend");
        const auto get = reinterpret_cast<Get>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_get"));
        const auto init = reinterpret_cast<Initialize>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_init"));
        reset = reinterpret_cast<Reset>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_reset"));
        stats = reinterpret_cast<Stats>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_fusion_stats_get"));
        require(get && init && reset && stats, "Metal fusion statistics are unavailable");
        info = get(device);
        // The existing backend keeps its default two command buffers; their host encoding is sequential at n_cb=1.
        init(info);
    }

    uint64_t count() const {
        const char * labels[64] = {};
        uint64_t counts[64] = {};
        const int n = stats(info, labels, counts, 64);
        uint64_t result = 0;
        for (int i = 0; i < n; ++i) if (std::strcmp(labels[i], "MUL_MAT+UNARY+MUL") == 0) result += counts[i];
        return result;
    }
};

using Snapshot = std::map<std::string, std::vector<float>>;

Snapshot evaluate(const fastconformer::Encoder & encoder, const fastconformer::Joint & joint, const ModelFile & model,
                  ggml_backend_t backend, const Fusion & fusion, const std::vector<float> & features, int64_t frames,
                  int64_t t, bool retained, uint64_t expected) {
    Graph graph;
    auto * projected = joint.project_encoder(graph.ctx(), encoder.build(graph, features, frames));
    graph.output(projected);
    std::map<std::string, ggml_tensor *> outputs{{"projected", projected}};
    int glus = 0;
    for (int layer = 0; layer < model.count("fastconformer.encoder.num_layers"); ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".conv_pw1_";
        for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
            auto * mul = ggml_graph_node(graph.cgraph(), i);
            if (mul->op != GGML_OP_MUL) continue;
            auto * unary = mul->src[1];
            auto * factor = mul->src[0];
            if (unary->op != GGML_OP_UNARY || ggml_get_unary_op(unary) != GGML_UNARY_OP_SIGMOID ||
                unary->src[0]->op != GGML_OP_MUL_MAT || factor->op != GGML_OP_MUL_MAT) continue;
            if (ggml_get_name(unary->src[0]->src[0]) != prefix + "gate.weight" ||
                ggml_get_name(factor->src[0]) != prefix + "a.weight") continue;
            auto * gate = unary->src[0];
            ++glus;
            require(ggml_are_same_shape(unary, factor) && unary->type == GGML_TYPE_F32 && factor->type == GGML_TYPE_F32 &&
                        unary->ne[0] == 1024 && unary->ne[2] == 1 && unary->ne[3] == 1,
                    "an actual Conformer GLU has another shape or type");
            require(gate->src[0] != factor->src[0], "the a and gate projections do not use independent weights");
            require((gate->src[0]->type == GGML_TYPE_F32 || gate->src[0]->type == GGML_TYPE_F16 ||
                     gate->src[0]->type == GGML_TYPE_Q8_0 || gate->src[0]->type == GGML_TYPE_Q4_K) &&
                        gate->src[0]->ne[0] == 1024 && gate->src[0]->ne[1] == 1024 &&
                        gate->src[0]->ne[2] == 1 && gate->src[0]->ne[3] == 1,
                    "an actual gate projection weight is not F32, F16, Q8_0 or Q4_K of 1024x1024");
            // The pinned public tensor stores the acc precision in op_params[0] and the src[1] precision hint in op_params[3].
            require(gate->src[1]->type == GGML_TYPE_F32,
                    "an actual gate projection takes activations of another dtype than F32");
            require(gate->op_params[3] == GGML_PREC_UNDEFINED,
                    "an actual gate projection carries a source precision hint other than the default");
            require(gate->op_params[0] == GGML_PREC_F32 && factor->op_params[0] == GGML_PREC_F32,
                    "an actual gate or factor projection does not ask for F32 accumulation");
            if (retained) {
                outputs[prefix + "a"] = factor;
                outputs[prefix + "gate"] = gate;
                outputs[prefix + "sigmoid"] = unary;
                outputs[prefix + "glu"] = mul;
            }
        }
    }
    require(glus == model.count("fastconformer.encoder.num_layers"), "the actual graph does not contain one GLU per layer");
    if (retained) for (const auto & item : outputs) graph.output(item.second);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(bool(allocator), "cannot allocate the encoder graph");
    fusion.reset(fusion.info);
    graph.compute(backend, allocator.get());
    const auto actual = fusion.count();
    std::printf("T=%lld mode=%s graph_glus=%d mm_glu_fusions=%llu expected=%llu\n", (long long) t,
                retained ? "retained" : "normal", glus, (unsigned long long) actual, (unsigned long long) expected);
    require(actual == expected, "the MUL_MAT+UNARY+MUL fusion count differs from the required path");
    Snapshot result;
    for (const auto & item : outputs) {
        auto values = Graph::read(item.second);
        for (float value : values) require(std::isfinite(value), "an encoder result is not finite");
        result.emplace(item.first, std::move(values));
    }
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        require(args.size() == 3 || args.size() == 4, "usage: haas-mm-glu-check model.gguf expected-eligible-fusions [output.bin]");
        const int expected = std::stoi(args[2]);
        require(expected == 0 || expected == 24, "expected eligible fusions must be 0 or 24");
        require(!std::getenv("GGML_METAL_FUSION_DEBUG") && !std::getenv("GGML_METAL_FUSION_DISABLE"),
                "the normal-policy check forbids fusion debug or disabling generic fusion");
        std::unique_ptr<FILE, decltype(&std::fclose)> dump(args.size() == 4 ? ggml_fopen(args[3].c_str(), "wb") : nullptr, std::fclose);
        require(args.size() != 4 || bool(dump), "cannot open the encoder comparison file");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device("gpu"), 4), ggml_backend_free);
        const Fusion fusion(backend.get());
        const ModelFile model(args[1], backend.get(), fastconformer::layout);
        require(model.str("fastconformer.decoder.prediction") == "kda" && !model.boolean("fastconformer.encoder.use_bias") &&
                    model.count("fastconformer.encoder.num_layers") == 24, "this check requires Haas's 24 unbiased layers");
        const fastconformer::Encoder encoder(model, backend.get());
        const fastconformer::Joint joint(model);
        for (int64_t length : {1, 8, 16, 17, 31, 32, 33, 63, 64, 65, 171, 172, 375, 376, 384, 385}) {
            const int64_t frames = length * encoder.subsampling_factor() - 1;
            const int64_t t = encoder.subsampled_frames(frames);
            // The trial's fused gate epilogue is eligible only for 17 <= T <= 384; every other length must report none.
            const uint64_t normal_expected = t >= 17 && t <= 384 ? (uint64_t) expected : 0;
            std::vector<float> features((size_t) frames * model.size("fastconformer.frontend.n_mels"));
            for (size_t i = 0; i < features.size(); ++i) features[i] = std::sin((float) i * 0.013f) + 0.1f * std::cos((float) i * 0.037f);
            const auto normal = evaluate(encoder, joint, model, backend.get(), fusion, features, frames, t, false, normal_expected);
            const auto retained = evaluate(encoder, joint, model, backend.get(), fusion, features, frames, t, true, 0);
            const auto & a = normal.at("projected");
            const auto & b = retained.at("projected");
            require(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0,
                    "retaining the gate intermediates changes the normal projected encoder output");
            if (dump) {
                for (const auto * snapshot : {&normal, &retained}) for (const auto & item : *snapshot) {
                    const uint64_t size = item.second.size()*sizeof(float);
                    require(std::fwrite(&size, sizeof(size), 1, dump.get()) == 1 &&
                                std::fwrite(item.second.data(), 1, size, dump.get()) == size, "cannot write encoder comparison bytes");
                }
            }
        }
        std::printf("passed: normal and retained outputs agree; official references and recognition outputs are not checked\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
