#include "kda.h"

#include <cmath>

#include "error.h"
#include "ggml-cpu.h"

namespace fastconformer {

namespace {

ggml_tensor * layer_norm(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name, float eps) {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps), m.tensor(name + ".weight")), m.tensor(name + ".bias"));
}

/**
 * fused_recurrent_kda()'s normalization of q and k: each head [rows of ne0] divided by the square root of its squares'
 * sum plus `eps`, which ggml_l2_norm() would instead bound below by eps. It is x / sqrt(mean(x^2) + eps / n) /
 * sqrt(n), two nodes where the sum, root and division take five: a step of the decoding is mostly the CPU's threads
 * meeting at each node, or a GPU's dispatches.
 */
ggml_tensor * qk_norm(ggml_context * ctx, ggml_tensor * x, float eps) {
    const float n = (float) x->ne[0];
    return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps / n), 1.0f / std::sqrt(n));
}

}  // namespace

struct KdaLayers::Constants {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> negative_a;

    Constants(const ModelFile & model, ggml_backend_t backend, int layers, int64_t heads) {
        context.reset(ggml_init({ggml_tensor_overhead() * (size_t) layers, nullptr, true}));
        if (!context) throw Error(Fault::OutOfMemory, "cannot create the KDA constants context");
        for (int l = 0; l < layers; l++) {
            auto * value = ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, 1, heads);
            ggml_set_name(value, ("pred.kda." + std::to_string(l) + ".negative_a").c_str());
            negative_a.push_back(value);
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) throw Error(Fault::OutOfMemory, "cannot allocate KDA constants");
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        if (!allocator) throw Error(Fault::OutOfMemory, "cannot create the KDA constants allocator");
        Graph graph(8 * layers + 8);
        for (int l = 0; l < layers; l++) {
            auto * input = ggml_reshape_2d(graph.ctx(), model.tensor("pred.kda." + std::to_string(l) + ".a_log"), 1, heads);
            graph.copy(ggml_neg(graph.ctx(), ggml_exp(graph.ctx(), input)), negative_a[l]);
        }
        graph.compute(backend, allocator.get());
    }
};

KdaLayers::~KdaLayers() = default;

KdaLayers::KdaLayers(const ModelFile & m, ggml_backend_t backend, bool cache_constants)
    : m_(m),
      layers_((int) m.u32("fastconformer.decoder.prediction_layers")),
      heads_(m.u32("fastconformer.decoder.kda.num_heads")),
      head_dim_(m.width("pred.kda.0.o_norm.weight", 0)),
      conv_size_(m.width("pred.kda.0.q_conv.weight", 0)),
      norm_eps_(m.f32("fastconformer.decoder.kda.norm_eps")),
      gate_norm_eps_(m.f32("fastconformer.decoder.kda.gate_norm_eps")),
      qk_norm_eps_(m.f32("fastconformer.decoder.kda.qk_norm_eps")),
      final_norm_(m.boolean("fastconformer.decoder.kda.final_norm")),
      backend_(backend), cache_constants_(cache_constants && ggml_backend_is_cpu(backend)) {
    if (cache_constants && !backend_) throw std::logic_error("KDA constant preparation needs its prediction backend");
}

PredictionConstants KdaLayers::constants() const {
    if (!constants_) return {};
    return {preparations_, ggml_backend_buffer_get_size(constants_->buffer.get()), backend_,
            std::vector<const ggml_tensor *>(constants_->negative_a.begin(), constants_->negative_a.end())};
}

ggml_tensor * KdaLayers::build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * state, ggml_tensor ** next) const {
    if (cache_constants_ && !constants_) {
        // A model serves one request at a time; only a completed preparation becomes visible to later graphs.
        auto prepared = std::make_unique<Constants>(m_, backend_, layers_, heads_);
        constants_ = std::move(prepared);
        ++preparations_;
    }
    const int64_t d = head_dim_, h = heads_, width = h * d, past = conv_size_ - 1;
    const size_t f = sizeof(float);
    // Each part of the next state is written into its place, in the order of the layout: a chain of concatenations
    // would copy all that comes before each part again, and ggml's CPU concatenates a row on one thread, element by
    // element.
    ggml_tensor * out = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, state_size());
    size_t written = 0;
    auto keep = [&](ggml_tensor * t) {
        out = ggml_nelements(t) == t->ne[0] ? ggml_set_1d_inplace(ctx, out, t, written) : ggml_set_2d_inplace(ctx, out, t, (size_t) t->ne[0] * f, written);
        written += (size_t) ggml_nelements(t) * f;
    };
    for (int l = 0; l < layers_; l++) {
        const std::string p = "pred.kda." + std::to_string(l) + ".";
        const size_t base = (size_t) (l * layer_state()) * f;
        ggml_tensor * normed = layer_norm(ctx, m_, x, p + "norm", norm_eps_);
        auto weight = [&](const std::string & name) { return m_.tensor(p + name); };

        // Each convolution takes its projection after the inputs before it, [conv_size, width], whose last
        // conv_size - 1 are the next state's.
        ggml_tensor * windows[3];
        ggml_tensor * convolved[3];
        const char * names[3] = {"q", "k", "v"};
        for (int i = 0; i < 3; i++) {
            ggml_tensor * before = ggml_view_2d(ctx, state, past, width, (size_t) past * f, base + (size_t) (width * d + i * past * width) * f);
            ggml_tensor * projected = mul_mat(ctx, weight(std::string(names[i]) + ".weight"), normed);
            windows[i] = ggml_concat(ctx, before, ggml_reshape_2d(ctx, projected, 1, width), 0);
            convolved[i] = ggml_silu(ctx, ggml_ssm_conv(ctx, ggml_reshape_3d(ctx, windows[i], conv_size_, width, 1), weight(std::string(names[i]) + "_conv.weight")));
        }
        ggml_tensor * q = qk_norm(ctx, ggml_reshape_2d(ctx, convolved[0], d, h), qk_norm_eps_);
        ggml_tensor * k = qk_norm(ctx, ggml_reshape_2d(ctx, convolved[1], d, h), qk_norm_eps_);
        ggml_tensor * v = ggml_reshape_2d(ctx, convolved[2], d, h);

        ggml_tensor * gate = mul_mat(ctx, weight("f_b.weight"), mul_mat(ctx, weight("f_a.weight"), normed));
        gate = ggml_softplus(ctx, ggml_add(ctx, ggml_reshape_2d(ctx, gate, d, h), ggml_reshape_2d(ctx, weight("dt_bias"), d, h)));
        ggml_tensor * negative_a = constants_ ? constants_->negative_a[l] :
            ggml_neg(ctx, ggml_exp(ctx, ggml_reshape_2d(ctx, weight("a_log"), 1, h)));
        gate = ggml_mul(ctx, gate, negative_a);
        ggml_tensor * beta = ggml_sigmoid(ctx, ggml_reshape_2d(ctx, mul_mat(ctx, weight("b.weight"), normed), 1, h));
        ggml_tensor * recurrent = ggml_view_4d(ctx, state, d, d, h, 1, d * f, d * d * f, width * d * f, base);
        ggml_tensor * attended =
            ggml_gated_delta_net(ctx, ggml_reshape_4d(ctx, q, d, h, 1, 1), ggml_reshape_4d(ctx, k, d, h, 1, 1), ggml_reshape_4d(ctx, v, d, h, 1, 1),
                                 ggml_reshape_4d(ctx, gate, d, h, 1, 1), ggml_reshape_4d(ctx, beta, 1, h, 1, 1), recurrent, 1);
        // The result holds the output of the heads, [head_dim, heads], then the next recurrent state.
        keep(ggml_view_1d(ctx, attended, width * d, (size_t) width * f));
        for (ggml_tensor * w : windows) keep(ggml_view_2d(ctx, w, past, width, w->nb[1], f));

        ggml_tensor * o = ggml_rms_norm(ctx, ggml_view_2d(ctx, attended, d, h, d * f, 0), gate_norm_eps_);
        ggml_tensor * output_gate = ggml_add(ctx, mul_mat(ctx, weight("g_b.weight"), mul_mat(ctx, weight("g_a.weight"), normed)), weight("g_b.bias"));
        o = ggml_mul(ctx, ggml_mul(ctx, o, weight("o_norm.weight")), ggml_sigmoid(ctx, ggml_reshape_2d(ctx, output_gate, d, h)));
        x = ggml_add(ctx, x, mul_mat(ctx, weight("o.weight"), ggml_reshape_1d(ctx, o, width)));
    }
    *next = out;
    return final_norm_ ? layer_norm(ctx, m_, x, "pred.norm", norm_eps_) : x;
}

std::vector<std::string> KdaLayers::matrices() const {
    std::vector<std::string> names;
    for (int l = 0; l < layers_; l++) {
        for (const char * part : {"q", "k", "v", "f_a", "f_b", "b", "g_a", "g_b", "o"}) {
            names.push_back("pred.kda." + std::to_string(l) + "." + part + ".weight");
        }
    }
    return names;
}

}  // namespace fastconformer
