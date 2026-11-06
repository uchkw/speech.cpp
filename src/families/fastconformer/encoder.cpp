#include "encoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "error.h"

namespace fastconformer {

/*
 * Activations are [channels, frames] (ne0 = channels). The subsampling runs on [channels, mels, frames], channels
 * fastest as well, an image whose width is the mel axis and height the time axis.
 */

namespace {

/**
 * The depthwise convolution of `x` [channels, width, height] by `kernel` [kw, kh, 1, channels], [channels, width',
 * height']: ggml_conv_2d_dw_direct() on the channel-fastest layout, which its result keeps, with the kernel's taps
 * channel-fastest as well, as the CPU asserts and Vulkan's shader reads them without looking at the strides.
 */
ggml_tensor * depthwise(ggml_context * ctx, ggml_tensor * x, ggml_tensor * kernel, int stride, int pad_w, int pad_h) {
    kernel = ggml_permute(ctx, ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0)), 3, 2, 0, 1);
    ggml_tensor * y = ggml_conv_2d_dw_direct(ctx, kernel, ggml_permute(ctx, x, 2, 0, 1, 3), stride, stride, pad_w, pad_h, 1, 1);
    return ggml_permute(ctx, y, 1, 2, 0, 3);
}

}  // namespace

struct Encoder::Positions {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{nullptr, ggml_backend_buffer_free};
    std::map<std::string, ggml_tensor *> projected;

    Positions(const ModelFile & model, ggml_backend_t backend, const std::vector<float> & positions, int width, int layers) {
        context.reset(ggml_init({ggml_tensor_overhead() * (size_t) layers, nullptr, true}));
        if (!context) throw Error(Fault::OutOfMemory, "cannot create a position context");
        const int64_t count = (int64_t) positions.size() / width;
        for (int l = 0; l < layers; l++) {
            projected.emplace("blk." + std::to_string(l) + ".attn", ggml_new_tensor_2d(context.get(), GGML_TYPE_F32, width, count));
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(context.get(), backend));
        if (!buffer) throw Error(Fault::OutOfMemory, "cannot allocate attention positions");
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        if (!allocator) throw Error(Fault::OutOfMemory, "cannot create a position allocator");
        Graph graph;
        auto * input = graph.input(positions, width, count);
        for (const auto & entry : projected) {
            graph.copy(mul_mat(graph.ctx(), model.tensor(entry.first + "_pos.weight"), input), entry.second);
        }
        graph.compute(backend, allocator.get());
    }
};

Encoder::~Encoder() = default;

Encoder::Encoder(const ModelFile & m, ggml_backend_t backend)
    : backend_(backend), m_(m),
      mels_((int) m.u32("fastconformer.frontend.n_mels")),
      d_model_((int) m.u32("fastconformer.encoder.d_model")),
      layers_((int) m.u32("fastconformer.encoder.num_layers")),
      heads_((int) m.u32("fastconformer.encoder.num_heads")),
      conv_kernel_((int) m.u32("fastconformer.encoder.conv_kernel")),
      eps_(m.f32("fastconformer.encoder.norm_eps")),
      pos_base_(m.f32("fastconformer.encoder.pos_base")),
      xscale_(m.f32("fastconformer.encoder.xscale")),
      ff_factor_(m.f32("fastconformer.encoder.ff_factor")),
      use_bias_(m.boolean("fastconformer.encoder.use_bias")) {
    local_ = m.one_of("fastconformer.encoder.attention", {"rel_pos", "rel_pos_local_attn"}) == "rel_pos_local_attn";
    if (local_) {
        context_ = (int) m.u32("fastconformer.encoder.attention_context");
        global_tokens_ = (int) m.u32("fastconformer.encoder.global_tokens");
    }
    // Global attention can reuse the same central positions across input lengths. Bound the retained projections
    // to 32 MiB; longer inputs compute their positions in the utterance graph. Local attention has a fixed span.
    const int64_t rows = 32 * 1024 * 1024 / ((int64_t) layers_ * d_model_ * (int64_t) sizeof(float));
    cached_span_ = local_ ? context_ : rows > 0 ? (rows - 1) / 2 : -1;
    sub_layers_ = 0;
    for (uint32_t f = m.u32("fastconformer.encoder.subsampling_factor"); f > 1; f /= 2) sub_layers_++;
}

int64_t Encoder::subsampled_frames(int64_t frames) const {
    for (int i = 0; i < sub_layers_; i++) frames = (frames - 1) / 2 + 1;
    return frames;
}

ggml_tensor * Encoder::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name, bool bias) const {
    ggml_tensor * y = mul_mat(ctx, m_.tensor(name + ".weight"), x);
    return bias ? ggml_add(ctx, y, m_.tensor(name + ".bias")) : y;
}

ggml_tensor * Encoder::layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(name + ".weight")), m_.tensor(name + ".bias"));
}

ggml_tensor * Encoder::subsample(Graph & g, ggml_tensor * features) const {
    ggml_context * ctx = g.ctx();
    // ConvSubsampling "dw_striding" (nemo/collections/asr/parts/submodules/subsampling.py): a 3x3 convolution of
    // stride 2, then (depthwise 3x3 of stride 2, pointwise) for each further halving, each followed by a ReLU. The
    // first convolution, of one input channel, is the product of its kernel [9, channels] with the input's im2col [9,
    // positions], channel-fastest: ggml_conv_2d_direct() took 18.9 ms of a 112.6 ms encoder on 25.5 s on Metal, and the
    // copies between it and the pointwise convolutions' layout 7 ms more (Apple M5, Haas-ja-v1.0 in Q8_0, 2026-10-31).
    ggml_tensor * w0 = m_.tensor("sub.conv.0.weight");
    ggml_tensor * cols = ggml_im2col(ctx, w0, features, 2, 2, 1, 1, 1, 1, true, GGML_TYPE_F32);
    const int64_t c = w0->ne[3];
    ggml_tensor * x = mul_mat(ctx, ggml_reshape_2d(ctx, w0, w0->ne[0] * w0->ne[1], c), ggml_reshape_2d(ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2]));
    x = ggml_reshape_3d(ctx, ggml_relu(ctx, ggml_add(ctx, x, m_.tensor("sub.conv.0.bias"))), c, cols->ne[1], cols->ne[2]);
    for (int i = 1; i < sub_layers_; i++) {
        const std::string p = "sub.conv." + std::to_string(i);
        x = ggml_add(ctx, depthwise(ctx, x, m_.tensor(p + ".dw.weight"), 2, 1, 1), m_.tensor(p + ".dw.bias"));
        const int64_t w = x->ne[1], h = x->ne[2];
        x = ggml_reshape_3d(ctx, ggml_relu(ctx, linear(ctx, ggml_reshape_2d(ctx, x, c, w * h), p + ".pw", true)), c, w, h);
    }
    // ConvSubsampling.forward() flattens (channels, mels) with the mel axis fastest.
    const int64_t w = x->ne[1], h = x->ne[2];
    x = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), w * c, h);
    return linear(ctx, x, "sub.out", true);
}

ggml_tensor * Encoder::feed_forward(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    ggml_tensor * h = ggml_silu(ctx, linear(ctx, layer_norm(ctx, x, name + "_norm"), name + "_up", use_bias_));
    return ggml_add(ctx, x, ggml_scale(ctx, linear(ctx, h, name + "_down", use_bias_), ff_factor_));
}

Encoder::AttentionInputs Encoder::attention_inputs(Graph & g, int64_t t) const {
    // RelPositionalEncoding.extend_pe() (multi_head_attention.py) encodes the relative positions T - 1 down to
    // -(T - 1), and LocalAttRelPositionalEncoding.extend_pe() those from the context down to minus the context; both
    // through create_pe(): sines in the even channels and cosines in the odd, computed in float32, so the angles
    // round as they do here.
    const int64_t needed = local_ ? context_ : t - 1;
    const bool cached = needed <= cached_span_;
    const int64_t span = cached ? cached_span_ : needed;
    std::vector<float> pe;
    if (!cached || !positions_) {
        pe.resize((size_t) ((2 * span + 1) * d_model_));
        const float step = (float) (-std::log((double) pos_base_) / d_model_);
        for (int64_t c = 0; c < 2 * span + 1; c++) {
            const float position = (float) (span - c);
            for (int i = 0; i < d_model_; i += 2) {
                const float angle = position * std::exp((float) i * step);
                pe[(size_t) (c * d_model_ + i)] = std::sin(angle);
                pe[(size_t) (c * d_model_ + i + 1)] = std::cos(angle);
            }
        }
        if (cached && !positions_) positions_ = std::make_unique<Positions>(m_, backend_, pe, d_model_, layers_);
    }
    AttentionInputs in{cached ? nullptr : g.input(pe, d_model_, 2 * span + 1), nullptr};
    if (local_) {
        // The scores of block k's query r are [global tokens, window], where window column c is the frame
        // (k - 1) w + c; a frame is seen when it is within w of the query and in the utterance. NeMo masks the frames
        // before the utterance with -inf and its padding after it with -10000, which both leave a weight of 0.
        const int64_t w = std::min<int64_t>(context_, t), blocks = (t + w - 1) / w, globals = std::min<int64_t>(global_tokens_, t);
        const int64_t row = globals + 3 * w;
        std::vector<float> mask((size_t) (row * w * blocks), -std::numeric_limits<float>::infinity());
        for (int64_t k = 0; k < blocks; k++) {
            for (int64_t r = 0; r < w; r++) {
                float * m = &mask[(size_t) ((k * w + r) * row)];
                for (int64_t c = 0; c < globals; c++) m[c] = 0.0f;
                for (int64_t c = r; c <= r + 2 * w; c++) {
                    const int64_t key = (k - 1) * w + c;
                    if (key >= 0 && key < t) m[globals + c] = 0.0f;
                }
            }
        }
        in.mask = g.input(mask, row, w, blocks);
    }
    return in;
}

/**
 * RelPositionMultiHeadAttention.forward() (nemo/collections/asr/parts/submodules/multi_head_attention.py):
 *
 *     scores = ((q + u) k^T + rel_shift((q + v) p^T)) / sqrt(d_k)
 *
 * where p is linear_pos of the encodings of the relative positions T - 1 down to -(T - 1). rel_shift() pads a
 * zero column on the left, reads the [T, 2T] result as [2T, T] and drops its first row; element (i, j) of the result
 * is then element T - 1 - i + j of row i of (q + v) p^T, which never reaches the zero column: element
 * i (2T - 2) + j + T - 1 of [T, 2T - 1], a view whose rows step by 2T - 2.
 */
ggml_tensor * Encoder::attention(Graph & g, ggml_tensor * x, const AttentionInputs & in, const std::string & name) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = x->ne[1], dk = d_model_ / heads_;
    ggml_tensor * h = layer_norm(ctx, x, name + "_norm");
    ggml_tensor * q = ggml_reshape_3d(ctx, linear(ctx, h, name + "_q", use_bias_), dk, heads_, t);
    ggml_tensor * k = ggml_reshape_3d(ctx, linear(ctx, h, name + "_k", use_bias_), dk, heads_, t);
    ggml_tensor * v = ggml_reshape_3d(ctx, linear(ctx, h, name + "_v", use_bias_), dk, heads_, t);
    if (local_) return ggml_add(ctx, x, linear(ctx, local_attention(g, q, k, v, in, name), name + "_out", use_bias_));

    auto heads = [&](ggml_tensor * y, int64_t n) {
        return ggml_permute(ctx, ggml_reshape_3d(ctx, y, dk, heads_, n), 0, 2, 1, 3);
    };
    ggml_tensor * projected;
    if (in.pos) projected = mul_mat(ctx, m_.tensor(name + "_pos.weight"), in.pos);
    else {
        projected = positions_->projected.at(name);
        projected = ggml_view_2d(ctx, projected, d_model_, 2 * t - 1, projected->nb[1],
                                 (size_t) (cached_span_ - (t - 1)) * projected->nb[1]);
    }
    ggml_tensor * p = heads(projected, 2 * t - 1);
    ggml_tensor * qu = heads(ggml_add(ctx, q, m_.tensor(name + "_pos_bias_u")), t);
    ggml_tensor * qv = heads(ggml_add(ctx, q, m_.tensor(name + "_pos_bias_v")), t);

    ggml_tensor * ac = mul_mat(ctx, heads(k, t), qu);
    ggml_tensor * bd = mul_mat(ctx, p, qv);
    bd = ggml_view_3d(ctx, bd, t, t, heads_, (size_t) (2 * t - 2) * sizeof(float), bd->nb[2], (size_t) (t - 1) * sizeof(float));
    ggml_tensor * scores = ggml_add(ctx, ac, bd);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt((float) dk), 0.0f);

    // [d_k, heads, T] to [T, d_k, heads], so that the product contracts over the keys.
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * out = mul_mat(ctx, vt, weights);
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)), d_model_, t);
    return ggml_add(ctx, x, linear(ctx, out, name + "_out", use_bias_));
}

/**
 * RelPositionMultiHeadAttentionLongformer.forward() (multi_head_attention.py) with a context of w frames on either
 * side and G global tokens, the first G frames, whose queries, keys and values are the layer's own
 * (global_attn_separate false). Frame i's scores are
 *
 *     [q_i k_g for each global token g, (q_i + u) k_j + (q_i + v) p_(i - j) for each j within w of i] / sqrt(d_k),
 *
 * a global token appearing among both when it is within w of i, as NeMo computes it; and a global token's own
 * output is replaced by its attention over all frames with q k^T / sqrt(d_k) alone. As in NeMo, the queries go in
 * blocks of w frames, padded to whole blocks, and block k's queries attend to the 3w frames from (k - 1) w: the keys
 * and values padded with w zeros on either side and three consecutive blocks of them concatenated, since a ggml view
 * cannot overlap itself. The mask leaves the band within the utterance. The positional term of query r of a block at
 * window column c is that of encoding c - r: the [w, 2w + 1] products padded to rows of 3w and read with a row stride
 * one shorter, a strided view as in rel_shift().
 */
ggml_tensor * Encoder::local_attention(Graph & g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, const AttentionInputs & in,
                                       const std::string & name) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = q->ne[2], dk = d_model_ / heads_, w = std::min<int64_t>(context_, t), blocks = (t + w - 1) / w, padded = blocks * w;
    const int64_t window = 3 * w, globals = std::min<int64_t>(global_tokens_, t);
    // [d_k, heads, T] to [d_k, T, heads].
    auto heads = [&](ggml_tensor * y) { return ggml_cont(ctx, ggml_permute(ctx, y, 0, 2, 1, 3)); };
    // [d_k, T, heads] padded to whole blocks and split into them, [d_k, w, blocks, heads].
    auto query_blocks = [&](ggml_tensor * y) { return ggml_reshape_4d(ctx, ggml_pad(ctx, y, 0, (int) (padded - t), 0, 0), dk, w, blocks, heads_); };
    ggml_tensor * kh = heads(k);
    // [T, d_k, heads], so that the products with the weights contract over the keys.
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));

    ggml_tensor * qu = query_blocks(heads(ggml_add(ctx, q, m_.tensor(name + "_pos_bias_u"))));
    ggml_tensor * qv = ggml_pad(ctx, heads(ggml_add(ctx, q, m_.tensor(name + "_pos_bias_v"))), 0, (int) (padded - t), 0, 0);
    // The keys padded to [d_k, padded + 2w, heads], and block k's window, blocks k to k + 2 of them, [d_k, 3w, blocks, heads].
    ggml_tensor * kp = ggml_concat(ctx, g.input(std::vector<float>((size_t) (dk * w * heads_), 0.0f), dk, w, heads_), kh, 1);
    kp = ggml_pad(ctx, kp, 0, (int) (padded - t + w), 0, 0);
    auto key_blocks = [&](int64_t first) {
        return ggml_view_4d(ctx, kp, dk, w, blocks, heads_, kp->nb[1], w * kp->nb[1], kp->nb[2], (size_t) first * w * kp->nb[1]);
    };
    ggml_tensor * kw = ggml_concat(ctx, ggml_concat(ctx, key_blocks(0), key_blocks(1), 1), key_blocks(2), 1);
    ggml_tensor * ac = mul_mat(ctx, kw, qu);

    // When the utterance fits inside the context, a block of its length sees the same valid keys without padding
    // to the full context. The fixed positional projection is sliced around its zero position.
    ggml_tensor * positions = positions_->projected.at(name);
    positions = ggml_view_2d(ctx, positions, d_model_, 2 * w + 1, positions->nb[1], (size_t) (context_ - w) * positions->nb[1]);
    ggml_tensor * p = heads(ggml_reshape_3d(ctx, positions, dk, heads_, 2 * w + 1));
    ggml_tensor * bd = ggml_pad(ctx, mul_mat(ctx, p, qv), (int) (window - (2 * w + 1)), 0, 0, 0);
    bd = ggml_view_4d(ctx, bd, window, w, blocks, heads_, (size_t) (window - 1) * sizeof(float), (size_t) (w * window) * sizeof(float),
                      bd->nb[2], 0);
    ggml_tensor * scores = ggml_add(ctx, ac, ggml_cont(ctx, bd));

    // The values the same way, with time first, [3w, d_k, blocks, heads].
    ggml_tensor * vp = ggml_concat(ctx, g.input(std::vector<float>((size_t) (w * dk * heads_), 0.0f), w, dk, heads_), vt, 0);
    vp = ggml_pad(ctx, vp, (int) (padded - t + w), 0, 0, 0);
    auto value_blocks = [&](int64_t first) {
        return ggml_view_4d(ctx, vp, w, dk, blocks, heads_, vp->nb[1], w * sizeof(float), vp->nb[2], (size_t) (first * w) * sizeof(float));
    };
    ggml_tensor * vw = ggml_concat(ctx, ggml_concat(ctx, value_blocks(0), value_blocks(1), 0), value_blocks(2), 0);
    // _compute_global_key_attn(): every query against the global tokens' keys, without the biases or positions.
    ggml_tensor * qn = query_blocks(heads(q));
    ggml_tensor * kg = ggml_cont(ctx, ggml_view_4d(ctx, kh, dk, globals, 1, heads_, kh->nb[1], kh->nb[2], kh->nb[2], 0));
    scores = ggml_concat(ctx, mul_mat(ctx, kg, qn), scores, 0);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, in.mask, 1.0f / std::sqrt((float) dk), 0.0f);
    auto columns = [&](int64_t from, int64_t n) {
        return ggml_cont(ctx, ggml_view_4d(ctx, weights, n, w, blocks, heads_, weights->nb[1], weights->nb[2], weights->nb[3],
                                           (size_t) from * sizeof(float)));
    };
    ggml_tensor * vg = ggml_cont(ctx, ggml_view_4d(ctx, vt, globals, dk, 1, heads_, vt->nb[1], vt->nb[2], vt->nb[2], 0));
    ggml_tensor * out = ggml_add(ctx, mul_mat(ctx, vg, columns(0, globals)), mul_mat(ctx, vw, columns(globals, window)));
    // [d_k, w, blocks, heads] to [d_model, padded], then the frames of the utterance.
    out = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, out, dk, padded, heads_), 0, 2, 1, 3)), d_model_, padded);

    // _compute_out_global_to_all(): the global tokens' queries against every frame.
    ggml_tensor * qg = ggml_cont(ctx, ggml_permute(ctx, ggml_view_3d(ctx, q, dk, heads_, globals, q->nb[1], q->nb[2], 0), 0, 2, 1, 3));
    ggml_tensor * gw = ggml_soft_max_ext(ctx, mul_mat(ctx, kh, qg), nullptr, 1.0f / std::sqrt((float) dk), 0.0f);
    ggml_tensor * go = ggml_reshape_2d(ctx, ggml_cont(ctx, ggml_permute(ctx, mul_mat(ctx, vt, gw), 0, 2, 1, 3)), d_model_, globals);
    if (globals == t) return go;
    return ggml_concat(ctx, go, ggml_view_2d(ctx, out, d_model_, t - globals, out->nb[1], (size_t) globals * out->nb[1]), 1);
}

/**
 * ConformerConvolution.forward() (nemo/collections/asr/parts/submodules/conformer_modules.py): pointwise to twice
 * the channels, GLU, a depthwise convolution over time with symmetric zero padding, batch norm (folded into the
 * depthwise convolution by the converter), Swish and pointwise back. The depthwise convolution takes [channels, T]
 * as it is, an image one row high with the padding of its width, rather than transposed for ggml_ssm_conv() and
 * padded by two concatenations, which copy it three times.
 */
ggml_tensor * Encoder::convolution(Graph & g, ggml_tensor * x, const std::string & name) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = x->ne[1], pad = (conv_kernel_ - 1) / 2;
    ggml_tensor * h = layer_norm(ctx, x, name + "_norm");
    h = ggml_mul(ctx, linear(ctx, h, name + "_pw1_a", use_bias_), ggml_sigmoid(ctx, linear(ctx, h, name + "_pw1_gate", use_bias_)));
    ggml_tensor * kernel = ggml_reshape_4d(ctx, m_.tensor(name + "_dw.weight"), conv_kernel_, 1, 1, d_model_);
    h = ggml_reshape_2d(ctx, depthwise(ctx, ggml_reshape_3d(ctx, h, d_model_, t, 1), kernel, 1, (int) pad, 0), d_model_, t);
    h = ggml_silu(ctx, ggml_add(ctx, h, m_.tensor(name + "_dw.bias")));
    return ggml_add(ctx, x, linear(ctx, h, name + "_pw2", use_bias_));
}

ggml_tensor * Encoder::build(Graph & g, const std::vector<float> & features, int64_t frames, EncoderStages * stages) const {
    return build(g, g.input(features, mels_, frames), stages);
}

ggml_tensor * Encoder::build(Graph & g, ggml_tensor * features, EncoderStages * stages) const {
    ggml_context * ctx = g.ctx();
    ggml_tensor * x = subsample(g, features);
    if (stages) stages->subsampled = x;
    const int64_t t = x->ne[1];
    if (xscale_ != 1.0f) x = ggml_scale(ctx, x, xscale_);

    const AttentionInputs in = attention_inputs(g, t);

    // ConformerLayer.forward() (conformer_modules.py): a half-step feed-forward, attention, convolution, another
    // half-step feed-forward, each on a pre-norm residual, and a final norm.
    for (int l = 0; l < layers_; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        x = feed_forward(ctx, x, p + "ff1");
        x = attention(g, x, in, p + "attn");
        x = convolution(g, x, p + "conv");
        x = feed_forward(ctx, x, p + "ff2");
        x = layer_norm(ctx, x, p + "out_norm");
        if (stages) stages->layers.push_back(x);
    }
    return x;
}

}  // namespace fastconformer
