// The B=2 batched FastConformer encoder of experiments/haas-mac/032b. This file is the experiment's own copy of the
// global-attention part of src/families/fastconformer/encoder.cpp with the batch dimension carried through: the
// production encoder stays as it is, and the two are compared by the check in this folder.

#include "batch-encoder.h"

#include <cmath>
#include <map>
#include <stdexcept>
#include <string>

#include "error.h"

namespace fastconformer {

namespace {

/*
 * Activations are [channels, frames, 1, 2] (ne0 = channels). The subsampling runs on [channels, mels, frames, 2],
 * channels fastest as well, an image whose width is the mel axis and height the time axis, one image per utterance.
 */

/**
 * The depthwise convolution of `x` [channels, width, height, 2] by `kernel` [kw, kh, 1, channels], [channels, width',
 * height', 2]: ggml_conv_2d_dw_direct() on the channel-fastest layout, which its result keeps, with the kernel's taps
 * channel-fastest as well. This is encoder.cpp's helper with its permutations, which leave the batch axis alone.
 */
ggml_tensor * depthwise(ggml_context * ctx, ggml_tensor * x, ggml_tensor * kernel, int stride, int pad_w, int pad_h) {
    kernel = ggml_permute(ctx, ggml_cont(ctx, ggml_permute(ctx, kernel, 2, 3, 1, 0)), 3, 2, 0, 1);
    ggml_tensor * y = ggml_conv_2d_dw_direct(ctx, kernel, ggml_permute(ctx, x, 2, 0, 1, 3), stride, stride, pad_w, pad_h, 1, 1);
    return ggml_permute(ctx, y, 1, 2, 0, 3);
}

}  // namespace

/** Encoder::Positions' own copy: the same projection of the same encodings into the same 32 MiB bound. */
struct BatchEncoder::Positions {
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

BatchEncoder::~BatchEncoder() = default;

BatchEncoder::BatchEncoder(const ModelFile & m, ggml_backend_t backend)
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
    if (m.one_of("fastconformer.encoder.attention", {"rel_pos", "rel_pos_local_attn"}) != "rel_pos") {
        throw Error(Fault::File, m.path() + ": fastconformer.encoder.attention is not rel_pos; the batched pair encoder of 032b "
                                   "takes the model's global relative attention alone; " + m.remedy());
    }
    // Global attention can reuse the same central positions across input lengths, bound as in Encoder to 32 MiB;
    // longer inputs compute their positions in the utterance graph.
    const int64_t rows = 32 * 1024 * 1024 / ((int64_t) layers_ * d_model_ * (int64_t) sizeof(float));
    cached_span_ = rows > 0 ? (rows - 1) / 2 : -1;
    sub_layers_ = 0;
    for (uint32_t f = m.u32("fastconformer.encoder.subsampling_factor"); f > 1; f /= 2) sub_layers_++;
}

int64_t BatchEncoder::subsampled_frames(int64_t frames) const {
    for (int i = 0; i < sub_layers_; i++) frames = (frames - 1) / 2 + 1;
    return frames;
}

ggml_tensor * BatchEncoder::linear(ggml_context * ctx, ggml_tensor * x, const std::string & name, bool bias) const {
    ggml_tensor * y = mul_mat(ctx, m_.tensor(name + ".weight"), x);
    return bias ? ggml_add(ctx, y, m_.tensor(name + ".bias")) : y;
}

ggml_tensor * BatchEncoder::layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    return ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, eps_), m_.tensor(name + ".weight")), m_.tensor(name + ".bias"));
}

ggml_tensor * BatchEncoder::subsample(Graph & g, const std::vector<float> & pair, int64_t frames, BatchStages * stages) const {
    ggml_context * ctx = g.ctx();
    // ConvSubsampling "dw_striding" (nemo/collections/asr/parts/submodules/subsampling.py): a 3x3 convolution of
    // stride 2, then (depthwise 3x3 of stride 2, pointwise) for each further halving, each followed by a ReLU, each
    // utterance's image padded on its own. The first convolution, of one input channel, is the product of its kernel
    // [9, channels] with the input's im2col [9, positions], channel-fastest; the matrix input keeps each utterance's
    // own positions as its columns, [9, W*H, 1, 2], rather than folding the utterances into one longer column count.
    ggml_tensor * w0 = m_.tensor("sub.conv.0.weight");
    ggml_tensor * cols = ggml_im2col(ctx, w0, g.input(pair, mels_, frames, 1, 2), 2, 2, 1, 1, 1, 1, true, GGML_TYPE_F32);
    if (stages) stages->im2col = cols;
    const int64_t c = w0->ne[3];
    ggml_tensor * x = mul_mat(ctx, ggml_reshape_2d(ctx, w0, w0->ne[0] * w0->ne[1], c),
                              ggml_reshape_4d(ctx, cols, cols->ne[0], cols->ne[1] * cols->ne[2], 1, 2));
    x = ggml_reshape_4d(ctx, ggml_relu(ctx, ggml_add(ctx, x, m_.tensor("sub.conv.0.bias"))), c, cols->ne[1], cols->ne[2], 2);
    for (int i = 1; i < sub_layers_; i++) {
        const std::string p = "sub.conv." + std::to_string(i);
        ggml_tensor * y = depthwise(ctx, x, m_.tensor(p + ".dw.weight"), 2, 1, 1);
        if (stages && i == 1) stages->depthwise = y;
        x = ggml_add(ctx, y, m_.tensor(p + ".dw.bias"));
        const int64_t w = x->ne[1], h = x->ne[2];
        x = ggml_reshape_4d(ctx, ggml_relu(ctx, linear(ctx, ggml_reshape_4d(ctx, x, c, w * h, 1, 2), p + ".pw", true)), c, w, h, 2);
    }
    // ConvSubsampling.forward() flattens (channels, mels) with the mel axis fastest, one row of the flattened pairs
    // per utterance per frame.
    const int64_t w = x->ne[1], h = x->ne[2];
    x = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3)), w * c, h, 1, 2);
    return linear(ctx, x, "sub.out", true);
}

ggml_tensor * BatchEncoder::feed_forward(ggml_context * ctx, ggml_tensor * x, const std::string & name) const {
    ggml_tensor * h = ggml_silu(ctx, linear(ctx, layer_norm(ctx, x, name + "_norm"), name + "_up", use_bias_));
    return ggml_add(ctx, x, ggml_scale(ctx, linear(ctx, h, name + "_down", use_bias_), ff_factor_));
}

ggml_tensor * BatchEncoder::attention_inputs(Graph & g, int64_t t) const {
    // RelPositionalEncoding.extend_pe() (multi_head_attention.py) encodes the relative positions T - 1 down to
    // -(T - 1) through create_pe(): sines in the even channels and cosines in the odd, computed in float32, so the
    // angles round as they do here. The two utterances share one frame count and one set of positions.
    const int64_t needed = t - 1;
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
    return cached ? nullptr : g.input(pe, d_model_, 2 * span + 1);
}

/**
 * RelPositionMultiHeadAttention.forward() (nemo/collections/asr/parts/submodules/multi_head_attention.py) over the
 * two utterances at once:
 *
 *     scores = ((q + u) k^T + rel_shift((q + v) p^T)) / sqrt(d_k)
 *
 * where p is linear_pos of the encodings of the relative positions T - 1 down to -(T - 1), shared by both utterances
 * with batch 1, which MUL_MAT broadcasts over the pair. rel_shift() pads a zero column on the left, reads the [T, 2T]
 * result as [2T, T] and drops its first row; element (i, j) of the result is then element T - 1 - i + j of row i of
 * (q + v) p^T, which never reaches the zero column: element i (2T - 2) + j + T - 1 of [T, 2T - 1], a view whose rows
 * step by 2T - 2. The head and batch strides of the raw product stay its own, so no row of one utterance reads
 * another's; at T = 1 the row stride is zero and the single row is still one valid row.
 */
ggml_tensor * BatchEncoder::attention(Graph & g, ggml_tensor * x, ggml_tensor * pos, const std::string & name,
                                      BatchStages * stages) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = x->ne[1], dk = d_model_ / heads_;
    ggml_tensor * h = layer_norm(ctx, x, name + "_norm");
    ggml_tensor * q = ggml_reshape_4d(ctx, linear(ctx, h, name + "_q", use_bias_), dk, heads_, t, 2);
    ggml_tensor * k = ggml_reshape_4d(ctx, linear(ctx, h, name + "_k", use_bias_), dk, heads_, t, 2);
    ggml_tensor * v = ggml_reshape_4d(ctx, linear(ctx, h, name + "_v", use_bias_), dk, heads_, t, 2);

    // [d_k, heads, n, batch] to [d_k, n, heads, batch]; the shared positions come with batch 1 and stay so.
    auto heads = [&](ggml_tensor * y, int64_t n) {
        return ggml_permute(ctx, ggml_reshape_4d(ctx, y, dk, heads_, n, y->ne[3]), 0, 2, 1, 3);
    };
    ggml_tensor * projected;
    if (pos) projected = mul_mat(ctx, m_.tensor(name + "_pos.weight"), pos);
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
    bd = ggml_view_4d(ctx, bd, t, t, heads_, 2, (size_t) (2 * t - 2) * sizeof(float), bd->nb[2], bd->nb[3],
                      (size_t) (t - 1) * sizeof(float));
    ggml_tensor * scores = ggml_add(ctx, ac, bd);
    ggml_tensor * weights = ggml_soft_max_ext(ctx, scores, nullptr, 1.0f / std::sqrt((float) dk), 0.0f);
    if (stages && !stages->scores) stages->scores = weights;

    // [d_k, heads, T] to [T, d_k, heads] per utterance, so that the product contracts over the keys.
    ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor * out = mul_mat(ctx, vt, weights);
    out = ggml_reshape_4d(ctx, ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3)), d_model_, t, 1, 2);
    return ggml_add(ctx, x, linear(ctx, out, name + "_out", use_bias_));
}

/**
 * ConformerConvolution.forward() (nemo/collections/asr/parts/submodules/conformer_modules.py): pointwise to twice
 * the channels, GLU, a depthwise convolution over time with symmetric zero padding — each utterance's frames padded
 * on their own, batch norm folded into the depthwise convolution by the converter — Swish and pointwise back.
 */
ggml_tensor * BatchEncoder::convolution(Graph & g, ggml_tensor * x, const std::string & name) const {
    ggml_context * ctx = g.ctx();
    const int64_t t = x->ne[1], pad = (conv_kernel_ - 1) / 2;
    ggml_tensor * h = layer_norm(ctx, x, name + "_norm");
    h = ggml_mul(ctx, linear(ctx, h, name + "_pw1_a", use_bias_), ggml_sigmoid(ctx, linear(ctx, h, name + "_pw1_gate", use_bias_)));
    ggml_tensor * kernel = ggml_reshape_4d(ctx, m_.tensor(name + "_dw.weight"), conv_kernel_, 1, 1, d_model_);
    h = depthwise(ctx, h, kernel, 1, (int) pad, 0);
    h = ggml_silu(ctx, ggml_add(ctx, h, m_.tensor(name + "_dw.bias")));
    return ggml_add(ctx, x, linear(ctx, h, name + "_pw2", use_bias_));
}

ggml_tensor * BatchEncoder::build(Graph & g, const std::vector<float> & a, int64_t frames_a, const std::vector<float> & b,
                                  int64_t frames_b, BatchStages * stages) const {
    if (frames_a <= 0 || frames_b <= 0) {
        throw Error(Fault::InvalidArgument, "a batched pair needs a positive number of frontend frames", "frames");
    }
    // Equal post-subsampling lengths are not enough: unequal frontend frame counts keep different convolution
    // boundaries, so the pair takes one frame count.
    if (frames_a != frames_b) {
        throw Error(Fault::InvalidArgument, "a batched pair needs one frontend frame count, not " + std::to_string(frames_a) + " and " +
                                                std::to_string(frames_b), "frames");
    }
    const int64_t frames = frames_a;
    if ((int64_t) a.size() != frames * mels_ || (int64_t) b.size() != frames * mels_) {
        throw Error(Fault::InvalidArgument, "the features of a batched pair do not hold their frame count of " + std::to_string(mels_) +
                                                " mel bins", "features");
    }
    ggml_context * ctx = g.ctx();
    // The two utterances' features in one input, [mels, frames, 1, 2]: the first half of the data the first
    // utterance's, the second half the second's, each [frames, mels] row-major as the frontend gives it.
    std::vector<float> pair((size_t) (frames * mels_ * 2));
    for (int which = 0; which < 2; which++) {
        const std::vector<float> & features = which ? b : a;
        for (int64_t f = 0; f < frames; f++) {
            for (int m = 0; m < mels_; m++) pair[(size_t) (m + mels_ * (f + frames * which))] = features[(size_t) (f * mels_ + m)];
        }
    }
    ggml_tensor * x = subsample(g, pair, frames, stages);
    if (stages) stages->subsampled = x;
    const int64_t t = x->ne[1];
    if (xscale_ != 1.0f) x = ggml_scale(ctx, x, xscale_);

    ggml_tensor * pos = attention_inputs(g, t);

    // ConformerLayer.forward() (conformer_modules.py): a half-step feed-forward, attention, convolution, another
    // half-step feed-forward, each on a pre-norm residual, and a final norm.
    for (int l = 0; l < layers_; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        x = feed_forward(ctx, x, p + "ff1");
        x = attention(g, x, pos, p + "attn", stages);
        x = convolution(g, x, p + "conv");
        x = feed_forward(ctx, x, p + "ff2");
        x = layer_norm(ctx, x, p + "out_norm");
        if (stages) stages->layers.push_back(x);
    }
    return x;
}

std::pair<std::vector<float>, std::vector<float>> split_batches(const ggml_tensor * t) {
    if (t->type != GGML_TYPE_F32 || t->ne[3] != 2 || !ggml_is_contiguous(t)) {
        throw std::runtime_error("a batched pair's result is not a contiguous float32 tensor of batch 2");
    }
    const std::vector<float> all = Graph::read(t);
    const size_t half = t->nb[3] / sizeof(float);
    if (all.size() != 2 * half) throw std::runtime_error("a batched pair's result does not hold two blocks of its batch stride");
    return {{all.begin(), all.begin() + half}, {all.begin() + half, all.end()}};
}

}  // namespace fastconformer
