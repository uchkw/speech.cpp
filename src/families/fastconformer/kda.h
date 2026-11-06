#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "transducer.h"

namespace fastconformer {

/**
 * The prediction network of Haas, haas_asr.prednet.KDAPredictionNetwork: pre-norm residual blocks
 * x + kda(layer_norm(x)), then a layer norm where fastconformer.decoder.kda.final_norm says, each kda the
 * KimiDeltaAttention of flash-linear-attention 0.5.2 (fla/layers/kda.py) as it decodes one label, with
 * fused_recurrent_kda().
 *
 * A layer projects its input to q, k and v, each through a causal depthwise convolution of conv_size taps and a SiLU
 * (ShortConvolution), and to a forget gate, -exp(A_log) softplus(f_b f_a x + dt_bias) per head and key channel in log
 * space, and beta, sigmoid(b x) per head. q and k are divided by the square root of their squares' sum per head plus
 * qk_norm_eps, and the recurrence decays its state S of each head by exp(gate) along the keys, adds beta (v - S k)
 * outer k, and gives S q / sqrt(head_dim). Its output is normalized per head with an RMS norm, scaled by
 * sigmoid(g_b g_a x + bias) and projected back.
 *
 * The state of each layer is its recurrent state, [head_dim keys, head_dim values, heads] as ggml_gated_delta_net()
 * keeps it and fla does with state_v_first, followed by the last conv_size - 1 inputs of the convolutions of q, k and
 * v, [conv_size - 1, heads * head_dim] each, the oldest first. fla's own cache holds conv_size inputs, the oldest of
 * which no later step reads.
 */
class KdaLayers : public PredictionLayers {
public:
    explicit KdaLayers(const ModelFile & m, ggml_backend_t backend = nullptr, bool cache_constants = false);
    ~KdaLayers() override;

    int64_t state_size() const override { return layers_ * layer_state(); }
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * state, ggml_tensor ** next) const override;
    std::vector<std::string> matrices() const override;
    PredictionConstants constants() const override;

private:
    int64_t layer_state() const { return heads_ * head_dim_ * (head_dim_ + 3 * (conv_size_ - 1)); }

    const ModelFile & m_;
    int layers_;
    int64_t heads_, head_dim_, conv_size_;
    float norm_eps_, gate_norm_eps_, qk_norm_eps_;
    bool final_norm_;
    ggml_backend_t backend_;
    bool cache_constants_;
    struct Constants;
    mutable std::unique_ptr<Constants> constants_;
    mutable size_t preparations_ = 0;
};

}  // namespace fastconformer
