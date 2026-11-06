#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** Intermediate results of the encoder, for the checks. */
struct EncoderStages {
    /** The subsampling's output before the input scaling, [d_model, frames / factor]. */
    ggml_tensor * subsampled = nullptr;
    /** The output of each conformer layer, [d_model, frames / factor] each. */
    std::vector<ggml_tensor *> layers;
};

/**
 * NeMo's ConformerEncoder with dw_striding subsampling and relative positional self-attention, with or without the
 * biases of its conformer layers (use_bias). The attention is fastconformer.encoder.attention: NeMo's
 * self_attention_model "rel_pos" over the whole utterance (att_context_size [-1, -1]), or "rel_pos_local_attn",
 * Longformer's attention over the frames within fastconformer.encoder.attention_context of each frame on either side,
 * with fastconformer.encoder.global_tokens frames from the first on that every frame attends to and that attend to
 * every frame.
 *
 * ConformerEncoder.forward_internal() (nemo/collections/asr/modules/conformer_encoder.py) runs on a batch padded to
 * its longest utterance and masks the padding out of every convolution and attention; MaskedConvSequential
 * (parts/submodules/subsampling.py) zeroes the subsampling's padded frames between its layers. Every valid output
 * depends on the valid frames and on zeros alone, so this runs on the valid frames of one utterance.
 */
class Encoder {
public:
    Encoder(const ModelFile & m, ggml_backend_t backend);
    ~Encoder();

    /** The encoder's output for `features` ([frames, mels] row-major), [d_model, subsampled_frames(frames)]. */
    ggml_tensor * build(Graph & g, const std::vector<float> & features, int64_t frames, EncoderStages * stages = nullptr) const;
    /** The same encoder from a graph's F32 feature input [mels, frames], which its owner can update between runs. */
    ggml_tensor * build(Graph & g, ggml_tensor * features, EncoderStages * stages = nullptr) const;

    /** The frames after subsampling: calc_length() of the three stride-2 convolutions. */
    int64_t subsampled_frames(int64_t frames) const;

    int d_model() const { return d_model_; }
    int subsampling_factor() const { return 1 << sub_layers_; }

private:
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name, bool bias) const;
    ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * subsample(Graph & g, ggml_tensor * features) const;
    ggml_tensor * feed_forward(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    /** The inputs every layer's attention shares: the positional encodings and, for local attention, its mask. */
    struct AttentionInputs {
        ggml_tensor * pos;
        ggml_tensor * mask;
    };
    AttentionInputs attention_inputs(Graph & g, int64_t t) const;
    ggml_tensor * attention(Graph & g, ggml_tensor * x, const AttentionInputs & in, const std::string & name) const;
    ggml_tensor * local_attention(Graph & g, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v, const AttentionInputs & in,
                                  const std::string & name) const;
    ggml_tensor * convolution(Graph & g, ggml_tensor * x, const std::string & name) const;

    /** The projections of attention's fixed positions, shared by every utterance. */
    struct Positions;
    mutable std::unique_ptr<Positions> positions_;
    ggml_backend_t backend_;
    const ModelFile & m_;
    int mels_, d_model_, layers_, heads_, conv_kernel_, sub_layers_;
    float eps_, pos_base_, xscale_, ff_factor_;
    /** Whether the conformer layers' linear layers and pointwise convolutions have biases; the subsampling's always do. */
    bool use_bias_;
    /** Whether the attention is local; the frames each frame sees on either side and the global tokens if it is. */
    bool local_;
    int context_ = 0, global_tokens_ = 0;
    int64_t cached_span_ = 0;
};

}  // namespace fastconformer
