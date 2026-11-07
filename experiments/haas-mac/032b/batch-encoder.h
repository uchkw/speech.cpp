#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** Intermediate results of the batched pair encoder, for the check of experiments/haas-mac/032b. */
struct BatchStages {
    /** The subsampling's output before the input scaling, [d_model, T, 1, 2]. */
    ggml_tensor * subsampled = nullptr;
    /** The output of each conformer layer, [d_model, T, 1, 2] each; the last is the encoder's output. */
    std::vector<ggml_tensor *> layers;
    /**
     * The first convolution's im2col [9, W, H, 2], the first depthwise convolution's output [C, W', H', 2] and the
     * first layer's attention weights [T, T, heads, 2]: evidence that one graph path carries both utterances, which
     * the check asserts against, where two independently built graphs would carry batch 1.
     */
    ggml_tensor * im2col = nullptr;
    ggml_tensor * depthwise = nullptr;
    ggml_tensor * scores = nullptr;
};

/**
 * The experiment of experiments/haas-mac/032b: fastconformer::Encoder's graph for exactly two utterances of one
 * frontend frame count in one batched graph, ne[3] = 2 on every activation and one operation for each stage, where
 * the production encoder takes one utterance at a time. It reads the same tensors in the same types with the same
 * arithmetic and the same accumulation requests, keeps each utterance's convolution padding and its matrices' M, N
 * and K, and shares the position projections between the two utterances. The frontend stays outside it: each
 * utterance's features are normalized on their own, as NemoFrontend does.
 *
 * It takes models of global relative attention alone (fastconformer.encoder.attention "rel_pos") and refuses any
 * other attention when it is constructed, and it builds nothing for one utterance: the differential baseline is
 * Encoder itself, run on each utterance separately.
 */
class BatchEncoder {
public:
    BatchEncoder(const ModelFile & m, ggml_backend_t backend);
    ~BatchEncoder();
    BatchEncoder(const BatchEncoder &) = delete;
    BatchEncoder & operator=(const BatchEncoder &) = delete;

    /**
     * The encoder's output for the feature arrays `a` and `b` ([frames, mels] row-major each), [d_model,
     * subsampled_frames(frames), 1, 2]. Frame counts that are nonpositive or unequal, and features that do not hold
     * their frame count of mel bins, throw an Error of the caller before any tensor is made.
     */
    ggml_tensor * build(Graph & g, const std::vector<float> & a, int64_t frames_a, const std::vector<float> & b, int64_t frames_b,
                        BatchStages * stages = nullptr) const;

    /** The frames after subsampling: calc_length() of the stride-2 convolutions, as Encoder's. */
    int64_t subsampled_frames(int64_t frames) const;

    /** The span of relative positions on either side of zero whose projections the cache holds, as Encoder's bound gives it. */
    int64_t cached_span() const { return cached_span_; }

private:
    ggml_tensor * linear(ggml_context * ctx, ggml_tensor * x, const std::string & name, bool bias) const;
    ggml_tensor * layer_norm(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    ggml_tensor * subsample(Graph & g, const std::vector<float> & pair, int64_t frames, BatchStages * stages) const;
    ggml_tensor * feed_forward(ggml_context * ctx, ggml_tensor * x, const std::string & name) const;
    /**
     * The encodings of the relative positions every layer's attention shares, as a graph input of batch 1 when they
     * are longer than the cache holds, or null when the cache holds their projections.
     */
    ggml_tensor * attention_inputs(Graph & g, int64_t t) const;
    ggml_tensor * attention(Graph & g, ggml_tensor * x, ggml_tensor * pos, const std::string & name, BatchStages * stages) const;
    ggml_tensor * convolution(Graph & g, ggml_tensor * x, const std::string & name) const;

    /** The projections of attention's fixed positions, shared by every utterance; Encoder::Positions' own copy. */
    struct Positions;
    mutable std::unique_ptr<Positions> positions_;
    ggml_backend_t backend_;
    const ModelFile & m_;
    int mels_, d_model_, layers_, heads_, conv_kernel_, sub_layers_;
    float eps_, pos_base_, xscale_, ff_factor_;
    /** Whether the conformer layers' linear layers and pointwise convolutions have biases; the subsampling's always do. */
    bool use_bias_;
    /** Half the positions each side of zero whose projections the cache retains, as Encoder's 32 MiB bound gives it. */
    int64_t cached_span_ = 0;
};

/**
 * A computed batched result [., ., 1, 2] as the two utterances' halves: the whole tensor read back and split at its
 * own batch stride nb[3], after checking that it is float32, of batch 2 and contiguous, so that one half cannot hold
 * any of the other's values. Throws when it is not such a tensor.
 */
std::pair<std::vector<float>, std::vector<float>> split_batches(const ggml_tensor * t);

}  // namespace fastconformer
