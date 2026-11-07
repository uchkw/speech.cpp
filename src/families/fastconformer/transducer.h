#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "graph.h"
#include "model-file.h"

namespace fastconformer {

/** The state of the prediction network's layers after a label, [PredictionLayers::state_size()] in their layout. */
using PredictionState = std::vector<float>;

/** The completed preparation, with read-only tensors that remain valid while their prediction network lives. */
struct PredictionConstants {
    size_t preparations = 0;
    size_t allocation_bytes = 0;
    ggml_backend_t backend = nullptr;
    std::vector<const ggml_tensor *> values;
};

/**
 * The layers of a prediction network after its embedding, by fastconformer.decoder.prediction: one step takes the
 * embedding of the last label and the state the labels before it left, and gives the output and the next state.
 */
class PredictionLayers {
public:
    virtual ~PredictionLayers() = default;

    /** The floats of the state, which is zeros before the first label. */
    virtual int64_t state_size() const = 0;

    /** The output [hidden] for the embedding `x` [hidden] from `state` [state_size()], with the next state in `next`. */
    virtual ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * state, ggml_tensor ** next) const = 0;

    /** The matrices a step multiplies by. */
    virtual std::vector<std::string> matrices() const = 0;

    /** The completed preparation, without creating a cache or computing on a backend. */
    virtual PredictionConstants constants() const { return {}; }
};

/**
 * The layers of fastconformer.decoder.prediction in the file `m`. Constructors only read metadata; an enabled
 * constant cache applies only when `backend` is a CPU backend and is prepared when the first step is built, after
 * its CPU pool has been installed. Any other backend keeps the original per-step expression and prepares nothing.
 */
std::unique_ptr<PredictionLayers> make_prediction_layers(const ModelFile & m, ggml_backend_t backend = nullptr,
                                                       bool cache_constants = false);

/**
 * Whether the tensor `name` is one that a decoding reads: the prediction network's and the joint's, but for the joint's
 * projection of the encoder, which the encoder's graph computes.
 */
bool decoding_tensor(const std::string & name);

/**
 * Whether every matrix a decoding multiplies by, the prediction network's layers' and the joint's projection of the
 * prediction and output layer, is quantized in the file `m`: Q8_0 or a K-quant rather than F16 or F32.
 */
bool quantized_decoding(const ModelFile & m);

/**
 * NeMo's RNNTDecoder (nemo/collections/asr/modules/rnnt.py) with the blank as the embedding's padding: an
 * embedding of the tokens and the blank, whose blank row is zero, then the layers of make_prediction_layers(). One step
 * takes the last label and the state and gives the last layer's output. The blank fed first is the start of the
 * sequence.
 */
class PredictionNetwork {
public:
    explicit PredictionNetwork(const ModelFile & m, ggml_backend_t backend = nullptr, bool cache_constants = false);

    /** The tensors of one step: the output [hidden] and the next state [state_size()]. */
    struct Step {
        ggml_tensor * output;
        ggml_tensor * state;
    };

    /** The state before the first label: zeros, as NeMo's LSTM and fla's KDA layers start when they are given none. */
    PredictionState initial_state() const { return PredictionState((size_t) layers_->state_size(), 0.0f); }

    Step build(Graph & g, int32_t label, const PredictionState & state) const;

    /** Builds a step from tensor inputs so that a decoder can reuse its graph for subsequent labels. */
    Step build(Graph & g, ggml_tensor * label, ggml_tensor * state) const;

    /** The next state of a step that has been computed. */
    static PredictionState read_state(const Step & step) { return Graph::read(step.state); }

    int hidden() const { return hidden_; }
    int64_t state_size() const { return layers_->state_size(); }
    /** The prepared constants owned by these layers, without starting preparation. */
    PredictionConstants constants() const { return layers_->constants(); }

private:
    const ModelFile & m_;
    std::unique_ptr<PredictionLayers> layers_;
    int hidden_;
};

/**
 * NeMo's RNNTJoint: the encoder's frames and the prediction network's output each projected to the joint's width,
 * summed, a ReLU and a linear layer to the outputs, the tokens and the blank then, for TDT, one per duration.
 * NeMo applies a log-softmax over all outputs on the CPU only, which changes no argmax and is left out.
 */
class Joint {
public:
    explicit Joint(const ModelFile & m);

    /** The encoder's output [d_model, T] projected, [hidden, T]; computed once per utterance. */
    ggml_tensor * project_encoder(ggml_context * ctx, ggml_tensor * encoded) const;
    /** The prediction network's output [prediction hidden, n] projected, [hidden, n]. */
    ggml_tensor * project_prediction(ggml_context * ctx, ggml_tensor * prediction) const;

    /**
     * The logits of projected frames `f` ([hidden, n]) with projected predictions `g`, one for all frames ([hidden])
     * or one each ([hidden, n]), [outputs, n].
     */
    ggml_tensor * build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const;

    int hidden() const { return hidden_; }
    int outputs() const { return outputs_; }

private:
    const ModelFile & m_;
    int hidden_, outputs_;
};

}  // namespace fastconformer
