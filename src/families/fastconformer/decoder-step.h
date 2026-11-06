#pragma once

#include <memory>

#include "transducer.h"

namespace fastconformer {

/** A request's prediction state, initialized once and kept in a buffer outside its step graphs. */
class DecoderState {
public:
    DecoderState(const PredictionNetwork & prediction, ggml_backend_t backend);
    ggml_tensor * tensor() const { return state_; }

private:
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context_{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer_{nullptr, ggml_backend_buffer_free};
    ggml_tensor * state_ = nullptr;
};

/** A fixed-shape prediction and joint graph, reused with new labels, prediction states and encoder frames. */
class DecoderStep {
public:
    /** A resident state is committed after every step and must outlive this graph. */
    DecoderStep(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend, int64_t frames,
                DecoderState * resident = nullptr);

    /** Computes the next prediction and the joint at each of the given projected encoder frames. */
    void compute(int32_t label, const PredictionState & state, const std::vector<float> & frames);
    /** Computes and commits a step from the resident state, without uploading or reading it on the host. */
    void compute(int32_t label, const std::vector<float> & frames);
    PredictionState state() const { return PredictionNetwork::read_state(step_); }
    std::vector<float> prediction() const { return Graph::read(predicted_); }
    void logits(std::vector<float> & out) const { Graph::read(logits_, out); }

private:
    void run(int32_t label, const std::vector<float> & frames);

    ggml_backend_t backend_;
    Graph graph_{512};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_tensor * label_;
    ggml_tensor * state_;
    ggml_tensor * frames_;
    ggml_tensor * predicted_;
    ggml_tensor * logits_;
    PredictionNetwork::Step step_;
    bool allocated_ = false;
    bool resident_ = false;
};

}  // namespace fastconformer
