#pragma once

#include <memory>

#include "transducer.h"

namespace fastconformer {

/** A fixed-shape prediction and joint graph, reused with new labels, prediction states and encoder frames. */
class DecoderStep {
public:
    DecoderStep(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend, int64_t frames);

    /** Computes the next prediction and the joint at each of the given projected encoder frames. */
    void compute(int32_t label, const PredictionState & state, const std::vector<float> & frames);
    PredictionState state() const { return PredictionNetwork::read_state(step_); }
    std::vector<float> prediction() const { return Graph::read(predicted_); }
    void logits(std::vector<float> & out) const { Graph::read(logits_, out); }

private:
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
};

/** A fixed-shape joint graph for successive TDT blank frames within one request. */
class JointStep {
public:
    JointStep(const Joint & joint, ggml_backend_t backend);

    void compute(const std::vector<float> & frame, const std::vector<float> & prediction);
    std::vector<float> logits() const { return Graph::read(logits_); }

private:
    ggml_backend_t backend_;
    Graph graph_{64};
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator_{nullptr, ggml_gallocr_free};
    ggml_tensor * frame_;
    ggml_tensor * prediction_;
    ggml_tensor * logits_;
    bool allocated_ = false;
};

}  // namespace fastconformer
