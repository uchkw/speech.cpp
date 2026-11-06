#include "decoder-step.h"

#include <stdexcept>

#include "error.h"

namespace fastconformer {

DecoderState::DecoderState(const PredictionNetwork & prediction, ggml_backend_t backend) {
    context_.reset(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    if (!context_) throw Error(Fault::OutOfMemory, "cannot create a prediction state context");
    state_ = ggml_new_tensor_1d(context_.get(), GGML_TYPE_F32, prediction.state_size());
    ggml_set_input(state_);
    buffer_.reset(ggml_backend_alloc_ctx_tensors(context_.get(), backend));
    if (!buffer_) throw Error(Fault::OutOfMemory, "cannot allocate the prediction state");
    ggml_backend_tensor_memset(state_, 0, 0, ggml_nbytes(state_));
}

DecoderStep::DecoderStep(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend, int64_t frames,
                         DecoderState * resident)
    : backend_(backend), allocator_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free),
      resident_(resident != nullptr) {
    if (!allocator_) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
    label_ = graph_.input(std::vector<int32_t>{0}, 1);
    state_ = resident ? resident->tensor() : graph_.input(prediction.initial_state(), prediction.state_size());
    frames_ = graph_.input(std::vector<float>((size_t) (joint.hidden() * frames)), joint.hidden(), frames);
    step_ = prediction.build(graph_, label_, state_);
    predicted_ = joint.project_prediction(graph_.ctx(), step_.output);
    logits_ = joint.build(graph_.ctx(), frames_, predicted_);
    graph_.output(logits_);
    graph_.output(predicted_);
    graph_.output(step_.state);
    // The copy follows every reader of the old state and commits the complete recurrent and convolution state.
    if (resident) graph_.copy(step_.state, state_);
}

void DecoderStep::compute(int32_t label, const PredictionState & state, const std::vector<float> & frames) {
    graph_.set(state_, state);
    run(label, frames);
}

void DecoderStep::compute(int32_t label, const std::vector<float> & frames) {
    if (!resident_) throw std::logic_error("a decoder step without resident state needs a prediction state");
    run(label, frames);
}

void DecoderStep::run(int32_t label, const std::vector<float> & frames) {
    graph_.set(label_, std::vector<int32_t>{label});
    graph_.set(frames_, frames);
    if (allocated_) graph_.compute_again(backend_);
    else {
        graph_.compute(backend_, allocator_.get());
        allocated_ = true;
    }
}

}  // namespace fastconformer
