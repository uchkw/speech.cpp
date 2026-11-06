#include "tdt-argmax.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "error.h"

namespace fastconformer {

TdtArgmax::TdtArgmax(Graph & graph, ggml_tensor * logits, int blank, ggml_backend_t backend)
    : logits_(logits) {
    if (!logits || logits->type != GGML_TYPE_F32 || !ggml_is_contiguous(logits) ||
        ggml_nelements(logits) != logits->ne[0] || !(logits->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        logits->ne[0] > std::numeric_limits<int32_t>::max() || blank < 0 || blank >= logits->ne[0] - 1) {
        throw std::logic_error("TDT argmax requires one retained F32 joint row with token, blank and duration outputs");
    }
    label_count_ = blank + 1;
    duration_count_ = (int) logits->ne[0] - label_count_;
    auto * labels = ggml_view_1d(graph.ctx(), logits, label_count_, 0);
    auto * durations = ggml_view_1d(graph.ctx(), logits, duration_count_, (size_t) label_count_ * sizeof(float));
    labels_ = ggml_argmax(graph.ctx(), labels);
    durations_ = ggml_argmax(graph.ctx(), durations);
    if (!ggml_backend_supports_op(backend, labels_) || !ggml_backend_supports_op(backend, durations_)) {
        throw Error(Fault::Device, "the decoding backend cannot compute TDT argmax; disable SPEECH_HAAS_TDT_DEVICE_ARGMAX");
    }
    graph.output(labels_);
    graph.output(durations_);
}

TdtDecision TdtArgmax::read() const {
    if (!logits_) throw std::logic_error("TDT argmax has no joint row");
    TdtDecision out;
    float label_first, duration_first;
    ggml_backend_tensor_get(labels_, &out.label, 0, sizeof(out.label));
    ggml_backend_tensor_get(durations_, &out.duration_index, 0, sizeof(out.duration_index));
    ggml_backend_tensor_get(logits_, &label_first, 0, sizeof(label_first));
    ggml_backend_tensor_get(logits_, &duration_first, (size_t) label_count_ * sizeof(float), sizeof(duration_first));
    // first_argmax starts at index zero, so a leading NaN prevents every later comparison from winning.
    if (std::isnan(label_first)) out.label = 0;
    if (std::isnan(duration_first)) out.duration_index = 0;
    if (out.label < 0 || out.label >= label_count_ || out.duration_index < 0 || out.duration_index >= duration_count_) {
        throw Error(Fault::Device, "TDT argmax returned an index outside its joint output range");
    }
    return out;
}

}  // namespace fastconformer
