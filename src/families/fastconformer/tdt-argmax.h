#pragma once

#include "graph.h"

namespace fastconformer {

/** The token or blank and the index into the TDT duration table. */
struct TdtDecision {
    int32_t label;
    int32_t duration_index;
};

/** Two reductions of one retained joint row, whose tensors live in the caller's graph. */
class TdtArgmax {
public:
    TdtArgmax() = default;
    TdtArgmax(Graph & graph, ggml_tensor * logits, int blank, ggml_backend_t backend);

    /** Reads two indices and two leading floats, keeping first_argmax's first-NaN behavior. */
    TdtDecision read() const;

private:
    ggml_tensor * logits_ = nullptr;
    ggml_tensor * labels_ = nullptr;
    ggml_tensor * durations_ = nullptr;
    int label_count_ = 0;
    int duration_count_ = 0;
};

}  // namespace fastconformer
