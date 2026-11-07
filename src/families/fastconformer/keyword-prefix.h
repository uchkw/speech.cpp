#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "transducer.h"

namespace fastconformer {

class TdtDecoder;

/** The first predictions of one request's plain decoding, borrowed by its boosted decoding on the same backend. */
class KeywordPrefix {
public:
    /** The state after feeding `label`, and that step's output through joint.pred, both complete F32 host copies. */
    struct Entry {
        int32_t label;
        PredictionState state;
        std::vector<float> prediction;
    };

    /**
     * At most this many prediction steps are kept. The bound is the payload alone: with Haas's state of 46,720 floats
     * and joint.pred output of 640, 64 records hold 11.5625 MiB.
     */
    static constexpr size_t kCapacity = 64;

    const std::vector<Entry> & entries() const { return entries_; }

    /** Allocated float payload alone, excluding the entry vector, allocator overhead and the live decoding's copies. */
    size_t payload_bytes() const {
        size_t bytes = 0;
        for (const auto & entry : entries_) bytes += (entry.state.capacity() + entry.prediction.capacity()) * sizeof(float);
        return bytes;
    }

private:
    friend class TdtDecoder;
    const TdtDecoder * decoder_ = nullptr;
    ggml_backend_t backend_ = nullptr;
    std::vector<Entry> entries_;
};

}  // namespace fastconformer
