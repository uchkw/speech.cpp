#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "nemo-frontend.h"

namespace fastconformer {

/** Reuses raw mel frames of the last successful recognition, up to 30 seconds, under the owning model's request lock. */
class FrontendCache {
public:
    /** A request's prepared features and unpublished cache entry; discarding it leaves the previous entry intact. */
    struct Reading {
        std::vector<float> features;
        std::vector<float> samples;
        std::vector<double> raw_mel;
        int64_t reused_frames = 0;
    };

    /** The frontend is fixed, normalizes per feature, and outlives this cache. */
    explicit FrontendCache(const NemoFrontend & frontend)
        : frontend_(frontend), maximum_samples_((size_t) frontend.sample_rate() * 30) {}

    /** Computes the current whole-input normalization, reusing only frames before its verified immutable boundary. */
    Reading prepare(const std::vector<float> & samples) const;

    /** Publishes a successful reading; an input longer than 30 seconds clears the previous entry. */
    void publish(Reading && reading) noexcept;

private:
    const NemoFrontend & frontend_;
    const size_t maximum_samples_;
    std::vector<float> samples_;
    std::vector<double> raw_mel_;
};

}  // namespace fastconformer
