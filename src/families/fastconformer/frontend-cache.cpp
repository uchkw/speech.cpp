#include "frontend-cache.h"

#include <algorithm>
#include <cstring>
#include <utility>

namespace fastconformer {

FrontendCache::Reading FrontendCache::prepare(const std::vector<float> & samples) const {
    Reading reading;
    if (samples.size() > maximum_samples_) {
        reading.features = frontend_.features(samples);
        return reading;
    }

    // Only an equal or longer input can continue the previous prefix. Byte comparison distinguishes signed zeros
    // and compares the actual resampled or enhanced samples, rather than assuming that their source is unchanged.
    size_t unchanged = 0;
    if (!samples_.empty() && samples.size() >= samples_.size()) {
        if (std::memcmp(samples_.data(), samples.data(), samples_.size() * sizeof(float)) == 0) {
            unchanged = samples_.size();
        } else {
            while (unchanged < samples_.size() && std::memcmp(&samples_[unchanged], &samples[unchanged], sizeof(float)) == 0) unchanged++;
        }
    }
    const int64_t reach = frontend_.window_end(0);
    if ((int64_t) unchanged >= reach) {
        reading.reused_frames = std::min(frontend_.frames(samples_.size()),
                                        ((int64_t) unchanged - reach) / frontend_.hop_length() + 1);
    }
    if (reading.reused_frames == 0) {
        // The original path also keeps short-input validation and its error attribution unchanged.
        reading.raw_mel = frontend_.log_mels(samples);
    } else {
        const int64_t frames = frontend_.frames(samples.size());
        std::vector<float> emphasized(samples.size());
        frontend_.preemphasize(samples.data(), samples.size(), true, 0.0f, emphasized.data());
        reading.raw_mel.resize((size_t) (frames * frontend_.mels()));
        const size_t reused = (size_t) (reading.reused_frames * frontend_.mels());
        std::copy_n(raw_mel_.begin(), reused, reading.raw_mel.begin());
        frontend_.log_mel(emphasized.data(), 0, (int64_t) emphasized.size(), reading.reused_frames, frames - reading.reused_frames,
                          reading.raw_mel.data() + reused);
    }
    reading.features = frontend_.normalized(reading.raw_mel);
    reading.samples = samples;
    return reading;
}

void FrontendCache::publish(Reading && reading) noexcept {
    samples_ = std::move(reading.samples);
    raw_mel_ = std::move(reading.raw_mel);
}

}  // namespace fastconformer
