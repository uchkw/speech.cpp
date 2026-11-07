#include "tdt-prefix.h"

#include <stdexcept>
#include <utility>

namespace fastconformer {

TdtPrefixCache::Reading::Reading(const TdtPrefixCache & source, size_t samples) : source_(source), samples_(samples) {
    entries_.reserve(kEntries);
}

void TdtPrefixCache::Reading::bind(const Decoder * decoder, ggml_backend_t backend) {
    if (decoder_) throw std::logic_error("a prediction prefix candidate belongs to one decoding");
    decoder_ = decoder;
    backend_ = backend;
    matching_ = source_.decoder_ == decoder && source_.backend_ == backend;
}

const TdtPrefixCache::Entry * TdtPrefixCache::Reading::find(int32_t label) {
    if (!matching_) return nullptr;
    if (step_ >= source_.entries_.size() || source_.entries_[step_].label != label) {
        matching_ = false;
        return nullptr;
    }
    hits_++;
    return &source_.entries_[step_];
}

void TdtPrefixCache::Reading::append(int32_t label, const PredictionState & state, const std::vector<float> & prediction,
                                    int32_t next_label) {
    if (entries_.size() < kEntries) entries_.push_back({label, state, prediction, next_label});
    // A match includes the label fed to this prediction. The joint's newly emitted label belongs to the next
    // prediction's key, even when its duration ends this request before that prediction is ever evaluated. Where
    // the source holds no entry at this step there is nothing to verify against, which ends matching as a
    // mismatch does.
    if (matching_ && (step_ >= source_.entries_.size() || source_.entries_[step_].next_label != next_label)) matching_ = false;
    step_++;
}

TdtPrefixCache::Reading TdtPrefixCache::begin(size_t samples) {
    if (samples < samples_) clear();
    return Reading(*this, samples);
}

void TdtPrefixCache::publish(Reading && reading) {
    if (&reading.source_ != this || !reading.decoder_ || !reading.complete_) {
        throw std::logic_error("cannot publish an incomplete prediction prefix or one from another cache");
    }
    entries_ = std::move(reading.entries_);
    samples_ = reading.samples_;
    decoder_ = reading.decoder_;
    backend_ = reading.backend_;
}

void TdtPrefixCache::clear() noexcept {
    entries_.clear();
    samples_ = 0;
    decoder_ = nullptr;
    backend_ = nullptr;
}

}  // namespace fastconformer
