#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "fft.h"
#include "model-file.h"

/**
 * NeMo's AudioToMelSpectrogramPreprocessor as it runs in evaluation, without dither (FilterbankFeatures.forward()
 * and normalize_batch() in nemo/collections/asr/parts/preprocessing/features.py): pre-emphasis, a centred STFT
 * with zero padding, the power through the checkpoint's mel filterbank, the log with its additive guard, and, for a
 * model whose preprocessor normalizes "per_feature", the normalization of each mel bin over the utterance.
 *
 * It runs on the host in double precision rather than in a ggml graph: it costs a few milliseconds, and
 * Metal's matrix kernel rounds its inputs to half precision, which the log of the quiet frames' power would
 * magnify. A frame depends on the samples within n_fft / 2 of its centre alone, so frames computed as their samples
 * arrive are the frames of the whole audio when nothing normalizes them.
 */
class NemoFrontend {
public:
    /**
     * The frontend whose settings and tensors are those of a family's file: <prefix>frontend.n_fft, .hop_length, .n_mels,
     * .preemphasis and .log_guard, with .std_guard where `per_feature` normalizes, and the tensors frontend.window and
     * frontend.filterbank.
     */
    NemoFrontend(const ModelFile & m, const std::string & prefix, bool per_feature);
    /** Limits whole-input frame computation to at most four host threads; incremental log_mel() stays sequential. */
    NemoFrontend(const ModelFile & m, const std::string & prefix, bool per_feature, int threads);

    /** The features of mono samples at sample_rate(), [frames, mels] row-major (ggml [mels, frames]). */
    std::vector<float> features(const std::vector<float> & samples) const;

    /** The log-mel power of mono samples at sample_rate() in double precision, [frames, mels] row-major, unnormalized. */
    std::vector<double> log_mels(const std::vector<float> & samples) const;

    /**
     * `mel`, [frames, mels] row-major from log_mels(), normalized per feature as normalize_batch() does: each mel bin less
     * its mean over the frames, over its unbiased standard deviation plus the guard. Takes a frontend that normalizes.
     */
    std::vector<float> normalized(const std::vector<double> & mel) const;

    /** The frames features() gives for that many samples. */
    int64_t frames(size_t samples) const;

    /**
     * The pre-emphasized samples of `n` samples, in float32 as FilterbankFeatures.forward() computes them: each less
     * preemphasis times the sample before it, `previous`, the last sample of the audio before these, for all but the
     * first sample of the audio, which `first` says these begin with.
     */
    void preemphasize(const float * samples, size_t n, bool first, float previous, float * out) const;

    /**
     * The log of the mel power of `count` frames from `frame` on, [count, mels] row-major, in double precision, of the
     * pre-emphasized samples of an audio of `total` samples, of which `x` holds those from `begin` on: every sample of
     * those frames' windows that lies within the audio. The samples outside it are the STFT's zero padding.
     */
    void log_mel(const float * x, int64_t begin, int64_t total, int64_t frame, int64_t count, double * out) const;

    /** The last sample, counted from the start of the audio, that the window of frame `frame` reaches, plus one. */
    int64_t window_end(int64_t frame) const { return frame * hop_ + n_fft_ / 2; }

    int sample_rate() const { return sample_rate_; }
    int hop_length() const { return hop_; }
    int mels() const { return mels_; }

private:
    int sample_rate_, n_fft_, hop_, mels_;
    bool per_feature_;
    int threads_;
    float preemphasis_, log_guard_, std_guard_ = 0;
    /** The window zero-padded to n_fft and centred, as torch.stft() pads it. */
    std::vector<double> window_;
    /** The mel filterbank, [mels, n_fft / 2 + 1] row-major. */
    std::vector<double> filterbank_;
    /** The bins [first, second) where each mel filter is not zero. */
    std::vector<std::pair<int, int>> bands_;
    RealFft fft_;
};
