#include "nemo-frontend.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <future>
#include <string>
#include <system_error>

#include "error.h"

namespace {

/** A tensor that the layout stores in F32, as doubles. */
std::vector<double> read_tensor(const ModelFile & m, const std::string & name) {
    ggml_tensor * t = m.tensor(name);
    std::vector<float> f(ggml_nelements(t));
    ggml_backend_tensor_get(t, f.data(), 0, ggml_nbytes(t));
    return std::vector<double>(f.begin(), f.end());
}

}  // namespace

NemoFrontend::NemoFrontend(const ModelFile & m, const std::string & prefix, bool per_feature)
    : NemoFrontend(m, prefix, per_feature, 1) {}

NemoFrontend::NemoFrontend(const ModelFile & m, const std::string & prefix, bool per_feature, int threads)
    : sample_rate_((int) m.u32("speech.sample_rate")),
      n_fft_((int) m.u32(prefix + "frontend.n_fft")),
      hop_((int) m.u32(prefix + "frontend.hop_length")),
      mels_((int) m.u32(prefix + "frontend.n_mels")),
      per_feature_(per_feature),
      threads_(std::min(threads, 4)),
      preemphasis_(m.f32(prefix + "frontend.preemphasis")),
      log_guard_(m.f32(prefix + "frontend.log_guard")),
      fft_(n_fft_) {
    if (threads < 1) throw std::invalid_argument("the frontend thread limit must be positive");
    if (per_feature) std_guard_ = m.f32(prefix + "frontend.std_guard");
    const std::vector<double> w = read_tensor(m, "frontend.window");
    const int64_t window = (int64_t) w.size();
    window_.assign(n_fft_, 0.0);
    // torch.stft() centres a window shorter than n_fft in it.
    const int64_t left = (n_fft_ - window) / 2;
    for (int64_t i = 0; i < window; i++) window_[left + i] = w[i];
    filterbank_ = read_tensor(m, "frontend.filterbank");
    const int bins = n_fft_ / 2 + 1;
    for (int b = 0; b < mels_; b++) {
        int first = 0, last = bins;
        while (first < bins && filterbank_[(size_t) b * bins + first] == 0) first++;
        while (last > first && filterbank_[(size_t) b * bins + last - 1] == 0) last--;
        bands_.push_back({first, last});
    }
}

int64_t NemoFrontend::frames(size_t samples) const {
    // FilterbankFeatures.get_seq_len(): the STFT gives one frame more, which the official code masks out.
    return (int64_t) samples / hop_;
}

void NemoFrontend::preemphasize(const float * samples, size_t n, bool first, float previous, float * out) const {
    for (size_t i = 0; i < n; i++) {
        const float before = i > 0 ? samples[i - 1] : previous;
        out[i] = i == 0 && first ? samples[0] : samples[i] - preemphasis_ * before;
    }
}

void NemoFrontend::log_mel(const float * x, int64_t begin, int64_t total, int64_t frame, int64_t count, double * out) const {
    const int bins = n_fft_ / 2 + 1;
    std::vector<double> windowed(n_fft_);
    std::vector<std::complex<double>> spectrum(bins);
    std::vector<double> power(bins);
    for (int64_t f = 0; f < count; f++) {
        // center=True with pad_mode="constant": the signal has n_fft / 2 zeros before it.
        const int64_t start = (frame + f) * hop_ - n_fft_ / 2;
        for (int k = 0; k < n_fft_; k++) {
            const int64_t i = start + k;
            windowed[k] = i >= 0 && i < total ? x[i - begin] * window_[k] : 0.0;
        }
        fft_.transform(windowed.data(), spectrum.data());
        for (int k = 0; k < bins; k++) power[k] = std::norm(spectrum[k]);
        for (int m = 0; m < mels_; m++) {
            double sum = 0;
            for (int k = bands_[m].first; k < bands_[m].second; k++) sum += filterbank_[(size_t) m * bins + k] * power[k];
            out[(size_t) (f * mels_ + m)] = std::log(sum + (double) log_guard_);
        }
    }
}

std::vector<float> NemoFrontend::features(const std::vector<float> & samples) const {
    const std::vector<double> mel = log_mels(samples);
    if (per_feature_) return normalized(mel);
    std::vector<float> out(mel.size());
    for (size_t i = 0; i < mel.size(); i++) out[i] = (float) mel[i];
    return out;
}

std::vector<double> NemoFrontend::log_mels(const std::vector<float> & samples) const {
    const int64_t frames = this->frames(samples.size());
    // normalize_batch() divides by frames - 1.
    if (frames < (per_feature_ ? 2 : 1)) {
        throw Error(Fault::OutOfRange, "the audio has " + std::to_string(samples.size()) + " samples at " + std::to_string(sample_rate_) +
                                           " Hz; the model needs at least " + std::to_string((per_feature_ ? 2 : 1) * hop_),
                    "audio");
    }
    std::vector<float> x(samples.size());
    preemphasize(samples.data(), samples.size(), true, 0.0f, x.data());
    std::vector<double> mel((size_t) (frames * mels_));
    const int workers = (int) std::max<int64_t>(1, std::min<int64_t>(threads_, frames / 128));
    if (workers == 1) {
        log_mel(x.data(), 0, (int64_t) x.size(), 0, frames, mel.data());
    } else {
        const auto compute = [&](int worker) {
            const int64_t first = frames * worker / workers, last = frames * (worker + 1) / workers;
            log_mel(x.data(), 0, (int64_t) x.size(), first, last - first, mel.data() + first * mels_);
        };
        // Async futures wait for their tasks on destruction, so a launch or worker failure cannot outlive x or mel.
        std::vector<std::future<void>> pending;
        pending.reserve((size_t) (workers - 1));
        try {
            for (int worker = 1; worker < workers; worker++) pending.push_back(std::async(std::launch::async, compute, worker));
        } catch (const std::system_error & e) {
            throw Error(Fault::OutOfMemory, std::string("the host cannot start another frontend thread (") + e.what() +
                                               "); close other programs and run it again");
        }
        compute(0);
        for (auto & task : pending) task.get();
    }
    return mel;
}

std::vector<float> NemoFrontend::normalized(const std::vector<double> & mel) const {
    const int64_t frames = (int64_t) mel.size() / mels_;
    std::vector<float> out(mel.size());
    // normalize_batch(normalize_type="per_feature"): the unbiased standard deviation, plus a guard.
    for (int m = 0; m < mels_; m++) {
        double mean = 0;
        for (int64_t f = 0; f < frames; f++) mean += mel[(size_t) (f * mels_ + m)];
        mean /= (double) frames;
        double var = 0;
        for (int64_t f = 0; f < frames; f++) {
            const double d = mel[(size_t) (f * mels_ + m)] - mean;
            var += d * d;
        }
        const double std = std::sqrt(var / (double) (frames - 1)) + (double) std_guard_;
        for (int64_t f = 0; f < frames; f++) {
            out[(size_t) (f * mels_ + m)] = (float) ((mel[(size_t) (f * mels_ + m)] - mean) / std);
        }
    }
    return out;
}
