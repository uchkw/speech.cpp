// Compares the explicit Haas vDSP frontend with the ordinary double FFT, checks the selector and request-local
// scratch, and optionally checks both paths against official frontend dumps without changing the 90 dB bound.
// Usage: fastconformer-frontend-vdsp-check <model.gguf> [reference out dir]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "error.h"
#include "fastconformer/frontend.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "reference-dumps.h"

namespace {

void setting(const char * value) {
#ifdef _WIN32
    const int status = _putenv_s("SPEECH_HAAS_FRONTEND_VDSP", value ? value : "");
#else
    const int status = value ? setenv("SPEECH_HAAS_FRONTEND_VDSP", value, 1) : unsetenv("SPEECH_HAAS_FRONTEND_VDSP");
#endif
    if (status != 0) throw std::runtime_error("cannot set the vDSP frontend selector");
}

struct Checks {
    size_t total = 0, failed = 0;
    void expect(bool condition, const std::string & what) {
        ++total;
        if (!condition) { ++failed; std::printf("FAIL: %s\n", what.c_str()); }
    }
    template <typename T>
    void values(const std::vector<T> & got, const std::vector<T> & want, const std::string & what, bool exact) {
        const bool finite = std::all_of(got.begin(), got.end(), [](T value) { return std::isfinite(value); }) &&
                            std::all_of(want.begin(), want.end(), [](T value) { return std::isfinite(value); });
        const Diff diff = compare(got, want);
        const bool same = got.size() == want.size() && (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0);
        expect(finite && got.size() == want.size() && (exact ? same : diff.snr_db > 90), what);
        print_diff(what + (exact ? " (exact)" : " (>90 dB)"), diff);
    }
};

void pair(const NemoFrontend & ordinary, const NemoFrontend & selected, const std::vector<float> & audio,
          int n_fft, const std::string & name, bool exact, Checks & checks) {
    const auto a = ordinary.log_mels(audio), b = selected.log_mels(audio);
    checks.values(b, a, name + " log mel", exact);
    const auto normalized = selected.normalized(b);
    checks.values(normalized, ordinary.normalized(a), name + " normalized", exact);
    checks.values(selected.features(audio), normalized, name + " repeated request", true);

    std::vector<float> emphasized(audio.size());
    selected.preemphasize(audio.data(), audio.size(), true, 0.0f, emphasized.data());
    std::vector<double> split(b.size(), std::numeric_limits<double>::quiet_NaN());
    const int64_t frames = selected.frames(audio.size()), first = frames / 2;
    selected.log_mel(emphasized.data(), 0, (int64_t) audio.size(), 0, first, split.data());
    const int64_t begin = std::max<int64_t>(0, first * selected.hop_length() - n_fft / 2);
    selected.log_mel(emphasized.data() + begin, begin, (int64_t) audio.size(), first, frames - first,
                     split.data() + first * selected.mels());
    checks.values(split, b, name + " split frames", true);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        if (args.size() < 2 || args.size() > 3) throw std::runtime_error("usage: fastconformer-frontend-vdsp-check <model.gguf> [reference out dir]");
        start_ggml();
        setting("1");
        const ModelFile metadata(args[1], fastconformer::layout);
        const bool eligible = metadata.str("general.name") == "Haas-ja-v1.0" &&
                              metadata.str("fastconformer.decoder.prediction") == "kda";
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) throw std::runtime_error("cannot start the CPU backend for frontend weights");
        const ModelFile model(args[1], backend.get(), fastconformer::layout, [](const std::string & name) {
            return name == "frontend.window" || name == "frontend.filterbank";
        });
        Checks checks;
        setting(nullptr);
        const auto unset = fastconformer::make_frontend(model);
        setting("0");
        const auto ordinary = fastconformer::make_frontend(model);
        setting("invalid");
        bool refused = false;
        try { const auto invalid = fastconformer::make_frontend(model); }
        catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
        checks.expect(refused == eligible, "only Haas KDA reads and validates the selector");
        setting("1");
        const NemoFrontend unchanged(model, "fastconformer.", true);
#ifndef __APPLE__
        if (eligible) {
            bool unavailable = false;
            try { const auto selected = fastconformer::make_frontend(model); }
            catch (const Error & error) { unavailable = error.fault() == Fault::InvalidArgument; }
            checks.expect(unavailable, "a non-Apple Haas request explicitly refuses vDSP");
            std::printf("vDSP unavailable: %zu checks, %zu failed; numerical cases are not run\n", checks.total, checks.failed);
            return checks.failed ? 1 : 0;
        }
#endif
        const auto selected = fastconformer::make_frontend(model);
        const int hop = ordinary.hop_length(), n_fft = (int) model.u32("fastconformer.frontend.n_fft");
        const std::vector<float> first((size_t) (2 * hop), 0.125f);
        const auto first_output = selected.features(first);
        checks.values(ordinary.features(first), unset.features(first), "unset equals zero", true);
        checks.values(unchanged.features(first), ordinary.features(first), "shared three-argument constructor ignores the selector", true);
        for (int pattern = 0; pattern < 5; ++pattern) {
            for (int64_t samples : {int64_t(2) * hop, int64_t(3) * hop - 1, int64_t(17) * hop + 3, int64_t(129) * hop + 17}) {
                std::vector<float> audio((size_t) samples);
                for (size_t i = 0; i < audio.size(); ++i) {
                    if (pattern == 0) audio[i] = 0.0f;
                    if (pattern == 1) audio[i] = 0.125f;
                    if (pattern == 2) audio[i] = i % 2 ? -0.125f : 0.125f;
                    if (pattern == 3) audio[i] = (float) (0.2 * std::sin(i * 0.037) + 0.1 * std::cos(i * 0.217));
                    if (pattern == 4) audio[i] = (float) (1e-20 * std::sin(i * 0.031));
                }
                pair(ordinary, selected, audio, n_fft, "pattern " + std::to_string(pattern) + " samples " + std::to_string(samples), !eligible, checks);
            }
        }
        checks.values(selected.features(first), first_output, "long-to-short request", true);
        for (const NemoFrontend * frontend : {&ordinary, &selected}) {
            bool rejected = false;
            try { frontend->features(std::vector<float>((size_t) (2 * hop - 1))); }
            catch (const Error & error) { rejected = error.fault() == Fault::OutOfRange && error.input() == "audio"; }
            checks.expect(rejected, "per-feature normalization still requires two frames");
        }
        if (args.size() == 3) {
            for (const auto & dump : reference_dumps(args[2], model)) {
                const Npy audio = read_npy((dump / "audio.npy").u8string());
                const Npy want = read_npy((dump / "features.npy").u8string());
                checks.expect(want.shape.size() == 2 && want.shape[0] == ordinary.frames(audio.f32.size()) &&
                              want.shape[1] == ordinary.mels(), "official feature shape");
                const std::string name = dump.filename().u8string();
                checks.values(ordinary.features(audio.f32), want.f32, name + " official ordinary", false);
                checks.values(selected.features(audio.f32), want.f32, name + " official selected", false);
                pair(ordinary, selected, audio.f32, n_fft, name, !eligible, checks);
            }
        }
        std::printf("frontend vDSP eligible=%s: %zu checks, %zu failed; official dumps %s; encoder, decoding and speed are not checked\n",
                    eligible ? "yes" : "no", checks.total, checks.failed, args.size() == 3 ? "checked" : "not checked");
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
