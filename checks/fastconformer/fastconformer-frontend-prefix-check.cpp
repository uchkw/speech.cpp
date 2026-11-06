// Checks raw-mel prefix reuse against a fresh frontend, including changing tails, transactional publication,
// short-input errors, the 30-second retention limit, and whole-prefix resampling.
//
// usage: fastconformer-frontend-prefix-check <model.gguf> [reference out dir]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <tuple>
#include <utility>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "error.h"
#include "fastconformer/frontend-cache.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "reference-dumps.h"
#include "resample.h"

using fastconformer::FrontendCache;

namespace {

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & name) {
        total++;
        if (!ok) {
            failed++;
            std::printf("FAIL: %s\n", name.c_str());
        }
    }

    template <typename T>
    void equal(const std::vector<T> & got, const std::vector<T> & want, const std::string & name) {
        const bool same = got.size() == want.size() && (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0);
        expect(same && std::all_of(got.begin(), got.end(), [](T x) { return std::isfinite(x); }), name);
    }
};

std::vector<float> audio(size_t n, int pattern = 3) {
    std::vector<float> out(n, 0.0f);
    uint32_t value = 7;
    for (size_t i = 0; i < n; i++) {
        value = value * 1664525u + 1013904223u;
        if (pattern == 1) out[i] = i == 0 ? 1.0f : i + 1 == n ? -0.5f : 0.0f;
        if (pattern == 2) out[i] = i % 2 ? 1e-6f : -1e-6f;
        if (pattern == 3) out[i] = ((float) (value & 65535u) / 32768.0f - 1.0f) * 0.25f;
    }
    return out;
}

FrontendCache::Reading checked(FrontendCache & cache, const NemoFrontend & frontend, const std::vector<float> & input,
                              Checks & checks, const std::string & name) {
    auto reading = cache.prepare(input);
    checks.equal(reading.features, frontend.features(input), name + ": normalized F32");
    if (!reading.raw_mel.empty()) checks.equal(reading.raw_mel, frontend.log_mels(input), name + ": raw F64");
    return reading;
}

template <typename Call>
std::tuple<int, std::string, std::string> rejected(Call call) {
    try {
        call();
    } catch (const Error & e) {
        return {(int) e.fault(), e.what(), e.input()};
    }
    return {-1, "", ""};
}

void cases(const NemoFrontend & frontend, Checks & checks) {
    const size_t rate = (size_t) frontend.sample_rate(), hop = (size_t) frontend.hop_length();
    for (int pattern = 0; pattern < 4; pattern++) {
        FrontendCache cache(frontend);
        const auto whole = audio(rate * 4, pattern);
        const std::vector<float> prefix(whole.begin(), whole.begin() + (ptrdiff_t) (rate * 2));
        auto cold = checked(cache, frontend, prefix, checks, "cold pattern " + std::to_string(pattern));
        checks.expect(cold.reused_frames == 0, "cold request reuses no frames");
        cache.publish(std::move(cold));
        auto growing = checked(cache, frontend, whole, checks, "growing pattern " + std::to_string(pattern));
        checks.expect(growing.reused_frames > 0, "an identical PCM prefix actually reuses frames");
        checks.expect(frontend.window_end(growing.reused_frames - 1) <= (int64_t) prefix.size(), "no reused frame reads future samples");
        cache.publish(std::move(growing));
        auto shorter = checked(cache, frontend, prefix, checks, "shortened pattern " + std::to_string(pattern));
        checks.expect(shorter.reused_frames == 0, "a shortened input resets reuse");
    }

    for (int delta : {-1, 0, 1}) {
        const size_t count = (size_t) (frontend.window_end(4) + delta);
        const auto longer = audio(count + 3 * hop);
        const std::vector<float> before(longer.begin(), longer.begin() + (ptrdiff_t) count);
        FrontendCache cache(frontend);
        cache.publish(checked(cache, frontend, before, checks, "window boundary seed"));
        auto reading = checked(cache, frontend, longer, checks, "window boundary " + std::to_string(delta));
        int64_t settled = 0;
        while (settled < frontend.frames(count) && frontend.window_end(settled) <= (int64_t) count) settled++;
        checks.expect(reading.reused_frames == settled, "exact window endpoint, including the exclusive sample boundary");
    }

    FrontendCache cache(frontend);
    const auto old = audio(rate * 2);
    cache.publish(checked(cache, frontend, old, checks, "mutation seed"));
    auto changed = old;
    changed[0] += .25f;
    checks.expect(checked(cache, frontend, changed, checks, "different utterance").reused_frames == 0,
                  "a changed first sample invalidates every frame");
    changed = old;
    changed[old.size() / 2] += .25f;
    auto middle = checked(cache, frontend, changed, checks, "same-length middle edit");
    checks.expect(middle.reused_frames > 0 && frontend.window_end(middle.reused_frames - 1) <= (int64_t) (old.size() / 2),
                  "same-length edits reuse only the verified prefix");
    changed = old;
    changed.back() += .5f;
    checked(cache, frontend, changed, checks, "last sample changes preemphasis and the padded tail");
    changed = old;
    changed.insert(changed.end(), hop + 1, -.75f);
    checked(cache, frontend, changed, checks, "append keeps the actual preemphasis predecessor");
    for (float & x : changed) x *= .5f;
    checks.expect(checked(cache, frontend, changed, checks, "changed enhancement output").reused_frames == 0,
                  "enhancement identity is never inferred from source audio");

    const auto zeros = audio(rate * 2, 0);
    cache.publish(checked(cache, frontend, zeros, checks, "zero seed"));
    auto signed_zeros = zeros;
    signed_zeros[0] = -0.0f;
    checks.expect(checked(cache, frontend, signed_zeros, checks, "signed zero").reused_frames == 0,
                  "PCM comparison uses bits, not floating-point equality");
    const auto matching = audio(rate * 4, 0);
    checks.expect(checked(cache, frontend, matching, checks, "discarded reading").reused_frames > 0,
                  "discarding a candidate does not publish its different input");
    for (size_t size : {size_t(0), hop - 1, 2 * hop - 1}) {
        const auto short_audio = audio(size);
        const auto want = rejected([&] { frontend.features(short_audio); });
        checks.expect(std::get<0>(want) == (int) Fault::OutOfRange && std::get<2>(want) == "audio", "short-input error is attributed to audio");
        checks.expect(rejected([&] { cache.prepare(short_audio); }) == want, "cache keeps the original short-input error");
    }
    checks.expect(checked(cache, frontend, matching, checks, "error leaves completed entry intact").reused_frames > 0,
                  "a failed reading does not replace the last completed entry");

    const auto limit = audio(rate * 30, 2);
    cache.publish(checked(cache, frontend, limit, checks, "30-second retention limit"));
    auto beyond = limit;
    beyond.push_back(.25f);
    auto uncached = checked(cache, frontend, beyond, checks, "valid audio beyond retention limit");
    checks.expect(uncached.samples.empty() && uncached.raw_mel.empty() && uncached.reused_frames == 0,
                  "long valid audio is processed whole without retaining another entry");
    checks.expect(checked(cache, frontend, limit, checks, "unpublished long input").reused_frames > 0,
                  "a cancelled long reading leaves the earlier completed entry intact");
    cache.publish(std::move(uncached));
    checks.expect(checked(cache, frontend, limit, checks, "successful long input clears old entry").reused_frames == 0,
                  "a successful uncached reading clears the old entry");

    for (int from : {8000, 44100, 48000}) {
        const auto input = audio((size_t) from * 4);
        const std::vector<float> first(input.begin(), input.begin() + (ptrdiff_t) ((size_t) from * 2));
        const Resampler resampler(from, (int) rate);
        FrontendCache resampled(frontend);
        resampled.publish(checked(resampled, frontend, resampler(first), checks, "resampled prefix seed"));
        const auto reading = checked(resampled, frontend, resampler(input), checks, "resampled tail changes " + std::to_string(from));
        checks.expect(reading.reused_frames > 0, "the unchanged resampled interior can be reused despite its changing right tail");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 2 || args.size() > 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> [reference out dir]\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) throw std::runtime_error("cannot start the CPU backend for frontend weights");
        const ModelFile model(args[1], backend.get(), fastconformer::layout, [](const std::string & name) { return name.rfind("frontend.", 0) == 0; });
        const NemoFrontend frontend(model, "fastconformer.", true);
        Checks checks;
        cases(frontend, checks);
        if (args.size() == 3) {
            for (const auto & dump : reference_dumps(args[2], model)) {
                const auto input = read_npy((dump / "audio.npy").u8string());
                const auto want = read_npy((dump / "features.npy").u8string());
                FrontendCache cache(frontend);
                for (size_t count = (size_t) frontend.sample_rate() * 2; count < input.f32.size(); count += (size_t) frontend.sample_rate() * 2) {
                    const std::vector<float> prefix(input.f32.begin(), input.f32.begin() + (ptrdiff_t) count);
                    cache.publish(checked(cache, frontend, prefix, checks, "official audio prefix"));
                }
                const auto reading = checked(cache, frontend, input.f32, checks, dump.filename().u8string());
                const Diff diff = compare(reading.features, want.f32);
                print_diff(dump.filename().u8string() + " official frontend", diff);
                checks.expect(want.shape.size() == 2 && want.shape[1] == frontend.mels() && reading.features.size() == want.f32.size() && diff.snr_db > 90,
                              "official frontend keeps its 90 dB bound");
            }
        } else {
            std::printf("Official frontend reference: not checked (no dump root supplied)\n");
        }
        std::printf("frontend prefix cache: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
