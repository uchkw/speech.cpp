// Compares sequential and parallel NeMo frontend frames bit for bit, in float64 before normalization and float32
// after it. Covers short-input rejection, hop/grain boundaries, shared immutable FFT tables and incremental frames
// after discarded samples. Optional official dumps retain the existing 90 dB frontend bound.
//
// usage: nemo-frontend-parallel-check <fastconformer|sortformer> <model.gguf> [reference out dir]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <future>
#include <memory>
#include <set>
#include <tuple>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "error.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "nemo-frontend.h"
#include "npy.h"
#include "reference-dumps.h"
#include "sortformer/layout.h"

namespace {

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & what) {
        total++;
        if (!ok) {
            failed++;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    template <typename T>
    void equal(const std::vector<T> & got, const std::vector<T> & want, const std::string & what) {
        const bool finite = std::all_of(got.begin(), got.end(), [](T x) { return std::isfinite(x); });
        const bool same = got.size() == want.size() && (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0);
        expect(finite && same, what);
    }
};

std::vector<float> samples(size_t size, int pattern) {
    std::vector<float> out(size, 0.0f);
    if (size == 0 || pattern == 0) return out;
    if (pattern == 1) {
        out.front() = 1.0f;
        out.back() = -0.5f;
    } else if (pattern == 2) {
        for (size_t i = 0; i < size; i++) out[i] = i % 2 ? 1e-6f : -1e-6f;
    } else {
        uint32_t value = 7;
        for (float & x : out) {
            value = value * 1664525u + 1013904223u;
            x = ((float) (value & 65535u) / 32768.0f - 1.0f) * 0.25f;
        }
    }
    return out;
}

std::tuple<int, std::string, std::string> rejected(const NemoFrontend & frontend, const std::vector<float> & audio) {
    try {
        frontend.log_mels(audio);
    } catch (const Error & e) {
        return {(int) e.fault(), e.what(), e.input()};
    }
    return {-1, "", ""};
}

/** Keeps only the samples the next frame needs, as Sortformer's stream does with a nonzero sample origin. */
std::vector<double> in_pieces(const NemoFrontend & frontend, const std::vector<float> & audio, size_t largest) {
    std::vector<float> x;
    std::vector<double> mel;
    int64_t begin = 0, received = 0, computed = 0;
    uint32_t random = 7;
    const int64_t reach = frontend.window_end(0);
    const auto append = [&](int64_t target) {
        if (target > computed) {
            mel.resize((size_t) (target * frontend.mels()));
            frontend.log_mel(x.data(), begin, received, computed, target - computed, mel.data() + computed * frontend.mels());
            computed = target;
        }
        const int64_t keep = std::max<int64_t>(computed * frontend.hop_length() - reach, 0);
        if (keep > begin) {
            x.erase(x.begin(), x.begin() + (keep - begin));
            begin = keep;
        }
    };
    for (size_t at = 0; at < audio.size();) {
        random = random * 1664525u + 1013904223u;
        const size_t count = std::min<size_t>(1 + random % largest, audio.size() - at);
        const size_t previous = x.size();
        x.resize(previous + count);
        frontend.preemphasize(audio.data() + at, count, at == 0, at == 0 ? 0.0f : audio[at - 1], x.data() + previous);
        at += count;
        received += (int64_t) count;
        append(received >= reach ? (received - reach) / frontend.hop_length() + 1 : 0);
    }
    append(frontend.frames(audio.size()));
    return mel;
}

void compare_input(const NemoFrontend & serial, const NemoFrontend & two, const NemoFrontend & four,
                   const std::vector<float> & audio, const std::string & label, Checks & checks) {
    const auto mel = serial.log_mels(audio);
    const auto features = serial.features(audio);
    for (const NemoFrontend * candidate : {&two, &four}) {
        checks.equal(candidate->log_mels(audio), mel, label + ": raw float64 mel");
        checks.equal(candidate->features(audio), features, label + ": final float32 features");
    }
}

void synthetic_cases(const NemoFrontend & serial, const NemoFrontend & two, const NemoFrontend & four,
                     bool normalized, Checks & checks) {
    const size_t hop = (size_t) serial.hop_length();
    const int minimum = normalized ? 2 : 1;
    for (size_t count : {size_t(0), hop - 1, minimum * hop - 1}) {
        const auto audio = samples(count, 3);
        const auto expected = rejected(serial, audio);
        checks.expect(std::get<0>(expected) == (int) Fault::OutOfRange && std::get<2>(expected) == "audio", "short input keeps its error attribution");
        checks.expect(rejected(two, audio) == expected && rejected(four, audio) == expected, "short input keeps the serial error");
    }
    std::set<size_t> sizes{minimum * hop, (size_t) serial.sample_rate() * 2, (size_t) serial.sample_rate() * 4,
                           (size_t) serial.sample_rate() * 30};
    for (size_t frames : {size_t(127), size_t(128), size_t(255), size_t(256), size_t(257), size_t(383), size_t(384),
                          size_t(385), size_t(511), size_t(512), size_t(513)}) {
        for (int delta : {-1, 0, 1}) sizes.insert((size_t) ((int64_t) (frames * hop) + delta));
    }
    for (size_t size : sizes) {
        for (int pattern = 0; pattern < 4; pattern++) {
            compare_input(serial, two, four, samples(size, pattern), std::to_string(size) + " samples pattern " + std::to_string(pattern), checks);
        }
    }
    const auto a = samples((size_t) serial.sample_rate() * 4 + 1, 3), b = samples((size_t) serial.sample_rate() * 2 - 1, 1);
    auto pending = std::async(std::launch::async, [&] { return four.log_mels(a); });
    const auto simultaneous = four.log_mels(b);
    checks.equal(pending.get(), serial.log_mels(a), "concurrent call shares only immutable frontend data");
    checks.equal(simultaneous, serial.log_mels(b), "caller and tasks keep independent scratch");
    for (size_t largest : {size_t(1), size_t(333), size_t(4800)}) {
        const auto pieces = in_pieces(four, a, largest);
        checks.equal(pieces, serial.log_mels(a), "incremental frames after discarding old samples");
        if (normalized) checks.equal(four.normalized(pieces), serial.features(a), "concatenated frames normalize in the same order");
        else checks.equal(std::vector<float>(pieces.begin(), pieces.end()), serial.features(a), "Sortformer pieces keep whole-input float32 values");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 3 || args.size() > 4 || (args[1] != "fastconformer" && args[1] != "sortformer")) {
        std::fprintf(stderr, "usage: %s <fastconformer|sortformer> <model.gguf> [reference out dir]\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(ggml_backend_cpu_init(), ggml_backend_free);
        if (!backend) throw std::runtime_error("cannot start the CPU backend for frontend weights");
        const bool normalized = args[1] == "fastconformer";
        const Layout & layout = normalized ? fastconformer::layout : sortformer::layout;
        const ModelFile model(args[2], backend.get(), layout, [](const std::string & name) { return name.rfind("frontend.", 0) == 0; });
        const std::string prefix = args[1] + ".";
        const NemoFrontend serial(model, prefix, normalized), two(model, prefix, normalized, 2), four(model, prefix, normalized, 4);
        Checks checks;
        synthetic_cases(serial, two, four, normalized, checks);
        if (args.size() == 4) {
            for (const auto & dump : reference_dumps(args[3], model)) {
                const auto audio = read_npy((dump / "audio.npy").u8string());
                const auto want = read_npy((dump / "features.npy").u8string());
                compare_input(serial, two, four, audio.f32, dump.filename().u8string(), checks);
                const auto got = four.features(audio.f32);
                const Diff diff = compare(got, want.f32);
                print_diff(dump.filename().u8string() + " official frontend", diff);
                checks.expect(want.shape.size() == 2 && want.shape[1] == four.mels() && got.size() == want.f32.size() && diff.snr_db > 90,
                              "official frontend keeps its existing 90 dB bound");
            }
        } else {
            std::printf("Official frontend reference: not checked (no dump root supplied)\n");
        }
        std::printf("frontend frame parallelism: %zu checks, %zu failed\n", checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
