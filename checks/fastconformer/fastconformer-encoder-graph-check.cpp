// Compares actual Recognizer::encode calls with and without its one-slot exact-frame graph cache.
// Checks final joint-input bytes, input updates, eviction, failed updates, decoding histories and real audio fixtures.
// This is a differential check, not an official reference check or a representative latency measurement.
// Usage: fastconformer-encoder-graph-check <model.gguf> <cpu|gpu|device> <threads> <audio.npy> [audio.npy ...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "fastconformer/recognizer.h"
#include "npy.h"

using namespace fastconformer;

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

void setting(const char * name, const char * value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, std::string("cannot set ") + name);
#else
    require(setenv(name, value, 1) == 0, std::string("cannot set ") + name);
#endif
}

bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
}

bool identical(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & name) {
        ++total;
        if (!ok) {
            ++failed;
            std::printf("FAIL: %s\n", name.c_str());
        }
    }

    void values(const std::vector<float> & a, const std::vector<float> & b, const std::string & name) {
        const bool exact = finite(a) && finite(b) && identical(a, b);
        expect(exact, name);
        const Diff diff = compare(a, b);
        std::printf("%s exact=%s values=%zu max_abs=%.9g snr_db=%.9g\n", name.c_str(), exact ? "true" : "false",
                    a.size(), diff.max_abs, diff.snr_db);
    }
};

bool same_path(const Decoding & a, const Decoding & b) {
    return a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs;
}

bool same_cache(const EncoderGraphCacheInfo & a, const EncoderGraphCacheInfo & b) {
    return a.enabled == b.enabled && a.builds == b.builds && a.hits == b.hits && a.frames == b.frames &&
        a.feature_bytes == b.feature_bytes && a.compute_bytes == b.compute_bytes;
}

using Outputs = std::pair<std::vector<float>, std::vector<float>>;

/** Two loaded recognizers keep independent model, position-cache, graph and decoder owners on the same backend. */
struct Pair {
    std::unique_ptr<Recognizer> ordinary, cached;
    Checks & checks;
    size_t last_callbacks = 0;
    bool after_step_fixture_used = false;

    Pair(const std::string & path, ggml_backend_t backend, int threads, Checks & checks) : checks(checks) {
        setting("SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE", "0");
        setting("SPEECH_HAAS_ENCODER_GRAPH_CACHE", "0");
        ordinary = std::make_unique<Recognizer>(path, backend, threads);
        setting("SPEECH_HAAS_ENCODER_GRAPH_CACHE", "1");
        cached = std::make_unique<Recognizer>(path, backend, threads);
        setting("SPEECH_HAAS_ENCODER_GRAPH_CACHE", "0");
        const auto a = ordinary->encoder_graph_cache_info(), b = cached->encoder_graph_cache_info();
        require(!a.enabled && b.enabled, "this check needs Haas-ja-v1.0 KDA weights and the two selected cache modes");
        require(a.frames == -1 && b.frames == -1 && a.compute_bytes == 0 && b.compute_bytes == 0 &&
                    a.builds == 0 && b.builds == 0 && a.hits == 0 && b.hits == 0,
                "constructing a recognizer eagerly created an encoder graph");
    }

    void counters(const EncoderGraphCacheInfo & before_a, const EncoderGraphCacheInfo & before_b,
                  int64_t frames, const std::string & name) {
        const auto a = ordinary->encoder_graph_cache_info(), b = cached->encoder_graph_cache_info();
        const bool hit = before_b.frames == frames;
        checks.expect(!a.enabled && a.frames == -1 && a.feature_bytes == 0 && a.compute_bytes > 0 &&
                          a.builds == before_a.builds + 1 && a.hits == 0,
                      name + " ordinary graph is not retained");
        checks.expect(b.enabled && b.frames == frames && b.builds == before_b.builds + !hit && b.hits == before_b.hits + hit &&
                          b.feature_bytes == (size_t) frames * cached->frontend().mels() * sizeof(float) && b.compute_bytes > 0,
                      name + " one-slot exact feature-frame key");
        if (hit) checks.expect(b.compute_bytes == before_b.compute_bytes, name + " same-shape allocation remains fixed");
        std::printf("%s frames=%lld hit=%s builds=%zu hits=%zu feature_bytes=%zu compute_bytes=%zu\n", name.c_str(),
                    (long long) frames, hit ? "true" : "false", b.builds, b.hits, b.feature_bytes, b.compute_bytes);
    }

    Outputs encode(const std::vector<float> & features, int64_t frames, const std::string & name) {
        const auto a = ordinary->encoder_graph_cache_info(), b = cached->encoder_graph_cache_info();
        Outputs output{ordinary->encode(features, frames), cached->encode(features, frames)};
        counters(a, b, frames, name);
        checks.values(output.second, output.first, name + " final joint input");
        return output;
    }

    Decoding decode(const Outputs & output, size_t allowed, const std::string & name,
                    DecodeOn on = DecodeOn::Cpu, const Boost * boost = nullptr) {
        const auto before_a = ordinary->encoder_graph_cache_info(), before_b = cached->encoder_graph_cache_info();
        std::vector<double> a_progress, b_progress;
        bool a_stopped = false, b_stopped = false;
        const auto progress_a = [&](double done) {
            a_progress.push_back(done);
            a_stopped = a_progress.size() > allowed;
            return !a_stopped;
        };
        const auto progress_b = [&](double done) {
            b_progress.push_back(done);
            b_stopped = b_progress.size() > allowed;
            return !b_stopped;
        };
        const std::string decoding = ordinary->decodings().front();
        const auto a = boost ? ordinary->decoding(output.first, decoding, on, progress_a, *boost) :
                               ordinary->decoding(output.first, decoding, on, progress_a);
        const auto b = boost ? cached->decoding(output.second, decoding, on, progress_b, *boost) :
                               cached->decoding(output.second, decoding, on, progress_b);
        checks.expect(same_path(a, b) && ordinary->detokenizer().text(a.ids) == cached->detokenizer().text(b.ids),
                      name + " TDT IDs, frames, durations, graphs and text");
        checks.expect(a_progress == b_progress, name + " progress and cancellation sequence");
        if (allowed != std::numeric_limits<size_t>::max()) {
            checks.expect(a_stopped && b_stopped, name + " both callbacks actually request cancellation");
            checks.expect(a_progress.size() == allowed + 1 && b_progress.size() == allowed + 1,
                          name + " cancellation occurs at exactly allowed + 1 callbacks");
            if (allowed == 0) {
                checks.expect(a.graphs == 0 && b.graphs == 0 && a.ids.empty() && b.ids.empty(),
                              name + " before-first cancellation computes no decoder step");
            } else {
                checks.expect(a.graphs >= allowed && b.graphs >= allowed,
                              name + " after-step cancellation follows completed decoder work");
            }
        }
        checks.expect(same_cache(before_a, ordinary->encoder_graph_cache_info()) &&
                          same_cache(before_b, cached->encoder_graph_cache_info()),
                      name + " decoding and boosting do not alter the encoder cache");
        last_callbacks = std::min(a_progress.size(), b_progress.size());
        std::printf("%s tokens=%zu graphs=%zu ordinary_callbacks=%zu cached_callbacks=%zu ordinary_stop_requested=%s cached_stop_requested=%s\n",
                    name.c_str(), b.ids.size(), b.graphs, a_progress.size(), b_progress.size(),
                    a_stopped ? "true" : "false", b_stopped ? "true" : "false");
        return b;
    }
};

std::vector<float> features(int mels, int64_t frames, int seed) {
    std::vector<float> values((size_t) mels * frames);
    for (size_t i = 0; i < values.size(); ++i) values[i] = 0.2f * std::sin((float) ((i + seed * 131) % 997) * 0.031f);
    return values;
}

void shapes(Pair & pair) {
    const int mels = pair.ordinary->frontend().mels();
    require(pair.ordinary->encoder().subsampled_frames(129) == pair.ordinary->encoder().subsampled_frames(130),
            "the shape test needs distinct feature-frame counts that share the same subsampled count");
    Outputs first, previous;
    int seed = 1;
    for (int64_t frames : {129, 129, 130, 1361, 1361, 1369, 1369, 3000, 129}) {
        auto input = features(mels, frames, seed);
        const auto result = pair.encode(input, frames, "synthetic shape " + std::to_string(seed));
        if (seed == 1) first = result;
        if (seed == 2) pair.checks.expect(!identical(previous.first, result.first), "new features affect a same-shape request");
        std::fill(input.begin(), input.end(), std::numeric_limits<float>::quiet_NaN());
        previous = result;
        ++seed;
    }
    const auto restored = pair.encode(features(mels, 129, 1), 129, "return to first feature values");
    pair.checks.values(restored.first, first.first, "ordinary long-to-short result");
    pair.checks.values(restored.second, first.second, "cached long-to-short result");
    for (const int64_t frames : {129, 130}) {
        const auto before = pair.cached->encoder_graph_cache_info();
        auto malformed = features(mels, frames, 1);
        malformed.pop_back();
        bool refused = false;
        try { pair.cached->encode(malformed, frames); }
        catch (const std::runtime_error &) { refused = true; }
        const auto after = pair.cached->encoder_graph_cache_info();
        pair.checks.expect(refused && after.frames == -1 && after.feature_bytes == 0 && after.compute_bytes == before.compute_bytes &&
                               after.builds == before.builds && after.hits == before.hits,
                           "a failed update or rebuild clears the slot without counting a completed request");
        pair.encode(features(mels, 129, 1), 129, "retry after an invalid feature size");
    }
}

void cancel_and_retry(Pair & pair, const std::vector<float> & input, int64_t frames, const Outputs & output,
                      const Decoding & plain, size_t allowed) {
    const std::string name = "cancellation allowed=" + std::to_string(allowed);
    pair.decode(output, allowed, name);
    const auto again = pair.encode(input, frames, "same shape after " + name);
    pair.checks.values(again.second, output.second, "encoder bytes after " + name);
    pair.checks.expect(same_path(pair.decode(again, std::numeric_limits<size_t>::max(), "request after " + name), plain),
                       "decoder state starts fresh after " + name);
}

void request_states(Pair & pair) {
    const size_t unlimited = std::numeric_limits<size_t>::max();
    const auto input = features(pair.ordinary->frontend().mels(), 129, 77);
    const auto output = pair.encode(input, 129, "decoder lifecycle input");
    const auto plain = pair.decode(output, unlimited, "plain CPU decoding");
    cancel_and_retry(pair, input, 129, output, plain, 0);
    const int blank = (int) pair.ordinary->model().u32("fastconformer.decoder.blank_id");
    const BoostingTree tree({{3, 4}, {5, 6}}, blank, 10.0, 1.0, 0.0f);
    const Boost boost{tree, 20.0f};
    pair.decode(output, unlimited, "boosted CPU decoding", DecodeOn::Cpu, &boost);
    const auto after_boost = pair.encode(input, 129, "same shape after boosting");
    pair.checks.expect(same_path(pair.decode(after_boost, unlimited, "plain request after boosting"), plain),
                       "keyword state does not cross requests");
    pair.decode(output, unlimited, "model-device decoding", DecodeOn::Device);
    pair.encode(input, 129, "same shape after changing decoder device");
}

void audio_inputs(Pair & pair, const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        const auto audio = read_npy(path);
        require(audio.shape.size() == 1 && !audio.f32.empty() && finite(audio.f32), "expected finite mono float32 audio in " + path);
        const auto & frontend = pair.ordinary->frontend();
        const int64_t frames = frontend.frames(audio.f32.size());
        require(frames > 0, "audio contains no complete frontend hop: " + path);
        const auto input = frontend.features(audio.f32);
        const auto output = pair.encode(input, frames, "audio " + path);
        const auto plain = pair.decode(output, std::numeric_limits<size_t>::max(), "audio TDT path");
        const bool eligible = pair.last_callbacks >= 3;
        std::printf("after-step-cancel candidate=%s samples=%zu frames=%lld minimum_full_callbacks=%zu eligible=%s\n",
                    path.c_str(), audio.f32.size(), (long long) frames, pair.last_callbacks, eligible ? "true" : "false");
        if (eligible && !pair.after_step_fixture_used) {
            std::printf("after-step-cancel fixture=%s\n", path.c_str());
            for (size_t allowed : {size_t{1}, size_t{2}}) cancel_and_retry(pair, input, frames, output, plain, allowed);
            pair.after_step_fixture_used = true;
        }
        const auto before_a = pair.ordinary->encoder_graph_cache_info(), before_b = pair.cached->encoder_graph_cache_info();
        const auto a = pair.ordinary->recognize(audio.f32), b = pair.cached->recognize(audio.f32);
        pair.counters(before_a, before_b, frames, "same-audio recognize replay");
        pair.checks.expect(a.text == b.text && same_path(a.decoding, b.decoding), "whole recognize replay agrees");
    }
}

void owner_destruction(Pair & pair, const std::string & path, ggml_backend_t backend, int threads) {
    const auto input = features(pair.ordinary->frontend().mels(), 129, 91);
    const auto before = pair.encode(input, 129, "owner destruction input");
    pair.cached.reset();
    pair.checks.values(pair.ordinary->encode(input, 129), before.first, "ordinary owner survives cached model destruction");
    setting("SPEECH_HAAS_ENCODER_GRAPH_CACHE", "1");
    pair.cached = std::make_unique<Recognizer>(path, backend, threads);
    setting("SPEECH_HAAS_ENCODER_GRAPH_CACHE", "0");
    const auto empty = pair.cached->encoder_graph_cache_info();
    pair.checks.expect(empty.enabled && empty.frames == -1 && empty.builds == 0 && empty.hits == 0 &&
                          empty.feature_bytes == 0 && empty.compute_bytes == 0,
                      "a new model has no graph, allocator buffers or counters from the destroyed owner");
    const auto after = pair.encode(input, 129, "new model after cache destruction");
    pair.checks.values(after.second, before.second, "model recreation preserves the final encoder bytes");
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        require(args.size() >= 5, "usage: fastconformer-encoder-graph-check <model.gguf> <device> <threads> <audio.npy> [audio.npy ...]");
        const int threads = std::stoi(args[3]);
        require(threads > 0, "threads must be positive");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[2]), threads), ggml_backend_free);
        Checks checks;
        Pair pair(args[1], backend.get(), threads, checks);
        std::printf("backend=%s threads=%d real_audio_inputs=%zu no_official_reference=true\n",
                    ggml_backend_name(backend.get()), threads, args.size() - 4);
        shapes(pair);
        request_states(pair);
        audio_inputs(pair, std::vector<std::string>(args.begin() + 4, args.end()));
        checks.expect(pair.after_step_fixture_used,
                      "after-step cancellation is unverified: supply real audio reaching at least three decoder callbacks");
        owner_destruction(pair, args[1], backend.get(), threads);
        std::printf("encoder graph: %zu checks, %zu failed; representative worker hit rate and API cancellation are not checked\n",
                    checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
