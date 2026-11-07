// Compares normal ggml encoder projections and external Core ML projections through the unchanged CPU-4 TDT decoder.
// Usage: haas-coreml-tdt-check MODEL.gguf gpu|cpu FEATURES.npy PROJECTED.npy [FEATURES.npy PROJECTED.npy ...]
// This is a core correctness check, not a worker request benchmark or an official KDA reference.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

#include "args.h"
#include "backend.h"
#include "fastconformer/recognizer.h"
#include "json.h"
#include "npy.h"

using namespace fastconformer;

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void tensor(const Npy & value, int64_t width, const std::string & name) {
    require(value.shape.size() == 2 && value.shape[0] > 0 && value.shape[1] == width &&
            value.f32.size() == (size_t) value.size(), name + " must be nonempty row-major F32 with the model's width");
    require(std::all_of(value.f32.begin(), value.f32.end(), [](float x) { return std::isfinite(x); }),
            name + " contains a nonfinite value");
}

void print_decoding(const char * source, const std::string & input, const Decoding & d,
                    const Recognizer & recognizer, const std::vector<double> & progress) {
    const auto texts = recognizer.detokenizer().token_texts(d.ids);
    const auto spans = token_spans(d, recognizer.detokenizer());
    const auto segments = recognizer.segments(d);
    std::printf("{\"source\":%s,\"input\":%s,\"graphs\":%zu,\"text\":%s,\"tokens\":[",
                json_string(source).c_str(), json_string(input).c_str(), d.graphs,
                json_string(recognizer.detokenizer().text(d.ids)).c_str());
    for (size_t i = 0; i < d.ids.size(); ++i) {
        std::printf("%s{\"id\":%d,\"frame\":%lld,\"duration\":%d,\"text\":%s,\"start\":%.17g,\"end\":%.17g}",
                    i ? "," : "", d.ids[i], (long long) d.frames[i], d.durations[i], json_string(texts[i]).c_str(),
                    recognizer.seconds(spans[i].start), recognizer.seconds(spans[i].end));
    }
    std::printf("],\"segments\":[");
    for (size_t i = 0; i < segments.size(); ++i) {
        const auto & segment = segments[i];
        std::printf("%s{\"first\":%zu,\"end_token\":%zu,\"text\":%s,\"start\":%.17g,\"end\":%.17g}",
                    i ? "," : "", segment.first, segment.end, json_string(segment.text).c_str(),
                    recognizer.seconds(segment.span.start), recognizer.seconds(segment.span.end));
    }
    std::printf("],\"progress\":[");
    for (size_t i = 0; i < progress.size(); ++i) std::printf("%s%.17g", i ? "," : "", progress[i]);
    std::printf("]}\n");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 5 || (args.size() - 3) % 2 || (args[2] != "gpu" && args[2] != "cpu")) {
        std::fprintf(stderr, "usage: %s MODEL.gguf gpu|cpu FEATURES.npy PROJECTED.npy [FEATURES.npy PROJECTED.npy ...]\n", args[0].c_str());
        return 2;
    }
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            start_device(find_device(args[2]), 4), ggml_backend_free);
        Recognizer recognizer(args[1], backend.get(), 4);
        const auto & model = recognizer.model();
        require(model.str("general.name") == "Haas-ja-v1.0" && model.str("fastconformer.decoder.prediction") == "kda" &&
                model.str("fastconformer.decoder.kind") == "tdt", "This experiment requires the pinned Haas KDA/TDT model");
        bool all_equal = true;
        for (size_t i = 3; i < args.size(); i += 2) {
            const auto features = read_npy(args[i]);
            const auto candidate = read_npy(args[i + 1]);
            tensor(features, 128, args[i]);
            tensor(candidate, recognizer.joint().hidden(), args[i + 1]);
            require(candidate.shape[0] == recognizer.encoder().subsampled_frames(features.shape[0]),
                    "Core ML projection must contain exactly the valid encoder frames");
            const auto baseline = recognizer.encode(features.f32, features.shape[0]);
            std::vector<double> baseline_progress, candidate_progress;
            const auto baseline_decoding = recognizer.decoding(baseline, "greedy", DecodeOn::Cpu, [&](double done) {
                baseline_progress.push_back(done);
                return true;
            });
            const auto candidate_decoding = recognizer.decoding(candidate.f32, "greedy", DecodeOn::Cpu, [&](double done) {
                candidate_progress.push_back(done);
                return true;
            });
            print_decoding("ggml_normal_encoder", args[i], baseline_decoding, recognizer, baseline_progress);
            print_decoding("coreml_normal_encoder", args[i + 1], candidate_decoding, recognizer, candidate_progress);
            const bool equal = baseline_decoding.ids == candidate_decoding.ids && baseline_decoding.frames == candidate_decoding.frames &&
                               baseline_decoding.durations == candidate_decoding.durations && baseline_progress == candidate_progress &&
                               baseline_decoding.graphs == candidate_decoding.graphs;
            all_equal = all_equal && equal;
            std::printf("{\"input\":%s,\"exact_tdt_decisions\":%s,\"scope\":\"Unboosted core TDT; not worker E2E or official Mac KDA validation\"}\n",
                        json_string(args[i]).c_str(), equal ? "true" : "false");
            std::fflush(stdout);
        }
        return all_equal ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
