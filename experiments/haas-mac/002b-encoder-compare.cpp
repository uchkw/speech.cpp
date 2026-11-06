// Measures paired encoder tensors and checks emitted TDT decisions with FFN expansion disabled and enabled.
// Usage: encoder-compare model.gguf audio_dump_directory...
// The audio is benchmark input, not an official reference dump, and no official stage threshold is applied.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "fastconformer/recognizer.h"
#include "json.h"
#include "npy.h"

namespace {

struct Snapshot {
    std::map<std::string, std::vector<float>> tensors;
    fastconformer::Decoding decoding;
    std::string text;
};

Snapshot evaluate(fastconformer::Recognizer & recognizer, ggml_backend_t backend, ggml_gallocr_t allocator,
                  const std::vector<float> & audio) {
    Snapshot result;
    auto & features = result.tensors["features"];
    features = recognizer.frontend().features(audio);
    Graph graph;
    fastconformer::EncoderStages stages;
    auto * encoded = recognizer.encoder().build(graph, features, recognizer.frontend().frames(audio.size()), &stages);
    auto * projected = recognizer.joint().project_encoder(graph.ctx(), encoded);
    graph.output(stages.subsampled);
    for (auto * layer : stages.layers) graph.output(layer);
    graph.output(encoded);
    graph.output(projected);
    graph.compute(backend, allocator);
    result.tensors["subsampled"] = Graph::read(stages.subsampled);
    for (size_t i = 0; i < stages.layers.size(); ++i) result.tensors["layer_" + std::to_string(i)] = Graph::read(stages.layers[i]);
    result.tensors["encoded"] = Graph::read(encoded);
    result.tensors["projected"] = Graph::read(projected);
    result.decoding = recognizer.decoding(result.tensors.at("projected"), "greedy", fastconformer::DecodeOn::Cpu);
    result.text = recognizer.detokenizer().text(result.decoding.ids);
    return result;
}

/** Reports the rounding difference without borrowing an official-reference tolerance for these paired inputs. */
bool difference(const std::string & input, const std::string & name, const std::vector<float> & actual,
                const std::vector<float> & reference) {
    if (actual.size() != reference.size()) throw std::runtime_error("Paired tensor shapes differ");
    double error = 0, power = 0, largest = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(reference[i])) throw std::runtime_error("A paired tensor is not finite");
        const double delta = (double) actual[i] - reference[i];
        error += delta * delta;
        power += (double) reference[i] * reference[i];
        largest = std::max(largest, std::abs(delta));
    }
    const bool exact = std::memcmp(actual.data(), reference.data(), actual.size() * sizeof(float)) == 0;
    std::printf("{\"input\":%s,\"tensor\":%s,\"elements\":%zu,\"equal_bits\":%s,\"nmse\":%.12g,\"absolute_difference_max\":%.12g}\n",
                json_string(input).c_str(), json_string(name).c_str(), actual.size(), exact ? "true" : "false",
                power > 0 ? error / power : error, largest);
    return exact;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: encoder-compare model.gguf audio_dump_directory...");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device("gpu"), 4), ggml_backend_free);
        setenv("SPEECH_HAAS_FFN_F16", "0", 1);
        fastconformer::Recognizer baseline(argv[1], backend.get(), 4);
        setenv("SPEECH_HAAS_FFN_F16", "1", 1);
        fastconformer::Recognizer candidate(argv[1], backend.get(), 4);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        if (!allocator) throw std::runtime_error("Cannot allocate the paired encoder graph");
        const bool native = !ggml_is_quantized(baseline.model().tensor("blk.0.ff1_up.weight")->type);
        bool ok = true;
        for (int i = 2; i < argc; ++i) {
            const auto directory = std::filesystem::u8path(argv[i]);
            const std::string input = directory.filename().u8string();
            const auto audio = read_npy((directory / "audio.npy").u8string()).f32;
            const Snapshot left = evaluate(baseline, backend.get(), allocator.get(), audio);
            const Snapshot right = evaluate(candidate, backend.get(), allocator.get(), audio);
            bool exact = true;
            for (const auto & item : left.tensors) {
                const bool equal = difference(input, item.first, right.tensors.at(item.first), item.second);
                if (item.first == "features" || item.first == "subsampled" || native) ok = ok && equal;
                exact = exact && equal;
            }
            const bool ids = left.decoding.ids == right.decoding.ids;
            const bool frames = left.decoding.frames == right.decoding.frames;
            const bool durations = left.decoding.durations == right.decoding.durations;
            const bool text = left.text == right.text;
            const bool graphs = left.decoding.graphs == right.decoding.graphs;
            ok = ok && ids && frames && durations && text && graphs;
            std::printf("{\"input\":%s,\"all_tensors_equal_bits\":%s,\"emitted_token_ids_equal\":%s,\"emission_frames_equal\":%s,\"emitted_durations_equal\":%s,\"text_equal\":%s,\"graph_counts_equal\":%s,\"baseline_tokens\":%zu,\"candidate_tokens\":%zu,\"baseline_graphs\":%zu,\"candidate_graphs\":%zu}\n",
                        json_string(input).c_str(), exact ? "true" : "false", ids ? "true" : "false", frames ? "true" : "false",
                        durations ? "true" : "false", text ? "true" : "false", graphs ? "true" : "false", left.decoding.ids.size(),
                        right.decoding.ids.size(), left.decoding.graphs, right.decoding.graphs);
        }
        std::printf("{\"status\":\"%s\",\"inputs\":%d,\"native_float_control\":%s,\"official_reference\":false,\"blank_duration_trace_observed\":false}\n",
                    ok ? "passed" : "failed", argc - 2, native ? "true" : "false");
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
