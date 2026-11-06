// Executes ordinary Haas encoder graphs with graph logging and saves their projected outputs.
// Usage: encoder-trace model.gguf output-directory audio-directory...
// This diagnostic retains the ordinary command-buffer policy and does not measure latency.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "fastconformer/recognizer.h"
#include "json.h"
#include "log.h"
#include "npy.h"

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 4) return 2;
    try {
        if (std::getenv("GGML_METAL_FUSION_DEBUG") || std::getenv("GGML_METAL_FUSION_DISABLE")) {
            throw std::runtime_error("the ordinary-policy trace forbids fusion debug or disabling generic fusion");
        }
        set_log_sink([](LogLevel, const char * text) { std::fputs(text, stderr); });
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            start_device(find_device("gpu"), 4), ggml_backend_free);
        fastconformer::Recognizer recognizer(args[1], backend.get(), 4);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        if (!allocator) throw std::runtime_error("cannot create an encoder allocator");
        const auto destination = std::filesystem::u8path(args[2]);
        std::filesystem::create_directories(destination);
        for (size_t i = 3; i < args.size(); ++i) {
            const auto directory = std::filesystem::u8path(args[i]);
            const auto audio = read_npy((directory / "audio.npy").u8string()).f32;
            const auto features = recognizer.frontend().features(audio);
            const auto frames = recognizer.frontend().frames(audio.size());
            Graph graph;
            auto * encoded = recognizer.encoder().build(graph, features, frames);
            auto * projected = recognizer.joint().project_encoder(graph.ctx(), encoded);
            graph.output(projected);
            const auto name = directory.filename().u8string();
            std::fprintf(stderr, "DIAGNOSTIC_BEGIN %s\n", name.c_str());
            graph.compute(backend.get(), allocator.get());
            std::fprintf(stderr, "DIAGNOSTIC_END %s\n", name.c_str());
            const auto output = Graph::read(projected);
            for (float value : output) {
                if (!std::isfinite(value)) throw std::runtime_error("an encoder output is not finite");
            }
            const auto path = destination / (name + ".f32");
            std::ofstream file(path, std::ios::binary);
            file.write(reinterpret_cast<const char *>(output.data()), output.size()*sizeof(float));
            file.close();
            if (!file) throw std::runtime_error("cannot save the projected encoder output");
            std::printf("{\"input\":%s,\"frames\":%lld,\"encoded_frames\":%lld,\"graph_nodes\":%d,"
                        "\"output_elements\":%zu,\"finite\":true,\"timing\":false,\"fusion_stats_initialized\":false}\n",
                        json_string(name).c_str(), (long long) frames, (long long) encoded->ne[1],
                        ggml_graph_n_nodes(graph.cgraph()), output.size());
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
