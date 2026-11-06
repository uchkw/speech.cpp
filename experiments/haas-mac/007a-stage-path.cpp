// Records raw TDT paths after timing FastConformer's frontend, encoder graph construction and execution, readback
// and decoding on audio.npy inputs, mono float32 at the model's sample rate. One untimed recognition of each input warms the device.
// Each JSON line is one measured run; core_total_ms
// includes detokenization but excludes input loading and model loading. Profiling does not change graph boundaries.
//
// The decoding computes on the CPU's copy of its tensors (cpu) or on the model's device (device).
//
// usage: 007a-stage-path <model.gguf> <gpu|cpu|device name> <decoding> <cpu|device> <runs> <threads> <audio dir>...

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "fastconformer/recognizer.h"
#include "json.h"
#include "npy.h"

using namespace fastconformer;

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename T>
void write_array(const char * key, const std::vector<T> & values) {
    std::printf(",\"%s\":[", key);
    for (size_t i = 0; i < values.size(); ++i) std::printf("%s%lld", i ? "," : "", (long long) values[i]);
    std::printf("]");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 8 || (args[4] != "cpu" && args[4] != "device")) {
        std::fprintf(stderr, "usage: %s <model.gguf> <gpu|cpu|device name> <decoding> <cpu|device> <runs> <threads> <audio dir>...\n", args[0].c_str());
        return 2;
    }
    try {
        const DecodeOn on = args[4] == "cpu" ? DecodeOn::Cpu : DecodeOn::Device;
        const int runs = std::stoi(args[5]);
        const int threads = std::stoi(args[6]);
        if (runs < 1) throw std::runtime_error("runs must be positive");
        if (threads < 1) throw std::runtime_error("threads must be positive");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[2]), threads), ggml_backend_free);
        Recognizer recognizer(args[1], backend.get(), threads);
        recognizer.decoder(args[3]);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        if (!allocator) throw std::runtime_error("cannot create an encoder allocator");
        for (size_t a = 7; a < args.size(); a++) {
            const auto dir = std::filesystem::u8path(args[a]);
            const auto audio = read_npy((dir / "audio.npy").u8string()).f32;
            for (int run = -1; run < runs; run++) {
                const auto start = Clock::now();
                const auto features = recognizer.frontend().features(audio);
                const auto frontend_end = Clock::now();
                Graph graph;
                auto * encoded = recognizer.encoder().build(graph, features, recognizer.frontend().frames(audio.size()));
                auto * projected = recognizer.joint().project_encoder(graph.ctx(), encoded);
                graph.output(projected);
                const auto build_end = Clock::now();
                graph.compute(backend.get(), allocator.get());
                const auto compute_end = Clock::now();
                const auto output = Graph::read(projected);
                const auto read_end = Clock::now();
                const auto decoding = recognizer.decoding(output, args[3], on);
                const auto decode_end = Clock::now();
                const auto text = recognizer.detokenizer().text(decoding.ids);
                const auto end = Clock::now();
                if (run < 0) continue;
                std::printf("{\"input\":%s,\"device\":%s,\"decoding\":%s,\"decode_on\":%s,\"threads\":%d,\"run\":%d,\"audio_seconds\":%.4f,"
                            "\"core_total_ms\":%.4f,\"frontend_ms\":%.4f,\"encoder_build_ms\":%.4f,"
                            "\"encoder_compute_ms\":%.4f,\"encoder_read_ms\":%.4f,\"decoder_ms\":%.4f,"
                            "\"graphs\":%zu,\"tokens\":%zu,\"text\":%s",
                            json_string(dir.filename().u8string()).c_str(), json_string(ggml_backend_name(backend.get())).c_str(),
                            json_string(args[3]).c_str(), json_string(args[4]).c_str(), threads, run, (double) audio.size() / recognizer.sample_rate(),
                            ms(start, end),
                            ms(start, frontend_end), ms(frontend_end, build_end), ms(build_end, compute_end), ms(compute_end, read_end),
                            ms(read_end, decode_end), decoding.graphs, decoding.ids.size(), json_string(text).c_str());
                write_array("ids", decoding.ids);
                write_array("frames", decoding.frames);
                write_array("durations", decoding.durations);
                std::printf("}\n");
                std::fflush(stdout);
            }
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
