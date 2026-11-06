// Observes ordinary Haas head-merge allocations and optionally saves final joint-projected encoder outputs.
// Usage: encoder-head-layout model.gguf allocate|compute output-directory audio-directory...
// Allocation mode does not execute the encoder graph; building it can populate the model's position cache.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

#include "args.h"
#include "backend.h"
#include "fastconformer/recognizer.h"
#include "json.h"
#include "log.h"
#include "npy.h"
#include "../../ggml/src/ggml-backend-impl.h"
#include "../../ggml/src/ggml-metal/ggml-metal-device.h"

static ggml_metal_buffer_id binding(const ggml_tensor * tensor) {
    auto * buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    return ggml_metal_buffer_get_id(static_cast<ggml_metal_buffer_t>(buffer->context), tensor);
}

static bool overlaps(const ggml_tensor * a, const ggml_tensor * b) {
    const auto ba = binding(a), bb = binding(b);
    if (ba.metal != bb.metal) return false;
    return ba.offs <= bb.offs ? bb.offs - ba.offs < ggml_nbytes(a) : ba.offs - bb.offs < ggml_nbytes(b);
}

static int observe(ggml_cgraph * graph, const std::string & input) {
    int found = 0, eligible = 0;
    for (int i = 0; i + 2 < ggml_graph_n_nodes(graph); i++) {
        auto * mm = ggml_graph_node(graph, i);
        auto * perm = ggml_graph_node(graph, i + 1);
        auto * cont = ggml_graph_node(graph, i + 2);
        if (mm->op != GGML_OP_MUL_MAT || perm->op != GGML_OP_PERMUTE || cont->op != GGML_OP_CONT ||
            perm->src[0] != mm || cont->src[0] != perm || mm->ne[0] != 128 || mm->ne[2] != 8 ||
            mm->ne[3] != 1 || cont->ne[0] != 128 || cont->ne[1] != 8 || cont->ne[2] != mm->ne[1]) continue;
        const auto * a = mm->src[0];
        const auto * b = mm->src[1];
        const auto ba = binding(a), bb = binding(b), bc = binding(cont);
        const bool same = ba.metal == bc.metal && bb.metal == bc.metal;
        const bool alias_a = overlaps(cont, a), alias_b = overlaps(cont, b);
        const bool range_eligible = mm->ne[1] >= 64 && same && !alias_a && !alias_b;
        found++;
        eligible += range_eligible;
        std::printf("{\"kind\":\"head_allocation\",\"input\":%s,\"node\":%d,\"frames\":%lld,"
                    "\"a_buffer\":\"%p\",\"a_offset\":%zu,\"a_bytes\":%zu,"
                    "\"b_buffer\":\"%p\",\"b_offset\":%zu,\"b_bytes\":%zu,"
                    "\"cont_buffer\":\"%p\",\"cont_offset\":%zu,\"cont_bytes\":%zu,"
                    "\"same_buffer\":%s,\"overlap_a\":%s,\"overlap_b\":%s,\"range_eligible\":%s}\n",
                    json_string(input).c_str(), i, (long long) mm->ne[1],
                    ba.metal, ba.offs, ggml_nbytes(a), bb.metal, bb.offs, ggml_nbytes(b),
                    bc.metal, bc.offs, ggml_nbytes(cont), same ? "true" : "false", alias_a ? "true" : "false",
                    alias_b ? "true" : "false", range_eligible ? "true" : "false");
    }
    if (found != 24) throw std::runtime_error("the ordinary Haas graph does not contain 24 head merges");
    std::printf("{\"kind\":\"allocation_summary\",\"input\":%s,\"patterns\":%d,\"range_eligible\":%d,"
                "\"extra_output_retention\":false,\"backend_graph_optimized\":false,\"fusion_stats_initialized\":false}\n",
                json_string(input).c_str(), found, eligible);
    std::fflush(stdout);
    return eligible;
}

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 5 || (args[2] != "allocate" && args[2] != "compute")) return 2;
    try {
        if (std::getenv("GGML_METAL_FUSION_DEBUG") || std::getenv("GGML_METAL_FUSION_DISABLE")) {
            throw std::runtime_error("the ordinary-policy diagnostic forbids fusion debug or disabling generic fusion");
        }
        set_log_sink([](LogLevel, const char * text) { std::fputs(text, stderr); });
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            start_device(find_device("gpu"), 4), ggml_backend_free);
        fastconformer::Recognizer recognizer(args[1], backend.get(), 4);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        if (!allocator) throw std::runtime_error("cannot create an encoder allocator");
        const auto destination = std::filesystem::u8path(args[3]);
        if (args[2] == "compute") std::filesystem::create_directories(destination);
        for (size_t i = 4; i < args.size(); i++) {
            const auto directory = std::filesystem::u8path(args[i]);
            const auto audio = read_npy((directory / "audio.npy").u8string()).f32;
            const auto features = recognizer.frontend().features(audio);
            const auto frames = recognizer.frontend().frames(audio.size());
            Graph graph;
            auto * encoded = recognizer.encoder().build(graph, features, frames);
            auto * projected = recognizer.joint().project_encoder(graph.ctx(), encoded);
            graph.output(projected);
            if (!ggml_gallocr_alloc_graph(allocator.get(), graph.cgraph())) {
                throw std::runtime_error("cannot allocate the ordinary encoder graph");
            }
            const auto name = directory.filename().u8string();
            const int eligible = observe(graph.cgraph(), name);
            if (args[2] == "allocate") continue;
            if (encoded->ne[1] >= 64 && eligible == 0) {
                throw std::runtime_error("the normal allocator permits no head-store fusion; stop before encoder execution");
            }
            std::fprintf(stderr, "DIAGNOSTIC_BEGIN %s\n", name.c_str());
            graph.compute_again(backend.get());
            std::fprintf(stderr, "DIAGNOSTIC_END %s\n", name.c_str());
            const auto output = Graph::read(projected);
            for (float value : output) {
                if (!std::isfinite(value)) throw std::runtime_error("the projected encoder output is not finite");
            }
            std::ofstream file(destination / (name + ".f32"), std::ios::binary);
            file.write(reinterpret_cast<const char *>(output.data()), output.size()*sizeof(float));
            file.close();
            if (!file) throw std::runtime_error("cannot save the projected encoder output");
            std::printf("{\"kind\":\"encoder_output\",\"input\":%s,\"elements\":%zu,\"finite\":true,\"timing\":false}\n",
                        json_string(name).c_str(), output.size());
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
