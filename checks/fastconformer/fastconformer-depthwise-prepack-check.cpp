// Compares the actual depthwise weight operands with their original PERMUTE/CONT products and unchanged model bytes.
// Ordinary final joint-input comparisons mark no intermediate outputs; a separate retention diagnostic reads weights.
// Usage: fastconformer-depthwise-prepack-check <model.gguf> <cpu|gpu|device> <threads> [audio.npy ...]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "error.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "npy.h"

using namespace fastconformer;

namespace {

void require(bool ok, const std::string & message) {
    if (!ok) throw std::runtime_error(message);
}

void setting(const char * value) {
#ifdef _WIN32
    require(_putenv_s("SPEECH_HAAS_DEPTHWISE_PREPACK", value) == 0, "cannot set the depthwise prepack switch");
#else
    require(setenv("SPEECH_HAAS_DEPTHWISE_PREPACK", value, 1) == 0, "cannot set the depthwise prepack switch");
#endif
}

std::vector<uint8_t> bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & message) {
        ++total;
        if (!ok) { ++failed; std::printf("FAIL: %s\n", message.c_str()); }
    }

    bool values(const std::vector<float> & a, const std::vector<float> & b, const std::string & name, bool gate = true) {
        const bool finite = std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(b.begin(), b.end(), [](float x) { return std::isfinite(x); });
        const bool exact = finite && a.size() == b.size() &&
            (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
        if (gate) expect(exact, name);
        else expect(finite && a.size() == b.size(), name + " finite diagnostic values and matching shapes");
        const auto diff = compare(a, b);
        std::printf("%s gate=%s exact=%s max_abs=%.9g snr_db=%.9g\n", name.c_str(), gate ? "true" : "false",
                    exact ? "true" : "false", diff.max_abs, diff.snr_db);
        return exact;
    }
};

struct Operand {
    std::array<int64_t, GGML_MAX_DIMS> shape;
    std::array<size_t, GGML_MAX_DIMS> strides;
    ggml_tensor * storage;
};

struct Result {
    std::vector<float> output;
    std::map<std::string, Operand> kernels;
    std::map<std::string, std::vector<uint8_t>> weight_bytes;
    size_t contiguous_nodes = 0;
};

/** One allocator and encoder survive the shape sequence; only the weight diagnostic retains intermediate outputs. */
struct Run {
    const ModelFile & model;
    ggml_backend_t backend;
    const Encoder & encoder;
    const Joint & joint;
    bool prepared;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator;

    Run(const ModelFile & model, ggml_backend_t backend, const Encoder & encoder, const Joint & joint, bool prepared)
        : model(model), backend(backend), encoder(encoder), joint(joint), prepared(prepared),
          allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {
        require(bool(allocator), "cannot allocate the depthwise check graph");
    }

    Result run(const std::vector<float> & features, int64_t frames, bool retain_weights = false) {
        Graph graph;
        auto * output = joint.project_encoder(graph.ctx(), encoder.build(graph, features, frames));
        graph.output(output);
        Result result;
        for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
            auto * node = ggml_graph_node(graph.cgraph(), i);
            if (node->op == GGML_OP_CONT) ++result.contiguous_nodes;
            if (node->op != GGML_OP_CONV_2D_DW) continue;
            auto * kernel = node->src[0];
            require(kernel->op == GGML_OP_PERMUTE && kernel->type == GGML_TYPE_F32, "a depthwise kernel is not the final F32 permutation");
            auto * storage = kernel->src[0];
            std::string name;
            if (prepared) {
                const std::string suffix = ".prepacked", full = ggml_get_name(storage);
                require(storage->op == GGML_OP_NONE && full.size() > suffix.size() &&
                            full.compare(full.size() - suffix.size(), suffix.size(), suffix) == 0,
                        "a depthwise convolution does not read an owned prepared tensor");
                name = full.substr(0, full.size() - suffix.size());
                require(ggml_backend_buffer_get_type(storage->buffer) == ggml_backend_buffer_get_type(model.tensor(name)->buffer) &&
                            ggml_backend_buffer_get_usage(storage->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
                        "prepared depthwise weights change their buffer type or usage");
            } else {
                require(storage->op == GGML_OP_CONT && storage->src[0]->op == GGML_OP_PERMUTE,
                        "the ordinary depthwise kernel misses its original PERMUTE/CONT path");
                const auto * original = storage->src[0]->view_src;
                require(original != nullptr, "the ordinary kernel permutation has no original tensor");
                name = ggml_get_name(original);
                require(original == model.tensor(name), "the ordinary kernel does not read its original model weight");
            }
            require(storage->type == GGML_TYPE_F32 && ggml_is_contiguous(storage) && storage->ne[1] == 1 &&
                        storage->ne[0] == kernel->ne[3] && storage->ne[2] == kernel->ne[0] && storage->ne[3] == kernel->ne[1],
                    "the depthwise storage is not contiguous [channels,1,kw,kh]");
            Operand operand;
            std::copy(kernel->ne, kernel->ne + GGML_MAX_DIMS, operand.shape.begin());
            std::copy(kernel->nb, kernel->nb + GGML_MAX_DIMS, operand.strides.begin());
            operand.storage = storage;
            require(result.kernels.emplace(name, operand).second, "one depthwise weight appears twice in the graph");
        }
        require(result.kernels.size() == 26, "the Haas graph does not contain exactly 26 depthwise convolutions");
        if (retain_weights && !prepared) {
            for (const auto & entry : result.kernels) graph.output(entry.second.storage);
        }
        graph.compute(backend, allocator.get());
        result.output = Graph::read(output);
        if (retain_weights) {
            for (const auto & entry : result.kernels) result.weight_bytes[entry.first] = bytes(entry.second.storage);
        }
        return result;
    }
};

void contract(const Result & ordinary, const Result & prepared, Checks & checks) {
    checks.expect(ordinary.contiguous_nodes == prepared.contiguous_nodes + 26, "the actual graph removes exactly 26 CONT nodes");
    checks.expect(ordinary.kernels.size() == prepared.kernels.size(), "both graphs use the same depthwise weight count");
    for (const auto & entry : ordinary.kernels) {
        const auto & got = prepared.kernels.at(entry.first);
        checks.expect(got.shape == entry.second.shape && got.strides == entry.second.strides,
                      "the actual convolution retains its kernel shape and byte strides: " + entry.first);
    }
    std::printf("graph CONT nodes: ordinary=%zu prepared=%zu removed=%lld\n", ordinary.contiguous_nodes,
                prepared.contiguous_nodes, (long long) ordinary.contiguous_nodes - (long long) prepared.contiguous_nodes);
}

std::vector<float> features(int mels, int64_t frames, int seed) {
    std::vector<float> result((size_t) frames * mels);
    for (size_t i = 0; i < result.size(); ++i) result[i] = 0.2f * std::sin((float) ((i + seed * 131) % 997) * 0.031f);
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        require(args.size() >= 4, "usage: fastconformer-depthwise-prepack-check <model.gguf> <device> <threads> [audio.npy ...]");
        const int threads = std::stoi(args[3]);
        require(threads > 0, "threads must be positive");
        setting("0");
        const ModelFile metadata(args[1], layout);
        const bool quantized = quantized_decoding(metadata);
        setting("1");
        {
            const ModelFile metadata_prepared(args[1], layout);
            require(quantized_decoding(metadata_prepared) == quantized,
                    "metadata-only decoding inspection changes with depthwise prepacking");
        }
        setting("0");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[2]), threads), ggml_backend_free);
        Recognizer recognizer(args[1], backend.get(), threads);
        const auto & model = recognizer.model();
        require(model.str("general.name") == "Haas-ja-v1.0" && model.u32("fastconformer.encoder.num_layers") == 24 &&
                    recognizer.encoder().subsampling_factor() == 8, "this check requires the Haas model's 24 layers and factor-eight subsampling");
        std::map<std::string, std::vector<uint8_t>> originals;
        for (int i = 1; i < 3; ++i) {
            const std::string name = "sub.conv." + std::to_string(i) + ".dw.weight";
            originals[name] = bytes(model.tensor(name));
        }
        for (int i = 0; i < 24; ++i) {
            const std::string name = "blk." + std::to_string(i) + ".conv_dw.weight";
            originals[name] = bytes(model.tensor(name));
        }
        Checks checks;
        setting("invalid");
        bool refused = false;
        try { const Encoder invalid(model, backend.get()); }
        catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
        checks.expect(refused, "the depthwise prepack selector refuses values other than zero and one");
        setting("0");
        Encoder ordinary(model, backend.get());
        require(ordinary.depthwise_prepack_bytes() == 0, "disabled prepacking retains additional weights");
        Run a(model, backend.get(), ordinary, recognizer.joint(), false);
        const auto short_input = features(recognizer.frontend().mels(), 129, 1);
        const auto first = a.run(short_input, 129);
        {
            setting("1");
            Encoder prepared(model, backend.get());
            const size_t retained = prepared.depthwise_prepack_bytes();
            require(retained >= 903168, "the additional depthwise buffer is smaller than its 882 KiB payload");
            setting("0");
            Run b(model, backend.get(), prepared, recognizer.joint(), true);
            const auto second = b.run(short_input, 129);
            contract(first, second, checks);
            checks.values(second.output, first.output, "ordinary final joint input");
            const auto diagnostic_a = a.run(short_input, 129, true), diagnostic_b = b.run(short_input, 129, true);
            contract(diagnostic_a, diagnostic_b, checks);
            size_t payload = 0;
            for (const auto & entry : diagnostic_a.weight_bytes) {
                checks.expect(entry.second == diagnostic_b.weight_bytes.at(entry.first), "prepared bytes equal the actual CONT product: " + entry.first);
                payload += entry.second.size();
            }
            checks.expect(payload == 903168, "all 26 prepared tensors retain exactly 882 KiB of weight payload");
            checks.values(diagnostic_a.output, first.output, "diagnostic ordinary retention versus normal", false);
            checks.values(diagnostic_b.output, second.output, "diagnostic prepared retention versus normal", false);
            for (int64_t frames : {1361, 1369, 3000, 129}) {
                const auto input = frames == 129 ? short_input : features(recognizer.frontend().mels(), frames, 2);
                const auto x = a.run(input, frames), y = b.run(input, frames);
                contract(x, y, checks);
                checks.values(y.output, x.output, "normal shape " + std::to_string(frames));
                for (const auto & entry : y.kernels) {
                    checks.expect(entry.second.storage == second.kernels.at(entry.first).storage, "prepared tensor remains owned across requests: " + entry.first);
                }
                checks.expect(prepared.depthwise_prepack_bytes() == retained, "prepared allocation does not change with input length");
                if (frames == 129) checks.values(y.output, second.output, "normal long-to-short output");
            }
            for (size_t i = 4; i < args.size(); ++i) {
                const auto audio = read_npy(args[i]);
                require(audio.shape.size() == 1 && !audio.f32.empty() &&
                            std::all_of(audio.f32.begin(), audio.f32.end(), [](float x) { return std::isfinite(x); }),
                        "expected finite mono F32 audio in " + args[i]);
                const int64_t frames = recognizer.frontend().frames(audio.f32.size());
                require(frames > 0, "audio contains no complete frontend hop: " + args[i]);
                const auto input = recognizer.frontend().features(audio.f32);
                const auto x = a.run(input, frames), y = b.run(input, frames);
                contract(x, y, checks);
                checks.values(y.output, x.output, "normal audio " + args[i]);
            }
            std::printf("additional depthwise weight payload=%zu retained_backend_bytes=%zu\n", payload, retained);
        }
        checks.values(a.run(short_input, 129).output, first.output, "original encoder after prepared owner destruction");
        for (const auto & entry : originals) checks.expect(bytes(model.tensor(entry.first)) == entry.second, "original weight bytes remain unchanged: " + entry.first);
        std::printf("backend=%s threads=%d checks=%zu failed=%zu; official references, API cancellation and timing are not checked\n",
                    ggml_backend_name(backend.get()), threads, checks.total, checks.failed);
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
