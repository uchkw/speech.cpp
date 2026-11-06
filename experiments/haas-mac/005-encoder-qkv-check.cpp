// Checks packed QKV bytes, the actual encoder graph, strided heads and paired encoder stages.
// Usage: encoder-qkv-check model.gguf cpu|gpu [audio_dump_directory...]
// The paired inputs are not official reference tensors, and this check requires exact output bits.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "error.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "json.h"
#include "npy.h"

namespace {

using Products = std::map<std::string, ggml_tensor *>;
using Snapshot = std::map<std::string, std::vector<float>>;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void setting(const char * value) {
#ifdef _WIN32
    require(_putenv_s("SPEECH_HAAS_ENCODER_QKV", value) == 0, "cannot set the QKV experiment switch");
#else
    require(setenv("SPEECH_HAAS_ENCODER_QKV", value, 1) == 0, "cannot set the QKV experiment switch");
#endif
}

/** The graph's matrix products whose left operands are named model or packed weights. */
Products products(const Graph & graph, const std::set<std::string> & names) {
    Products result;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op != GGML_OP_MUL_MAT) continue;
        const std::string name = ggml_get_name(node->src[0]);
        if (names.count(name)) require(result.emplace(name, node).second, "a named weight is multiplied twice: " + name);
    }
    return result;
}

std::set<std::string> matrix_names(const ModelFile & model) {
    std::set<std::string> names;
    for (const TensorSpec & spec : fastconformer::layout.tensors(model)) names.insert(spec.name);
    for (int l = 0; l < model.count("fastconformer.encoder.num_layers"); ++l) {
        names.insert("blk." + std::to_string(l) + ".attn_qkv.weight");
    }
    return names;
}

/** The three production views of one packed product, ordered Q, K, V. */
std::array<ggml_tensor *, 3> head_views(const Graph & graph, ggml_tensor * product, int64_t heads) {
    std::array<ggml_tensor *, 3> result{};
    const int64_t width = product->ne[0] / 3, dk = width / heads, frames = product->ne[1];
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op != GGML_OP_VIEW || node->view_src != product) continue;
        if (node->ne[0] != dk || node->ne[1] != heads || node->ne[2] != frames || node->ne[3] != 1) continue;
        const size_t bytes = (size_t) width * sizeof(float);
        require(node->view_offs % bytes == 0 && node->view_offs / bytes < 3, "a QKV view has the wrong offset");
        const size_t part = node->view_offs / bytes;
        require(!result[part], "a QKV view is duplicated");
        require(node->nb[0] == sizeof(float) && node->nb[1] == (size_t) dk * sizeof(float) &&
                    node->nb[2] == 3 * bytes, "a QKV view loses the packed frame stride");
        result[part] = node;
    }
    for (auto * view : result) require(view, "the actual graph does not use every QKV view");
    return result;
}

/** Both views used by the score product and the materialized V layout keep the packed source's offsets. */
void head_consumers(const Graph & graph, const std::array<ggml_tensor *, 3> & views) {
    bool q_bias = false, k_scores = false, v_cont = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        const auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op == GGML_OP_ADD && node->src[0] == views[0]) q_bias = true;
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->op == GGML_OP_PERMUTE &&
            node->src[0]->src[0] == views[1]) k_scores = true;
        if (node->op == GGML_OP_CONT && node->src[0]->op == GGML_OP_PERMUTE &&
            node->src[0]->src[0] == views[2]) v_cont = true;
    }
    require(q_bias && k_scores && v_cont, "the packed heads do not reach the original attention operations");
}

void graph_contract(const ModelFile & model, const fastconformer::Encoder & ordinary,
                    const fastconformer::Encoder & packed, ggml_backend_t backend) {
    const int64_t frames = 264, mels = model.size("fastconformer.frontend.n_mels");
    const int64_t width = model.size("fastconformer.encoder.d_model"), heads = model.size("fastconformer.encoder.num_heads");
    const int layers = model.count("fastconformer.encoder.num_layers");
    const std::vector<float> features((size_t) frames * mels, 0.25f);
    Graph left, right;
    left.output(ordinary.build(left, features, frames));
    right.output(packed.build(right, features, frames));
    const auto names = matrix_names(model);
    const auto before = products(left, names), after = products(right, names);
    require(after.size() + (size_t) (2 * layers) == before.size(), "the encoder does not replace three QKV products by one");
    std::set<std::string> replaced;
    size_t total_bytes = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".attn";
        auto * product = after.at(prefix + "_qkv.weight");
        const auto * weight = product->src[0];
        const auto * q = model.tensor(prefix + "_q.weight");
        require(weight->type == q->type && weight->ne[0] == width && weight->ne[1] == 3 * width &&
                    weight->ne[2] == 1 && weight->ne[3] == 1 && ggml_is_contiguous(weight), "the packed weight has a wrong type or layout");
        require(ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
                "the packed buffer is not marked as weights");
        size_t offset = 0;
        for (const char * part : {"q", "k", "v"}) {
            const std::string name = prefix + "_" + part + ".weight";
            const auto * source = model.tensor(name);
            require(before.at(name)->src[0] == source && !after.count(name), "a QKV product does not use the selected weight");
            require(source->type == weight->type, "a packed QKV source has another type");
            std::vector<uint8_t> reference(ggml_nbytes(source)), actual(reference.size());
            ggml_backend_tensor_get(source, reference.data(), 0, reference.size());
            ggml_backend_tensor_get(weight, actual.data(), offset, actual.size());
            require(actual == reference, "a packed QKV row differs from the source bytes: " + name);
            offset += actual.size();
            replaced.insert(name);
        }
        require(offset == ggml_nbytes(weight), "the packed weight size differs from its three sources");
        total_bytes += offset;
        head_consumers(right, head_views(right, product, heads));
        std::printf("{\"kind\":\"weight\",\"layer\":%d,\"type\":%s,\"bytes\":%zu,\"all_bytes_equal\":true}\n",
                    layer, json_string(ggml_type_name(weight->type)).c_str(), offset);
    }
    for (const auto & item : before) {
        require(item.second->src[0] == model.tensor(item.first), "the disabled switch replaces an original weight");
        if (!replaced.count(item.first)) require(after.at(item.first)->src[0] == item.second->src[0], "another weight operand changes");
    }
    for (int i = 0; i < ggml_graph_n_nodes(right.cgraph()); ++i) {
        const auto * node = ggml_graph_node(right.cgraph(), i);
        require(ggml_backend_supports_op(backend, node), "the backend does not support an actual packed encoder node");
    }
    std::printf("{\"kind\":\"graph\",\"packed_products\":%d,\"removed_products\":%d,\"retained_bytes\":%zu,\"unselected_weights_unchanged\":true}\n",
                layers, 2 * layers, total_bytes);
}

/** A changed descriptor is sufficient to test the refusal before a mixed source's bytes are ever read. */
void refusal_contract(const ModelFile & model, ggml_backend_t backend) {
    setting("invalid");
    bool refused = false;
    try {
        fastconformer::Encoder encoder(model, backend);
    } catch (const Error & error) {
        refused = error.fault() == Fault::InvalidArgument;
    }
    require(refused, "an invalid QKV switch is not refused");
    auto * key = model.tensor("blk.0.attn_k.weight");
    const ggml_type original = key->type;
    key->type = original == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
    try {
        setting("0");
        fastconformer::Encoder disabled(model, backend);
        setting("1");
        refused = false;
        try {
            fastconformer::Encoder encoder(model, backend);
        } catch (const Error & error) {
            refused = error.fault() == Fault::InvalidArgument;
        }
        require(refused, "mixed QKV weight types are not refused");
    } catch (...) {
        key->type = original;
        throw;
    }
    key->type = original;
    std::printf("{\"kind\":\"refusal\",\"invalid_switch_refused\":true,\"mixed_descriptor_refused\":true,\"disabled_descriptor_untouched\":true}\n");
}

/** Runs the ordinary graph first, then one that retains every layer and attention result for comparison. */
Snapshot evaluate(const fastconformer::Encoder & encoder, const fastconformer::Joint & joint, const ModelFile & model,
                  ggml_backend_t backend, ggml_gallocr_t allocator, const std::vector<float> & features, int64_t frames,
                  bool packed) {
    Snapshot result;
    {
        Graph graph;
        auto * out = joint.project_encoder(graph.ctx(), encoder.build(graph, features, frames));
        graph.output(out);
        graph.compute(backend, allocator);
        result["actual_projected"] = Graph::read(out);
    }
    Graph graph;
    fastconformer::EncoderStages stages;
    auto * encoded = encoder.build(graph, features, frames, &stages);
    auto * projected = joint.project_encoder(graph.ctx(), encoded);
    graph.output(projected);
    const auto matrices = products(graph, matrix_names(model));
    std::map<std::string, ggml_tensor *> retained{{"subsampled", stages.subsampled}, {"encoded", encoded}, {"projected", projected}};
    const int64_t heads = model.size("fastconformer.encoder.num_heads");
    for (size_t layer = 0; layer < stages.layers.size(); ++layer) {
        const std::string index = std::to_string(layer), prefix = "blk." + index + ".attn";
        retained["layer_" + index] = stages.layers[layer];
        retained["attention_" + index] = matrices.at(prefix + "_out.weight")->src[1];
        if (packed) {
            const auto views = head_views(graph, matrices.at(prefix + "_qkv.weight"), heads);
            for (size_t part = 0; part < views.size(); ++part) {
                retained["qkv_" + index + "_" + std::to_string(part)] = ggml_cont(graph.ctx(), views[part]);
            }
        } else {
            int part = 0;
            for (const char * suffix : {"q", "k", "v"}) {
                retained["qkv_" + index + "_" + std::to_string(part++)] = matrices.at(prefix + "_" + suffix + ".weight");
            }
        }
    }
    for (const auto & item : retained) graph.output(item.second);
    graph.compute(backend, allocator);
    for (const auto & item : retained) result[item.first] = Graph::read(item.second);
    return result;
}

/** Reports numerical differences while keeping the paired check's exact-bit acceptance separate from official bounds. */
bool difference(const std::string & input, const std::string & name, const std::vector<float> & actual,
                const std::vector<float> & reference) {
    require(actual.size() == reference.size(), "paired tensor sizes differ: " + name);
    double error = 0, power = 0, largest = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]) && std::isfinite(reference[i]), "a paired tensor is not finite: " + name);
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
        const auto args = utf8_args(argc, argv);
        require(args.size() >= 3, "usage: encoder-qkv-check model.gguf cpu|gpu [audio_dump_directory...]");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[2]), 4), ggml_backend_free);
        setting("0");
        fastconformer::Recognizer recognizer(args[1], backend.get(), 4);
        const auto & model = recognizer.model();
        require(model.str("fastconformer.decoder.prediction") == "kda" && !model.boolean("fastconformer.encoder.use_bias") &&
                    model.str("fastconformer.encoder.attention") == "rel_pos", "the QKV experiment check requires Haas's unbiased global attention");
        refusal_contract(model, backend.get());
        setting("1");
        fastconformer::Encoder candidate(model, backend.get());
        graph_contract(model, recognizer.encoder(), candidate, backend.get());
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        require(bool(allocator), "cannot allocate the paired encoder graphs");
        bool ok = true;
        size_t inputs = 0;
        auto compare_input = [&](const std::string & input, const std::vector<float> & features, int64_t frames, bool decode) {
            const auto left = evaluate(recognizer.encoder(), recognizer.joint(), model, backend.get(), allocator.get(), features, frames, false);
            const auto right = evaluate(candidate, recognizer.joint(), model, backend.get(), allocator.get(), features, frames, true);
            for (const auto & item : left) ok = difference(input, item.first, right.at(item.first), item.second) && ok;
            ok = difference(input, "baseline_actual_vs_instrumented", left.at("actual_projected"), left.at("projected")) && ok;
            ok = difference(input, "candidate_actual_vs_instrumented", right.at("actual_projected"), right.at("projected")) && ok;
            if (decode) {
                const auto a = recognizer.decoding(left.at("actual_projected"), "greedy", fastconformer::DecodeOn::Cpu);
                const auto b = recognizer.decoding(right.at("actual_projected"), "greedy", fastconformer::DecodeOn::Cpu);
                const bool equal = a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs &&
                                   recognizer.detokenizer().text(a.ids) == recognizer.detokenizer().text(b.ids);
                ok = equal && ok;
                std::printf("{\"input\":%s,\"emitted_ids_frames_durations_text_graphs_equal\":%s}\n", json_string(input).c_str(), equal ? "true" : "false");
            }
            ++inputs;
            std::fflush(stdout);
        };
        const int64_t mels = model.size("fastconformer.frontend.n_mels");
        for (const int64_t encoded : {1, 2, 31, 32, 33, 170, 171, 172, 375}) {
            const int64_t frames = encoded * recognizer.encoder().subsampling_factor() - 1;
            std::vector<float> features((size_t) frames * mels);
            for (size_t i = 0; i < features.size(); ++i) features[i] = std::sin((float) i * 0.013f) + 0.1f * std::cos((float) i * 0.037f);
            require(recognizer.encoder().subsampled_frames(frames) == encoded, "a synthetic input has the wrong encoded length");
            compare_input("synthetic_" + std::to_string(encoded), features, frames, false);
        }
        for (size_t i = 3; i < args.size(); ++i) {
            const auto directory = std::filesystem::u8path(args[i]);
            const auto audio = read_npy((directory / "audio.npy").u8string()).f32;
            require(!audio.empty(), "benchmark audio is not a nonempty float32 array");
            compare_input(directory.filename().u8string(), recognizer.frontend().features(audio), recognizer.frontend().frames(audio.size()), true);
        }
        std::printf("{\"status\":\"%s\",\"inputs\":%zu,\"exact_bits_required\":true,\"official_reference\":false,\"blank_duration_trace_observed\":false}\n",
                    ok ? "passed" : "failed", inputs);
        return ok ? 0 : 1;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
