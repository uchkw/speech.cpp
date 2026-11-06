// Checks the encoder's QKV packing against its separate projections: source bytes, the actual graph and head
// strides, then exact F32 activation bits at projection, attention, layer and encoder outputs over boundary lengths.
// CPU, F16 and mixed-type controls retain separate projections. These paired inputs are not official reference dumps.
//
// usage: fastconformer-qkv-check <model.gguf> [gpu|cpu|device name]

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "fastconformer/encoder.h"
#include "fastconformer/layout.h"

namespace {

using Products = std::map<std::string, ggml_tensor *>;
using Snapshot = std::map<std::string, std::vector<float>>;

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

/**
 * Selects the separate path with a mixed descriptor in the last layer. Construction reads no weight bytes on this
 * path. The original descriptor is restored before a graph is built, so both encoders compute the same real weights.
 */
std::unique_ptr<fastconformer::Encoder> separate_encoder(const ModelFile & model, ggml_backend_t backend) {
    const std::string name = "blk." + std::to_string(model.count("fastconformer.encoder.num_layers") - 1) + ".attn";
    auto * key = model.tensor(name + "_k.weight");
    struct Restore {
        ggml_tensor * tensor;
        ggml_type type;
        ~Restore() { tensor->type = type; }
    } restore{key, key->type};
    key->type = model.tensor(name + "_q.weight")->type == GGML_TYPE_Q4_K ? GGML_TYPE_Q8_0 : GGML_TYPE_Q4_K;
    return std::make_unique<fastconformer::Encoder>(model, backend);
}

/** Whether the loaded model and backend belong to the automatic packing's checked scope. */
bool expects_packing(const ModelFile & model, ggml_backend_t backend) {
    const bool metal = std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend))), "MTL") == 0;
    bool expected = metal && model.str("fastconformer.decoder.prediction") == "kda" &&
                    model.str("fastconformer.encoder.attention") == "rel_pos" && !model.boolean("fastconformer.encoder.use_bias");
    for (int layer = 0; layer < model.count("fastconformer.encoder.num_layers"); ++layer) {
        const std::string name = "blk." + std::to_string(layer) + ".attn";
        const auto q = model.tensor(name + "_q.weight")->type;
        expected = expected && (q == GGML_TYPE_Q4_K || q == GGML_TYPE_Q8_0) &&
                   model.tensor(name + "_k.weight")->type == q && model.tensor(name + "_v.weight")->type == q;
    }
    return expected;
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
void head_consumers(const Graph & graph, const std::array<ggml_tensor *, 3> & views,
                    const ModelFile & model, const std::string & prefix) {
    bool bias_u = false, bias_v = false, k_scores = false, v_cont = false;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        const auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op == GGML_OP_ADD && node->src[0] == views[0]) {
            bias_u = bias_u || node->src[1] == model.tensor(prefix + "_pos_bias_u");
            bias_v = bias_v || node->src[1] == model.tensor(prefix + "_pos_bias_v");
        }
        if (node->op == GGML_OP_MUL_MAT && node->src[0]->op == GGML_OP_PERMUTE &&
            node->src[0]->src[0] == views[1]) k_scores = true;
        if (node->op == GGML_OP_CONT && node->src[0]->op == GGML_OP_PERMUTE &&
            node->src[0]->src[0] == views[2]) v_cont = true;
    }
    require(bias_u && bias_v && k_scores && v_cont, "the packed heads do not reach the original attention operations");
}

void graph_contract(const ModelFile & model, const fastconformer::Encoder & ordinary,
                    const fastconformer::Encoder & selected, ggml_backend_t backend, bool packed) {
    const int64_t frames = 264, mels = model.size("fastconformer.frontend.n_mels");
    const int64_t width = model.size("fastconformer.encoder.d_model"), heads = model.size("fastconformer.encoder.num_heads");
    const int layers = model.count("fastconformer.encoder.num_layers");
    const std::vector<float> features((size_t) frames * mels, 0.25f);
    Graph left, right;
    left.output(ordinary.build(left, features, frames));
    right.output(selected.build(right, features, frames));
    const auto names = matrix_names(model);
    const auto before = products(left, names), after = products(right, names);
    require(after.size() + (packed ? (size_t) (2 * layers) : 0) == before.size(), "the encoder selects the wrong number of products");
    std::set<std::string> replaced;
    size_t total_bytes = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".attn";
        for (const char * part : {"q", "k", "v"}) {
            const std::string name = prefix + "_" + part + ".weight";
            require(before.at(name)->src[0] == model.tensor(name), "the mixed-type control does not use separate projections");
            std::printf("%s: %s\n", name.c_str(), ggml_type_name(model.tensor(name)->type));
        }
        require(!before.count(prefix + "_qkv.weight"), "the mixed-type control packs a layer");
        if (!packed) {
            require(!after.count(prefix + "_qkv.weight"), "an excluded model or backend packs a layer");
            continue;
        }
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
        head_consumers(right, head_views(right, product, heads), model, prefix);
        std::printf("layer %d: packed %zu bytes, all source bytes equal\n", layer, offset);
    }
    for (const auto & item : before) {
        require(item.second->src[0] == model.tensor(item.first), "the separate path replaces an original weight");
        if (!replaced.count(item.first)) require(after.at(item.first)->src[0] == item.second->src[0], "another weight operand changes");
    }
    for (int i = 0; i < ggml_graph_n_nodes(right.cgraph()); ++i) {
        const auto * node = ggml_graph_node(right.cgraph(), i);
        require(ggml_backend_supports_op(backend, node), "the backend does not support an actual packed encoder node");
    }
    std::printf("graph: %d packed products, %zu retained weight bytes, other weights unchanged\n", packed ? layers : 0, total_bytes);
}

/** Runs the ordinary graph first, then one that retains every layer and attention result for comparison. */
Snapshot evaluate(const fastconformer::Encoder & encoder, const ModelFile & model,
                  ggml_backend_t backend, ggml_gallocr_t allocator, const std::vector<float> & features, int64_t frames,
                  bool packed) {
    Snapshot result;
    {
        Graph graph;
        auto * out = encoder.build(graph, features, frames);
        graph.output(out);
        graph.compute(backend, allocator);
        result["actual_encoded"] = Graph::read(out);
    }
    Graph graph;
    fastconformer::EncoderStages stages;
    auto * encoded = encoder.build(graph, features, frames, &stages);
    graph.output(encoded);
    const auto matrices = products(graph, matrix_names(model));
    std::map<std::string, ggml_tensor *> retained{{"subsampled", stages.subsampled}, {"encoded", encoded}};
    for (size_t layer = 0; layer < stages.layers.size(); ++layer) {
        const std::string index = std::to_string(layer), prefix = "blk." + index + ".attn";
        retained["layer_" + index] = stages.layers[layer];
        retained["attention_" + index] = matrices.at(prefix + "_out.weight")->src[1];
        if (packed) {
            // Retain the projection itself, as on the ordinary side. Extra graph copies can change Metal fusion.
            retained["packed_qkv_" + index] = matrices.at(prefix + "_qkv.weight");
        } else {
            int part = 0;
            for (const char * suffix : {"q", "k", "v"}) {
                retained["qkv_" + index + "_" + std::to_string(part++)] = matrices.at(prefix + "_" + suffix + ".weight");
            }
        }
    }
    for (const auto & item : retained) graph.output(item.second);
    graph.compute(backend, allocator);
    const size_t width = (size_t) model.size("fastconformer.encoder.d_model");
    for (const auto & item : retained) {
        const auto values = Graph::read(item.second);
        if (item.first.rfind("packed_qkv_", 0) != 0) {
            result[item.first] = values;
            continue;
        }
        const std::string index = item.first.substr(11);
        for (size_t part = 0; part < 3; ++part) {
            auto & head = result["qkv_" + index + "_" + std::to_string(part)];
            head.resize(values.size() / 3);
            for (size_t frame = 0; frame < head.size() / width; ++frame) {
                std::copy_n(values.begin() + (frame * 3 + part) * width, width, head.begin() + frame * width);
            }
        }
    }
    return result;
}

/** Reports numerical differences while keeping the paired check's exact-bit acceptance separate from official bounds. */
void difference(const std::string & input, const std::string & name, const std::vector<float> & actual,
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
    std::printf("%s %s: type=F32 elements=%zu equal_bits=%s nmse=%.12g max_abs=%.12g\n",
                input.c_str(), name.c_str(), actual.size(), exact ? "true" : "false", power > 0 ? error / power : error, largest);
    require(exact, "first difference at " + input + " " + name + " type=F32 elements=" + std::to_string(actual.size()));
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 2 || args.size() > 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
            start_device(find_device(args.size() > 2 ? args[2] : "gpu"), 4), ggml_backend_free);
        const ModelFile model(args[1], backend.get(), fastconformer::layout, [](const std::string & name) {
            return name.rfind("sub.", 0) == 0 || name.rfind("blk.", 0) == 0;
        });
        const auto ordinary = separate_encoder(model, backend.get());
        const fastconformer::Encoder selected(model, backend.get());
        const bool packed = expects_packing(model, backend.get());
        std::printf("backend: %s, expected path: %s, F32 activations with the file's original weight types\n",
                    ggml_backend_name(backend.get()), packed ? "packed" : "separate");
        graph_contract(model, *ordinary, selected, backend.get(), packed);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
        require(bool(allocator), "cannot allocate the paired encoder graphs");
        std::vector<std::string> dataflow{"subsampled"};
        for (int layer = 0; layer < model.count("fastconformer.encoder.num_layers"); ++layer) {
            const std::string index = std::to_string(layer);
            for (int part = 0; part < 3; ++part) dataflow.push_back("qkv_" + index + "_" + std::to_string(part));
            dataflow.push_back("attention_" + index);
            dataflow.push_back("layer_" + index);
        }
        for (const char * name : {"encoded", "actual_encoded"}) dataflow.push_back(name);
        const int64_t mels = model.size("fastconformer.frontend.n_mels");
        for (const int64_t frames : {1, 2, 31, 32, 33, 170, 171, 172, 375}) {
            const int64_t before_subsampling = frames * selected.subsampling_factor() - 1;
            std::vector<float> features((size_t) before_subsampling * mels);
            for (size_t i = 0; i < features.size(); ++i) features[i] = std::sin((float) i * 0.013f) + 0.1f * std::cos((float) i * 0.037f);
            require(selected.subsampled_frames(before_subsampling) == frames, "a synthetic input has the wrong encoded length");
            const auto left = evaluate(*ordinary, model, backend.get(), allocator.get(), features, before_subsampling, false);
            const auto right = evaluate(selected, model, backend.get(), allocator.get(), features, before_subsampling, packed);
            const std::string input = "frames=" + std::to_string(frames);
            for (const auto & name : dataflow) difference(input, name, right.at(name), left.at(name));
            difference(input, "separate_actual_vs_retained", left.at("actual_encoded"), left.at("encoded"));
            difference(input, "selected_actual_vs_retained", right.at("actual_encoded"), right.at("encoded"));
            std::fflush(stdout);
        }
        std::printf("passed: nine boundary lengths, exact activation bits; no official-reference comparison\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
