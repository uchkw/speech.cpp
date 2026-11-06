// Checks the actual encoder graph's F16 FFN operands against the original quantized GGUF values.
// Usage: ffn-f16-check model.gguf expected_expanded_count
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend.h"
#include "fastconformer/encoder.h"
#include "fastconformer/layout.h"

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

/** The original named weight operands, excluding products of intermediate activations. */
std::map<std::string, ggml_tensor *> matrices(const Graph & graph, const std::set<std::string> & names) {
    std::map<std::string, ggml_tensor *> result;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        const auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op != GGML_OP_MUL_MAT) continue;
        auto * weight = node->src[0];
        const std::string name = ggml_get_name(weight);
        if (names.count(name)) require(result.emplace(name, weight).second, "A weight is multiplied more than once in the graph");
    }
    return result;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: ffn-f16-check model.gguf expected_expanded_count");
        const size_t expected = (size_t) std::stoi(argv[2]);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device("gpu"), 4), ggml_backend_free);
        ModelFile model(argv[1], backend.get(), fastconformer::layout);
        std::set<std::string> names, ffn;
        for (const TensorSpec & spec : fastconformer::layout.tensors(model)) names.insert(spec.name);
        const int layers = model.count("fastconformer.encoder.num_layers");
        for (int layer = 0; layer < layers; ++layer) {
            for (const char * suffix : {"ff1_up", "ff1_down", "ff2_up", "ff2_down"}) {
                ffn.insert("blk." + std::to_string(layer) + "." + suffix + ".weight");
            }
        }
        setenv("SPEECH_HAAS_FFN_F16", "0", 1);
        fastconformer::Encoder ordinary(model, backend.get());
        setenv("SPEECH_HAAS_FFN_F16", "1", 1);
        fastconformer::Encoder expanded(model, backend.get());
        const int64_t frames = 64;
        const int mels = model.size("fastconformer.frontend.n_mels");
        std::vector<float> features((size_t) frames * mels);
        for (size_t i = 0; i < features.size(); ++i) features[i] = std::sin((float) i * 0.01f);
        Graph plain_graph, expanded_graph;
        plain_graph.output(ordinary.build(plain_graph, features, frames));
        expanded_graph.output(expanded.build(expanded_graph, features, frames));
        const auto plain = matrices(plain_graph, names), candidate = matrices(expanded_graph, names);
        require(plain.size() == candidate.size(), "The graph's named weight operands differ");
        size_t changed = 0, unchanged = 0, ffn_seen = 0, source_bytes = 0, retained_bytes = 0, rounded_values = 0;
        double squared_difference = 0, squared_reference = 0, largest_difference = 0;
        for (const auto & item : plain) {
            const std::string & name = item.first;
            const auto * source = model.tensor(name);
            const auto * weight = candidate.at(name);
            require(item.second == source, "The disabled experiment does not use the original tensor");
            if (ffn.count(name)) ++ffn_seen;
            if (!ffn.count(name) || !ggml_is_quantized(source->type)) {
                require(weight == source, "An unselected tensor is replaced");
                ++unchanged;
                continue;
            }
            require(weight != source && weight->type == GGML_TYPE_F16 && ggml_are_same_shape(weight, source),
                    "An expanded FFN tensor has the wrong identity, type or shape");
            const size_t count = (size_t) ggml_nelements(source);
            std::vector<uint8_t> quantized(ggml_nbytes(source));
            std::vector<float> reference(count);
            std::vector<ggml_fp16_t> actual(count);
            ggml_backend_tensor_get(source, quantized.data(), 0, quantized.size());
            ggml_backend_tensor_get(weight, actual.data(), 0, actual.size() * sizeof(ggml_fp16_t));
            ggml_get_type_traits(source->type)->to_float(quantized.data(), reference.data(), (int64_t) count);
            for (size_t i = 0; i < count; ++i) {
                require(actual[i] == ggml_fp32_to_fp16(reference[i]), "An uploaded FFN value differs from its scalar F16 conversion");
                const float value = ggml_fp16_to_fp32(actual[i]);
                require(std::isfinite(value), "An expanded FFN value is not finite");
                const double delta = (double) value - reference[i];
                squared_difference += delta * delta;
                squared_reference += (double) reference[i] * reference[i];
                largest_difference = std::max(largest_difference, std::abs(delta));
                rounded_values += value != reference[i];
            }
            ++changed;
            source_bytes += ggml_nbytes(source);
            retained_bytes += ggml_nbytes(weight);
            std::printf("{\"name\":\"%s\",\"source_type\":\"%s\",\"input\":%lld,\"output\":%lld,\"expanded_type\":\"f16\",\"converted_values_exact\":%zu}\n",
                        name.c_str(), ggml_type_name(source->type), (long long) weight->ne[0], (long long) weight->ne[1], count);
        }
        require(ffn_seen == ffn.size() && changed == expected, "The observed FFN or expanded matrix count differs");
        size_t supported = 0;
        for (int i = 0; i < ggml_graph_n_nodes(expanded_graph.cgraph()); ++i) {
            const auto * node = ggml_graph_node(expanded_graph.cgraph(), i);
            if (node->op == GGML_OP_MUL_MAT && ffn.count(ggml_get_name(node->src[0]))) {
                require(ggml_backend_supports_op(backend.get(), node), "The backend does not support an actual FFN product");
                ++supported;
            }
        }
        require(supported == ffn.size(), "The actual graph does not contain every supported FFN product");
        const double nmse = squared_reference > 0 ? squared_difference / squared_reference : squared_difference;
        std::printf("{\"status\":\"passed\",\"expanded_count\":%zu,\"ffn_products\":%zu,\"unselected_named_products\":%zu,\"source_bytes\":%zu,\"retained_f16_bytes\":%zu,\"rounded_values\":%zu,\"rounding_nmse\":%.12g,\"rounding_absolute_difference_max\":%.12g}\n",
                    changed, supported, unchanged, source_bytes, retained_bytes, rounded_values, nmse, largest_difference);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
