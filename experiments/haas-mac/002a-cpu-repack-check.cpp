// Compares Haas's selected CPU_REPACK products with the same Q8_0 weights in ordinary CPU buffers.
// Usage: cpu-repack-check model.gguf expected_repacked_count
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

#include "backend.h"
#include "fastconformer/layout.h"
#include "fastconformer/transducer.h"
#include "ggml-cpu.h"

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

bool same_bytes(const ggml_tensor * left, const ggml_tensor * right) {
    if (left->type != right->type || !ggml_are_same_shape(left, right) || std::memcmp(left->nb, right->nb, sizeof(left->nb))) return false;
    std::vector<uint8_t> a(ggml_nbytes(left)), b(ggml_nbytes(right));
    ggml_backend_tensor_get(left, a.data(), 0, a.size());
    ggml_backend_tensor_get(right, b.data(), 0, b.size());
    return a == b;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: cpu-repack-check model.gguf expected_repacked_count");
        const size_t expected = (size_t) std::stoi(argv[2]);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device("cpu"), 4), ggml_backend_free);
        ModelFile plain(argv[1], backend.get(), fastconformer::layout, fastconformer::decoding_tensor);
        setenv("SPEECH_HAAS_CPU_REPACK", "0", 1);
        require(fastconformer::decoding_buffer_types(plain, backend.get()).empty(), "Disabled repacking selected tensors");
        ModelFile ordinary(plain, backend.get(), fastconformer::decoding_tensor);
        setenv("SPEECH_HAAS_CPU_REPACK", "1", 1);
        const auto selected = fastconformer::decoding_buffer_types(plain, backend.get());
        require(selected.size() == expected, "The repacked tensor count differs from the requested expectation");
        ModelFile packed(plain, backend.get(), fastconformer::decoding_tensor, selected);
        size_t unchanged = 0, untouched = 0, repacked_bytes = 0;
        for (const TensorSpec & spec : fastconformer::layout.tensors(plain)) {
            if (!fastconformer::decoding_tensor(spec.name)) continue;
            require(same_bytes(plain.tensor(spec.name), ordinary.tensor(spec.name)), "An ordinary decoder copy differs byte for byte");
            ++unchanged;
            if (!selected.count(spec.name)) {
                require(same_bytes(plain.tensor(spec.name), packed.tensor(spec.name)), "An unselected tensor differs byte for byte");
                require(std::string(ggml_backend_buffer_name(packed.tensor(spec.name)->buffer)) != "CPU_REPACK", "An unselected tensor uses CPU_REPACK");
                ++untouched;
            }
        }
        double worst_nmse = 0, worst_abs = 0;
        for (const auto & item : selected) {
            const auto * reference = plain.tensor(item.first);
            auto * weight = packed.tensor(item.first);
            require(std::string(ggml_backend_buffer_name(weight->buffer)) == "CPU_REPACK", "Selected tensor lacks a CPU_REPACK buffer");
            require(weight->extra != nullptr, "Selected tensor lacks the CPU_REPACK trait");
            require(weight->type == reference->type && ggml_are_same_shape(weight, reference), "Packing changed the type or shape");
            repacked_bytes += ggml_nbytes(weight);
            double tensor_nmse = 0, tensor_abs = 0;
            for (int test = 0; test < 7; ++test) {
                std::vector<float> input((size_t) weight->ne[0]);
                const float scale = test == 0 ? 0 : test < 3 ? 0.001f : test < 5 ? 1.0f : 100.0f;
                for (size_t i = 0; i < input.size(); ++i) {
                    input[i] = scale * (std::sin((float) i * (0.031f + test * 0.007f)) + 0.3f * std::cos((float) i * 0.17f));
                }
                Graph graph(16);
                auto * x = graph.input(input, weight->ne[0]);
                auto * baseline = mul_mat(graph.ctx(), plain.tensor(item.first), x);
                auto * candidate = mul_mat(graph.ctx(), weight, x);
                require(ggml_backend_supports_op(backend.get(), candidate), "CPU_REPACK does not support the actual row-one product");
                graph.output(baseline);
                graph.output(candidate);
                std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
                    ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend.get())), ggml_gallocr_free);
                require(allocator != nullptr, "Cannot allocate the product graph");
                graph.compute(backend.get(), allocator.get());
                const auto a = Graph::read(baseline), b = Graph::read(candidate);
                double error = 0, power = 0, maximum = 0, largest = 0;
                for (size_t i = 0; i < a.size(); ++i) {
                    require(std::isfinite(a[i]) && std::isfinite(b[i]), "A product is not finite");
                    const double delta = (double) a[i] - b[i];
                    error += delta * delta;
                    power += (double) a[i] * a[i];
                    maximum = std::max(maximum, std::abs((double) a[i]));
                    largest = std::max(largest, std::abs(delta));
                }
                const double nmse = power > 0 ? error / power : error;
                require(nmse <= 1e-10 && largest <= 2e-5 * std::max(1.0, maximum), "Repacked multiplication exceeds the rounding bound");
                tensor_nmse = std::max(tensor_nmse, nmse);
                tensor_abs = std::max(tensor_abs, largest);
            }
            worst_nmse = std::max(worst_nmse, tensor_nmse);
            worst_abs = std::max(worst_abs, tensor_abs);
            std::printf("{\"name\":\"%s\",\"input\":%lld,\"output\":%lld,\"columns\":1,\"cases\":7,\"buffer\":\"CPU_REPACK\",\"trait\":true,\"supports_op\":true,\"nmse_max\":%.12g,\"absolute_difference_max\":%.12g}\n",
                        item.first.c_str(), (long long) weight->ne[0], (long long) weight->ne[1], tensor_nmse, tensor_abs);
        }
        std::printf("{\"status\":\"passed\",\"selected_count\":%zu,\"selected_bytes\":%zu,\"ordinary_copies_equal\":%zu,\"unselected_copies_equal\":%zu,\"neon\":%d,\"dotprod\":%d,\"i8mm\":%d,\"nmse_max\":%.12g,\"absolute_difference_max\":%.12g}\n",
                    selected.size(), repacked_bytes, unchanged, untouched, ggml_cpu_has_neon(), ggml_cpu_has_dotprod(), ggml_cpu_has_matmul_int8(), worst_nmse, worst_abs);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
