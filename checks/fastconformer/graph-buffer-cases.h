#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#include "tdt-buffer-cases.h"

namespace buffer_check {

/** The exception category and message, so a rejected pointer input is checked against the vector interface. */
std::pair<std::string, std::string> rejection(const std::function<void()> & run) {
    try {
        run();
    } catch (const std::logic_error & e) {
        return {"logic_error", e.what()};
    } catch (const std::runtime_error & e) {
        return {"runtime_error", e.what()};
    }
    return {};
}

/** Checks that both float interfaces own their data before allocation, including updates and F16 conversion. */
void graph_cases(ggml_backend_t backend, Checks & checks) {
    const std::vector<float> first{1.25f, -2.5f, 0.1f, 4.125f};
    const std::vector<float> first_half{1.25f, -2.5f, 0.0999755859375f, 4.125f};
    const std::vector<float> second{3.25f, 0.0625f, -5.5f, 8.0f};
    Graph graph(64);
    std::vector<float> vector_source = first, pointer_source = first;
    ggml_tensor * inputs[] = {
        graph.input(vector_source, 2, 2),
        graph.input(pointer_source.data(), pointer_source.size(), 2, 2),
        graph.half_input(vector_source, 2, 2),
        graph.half_input(std::vector<float>(4), 2, 2),
    };
    graph.set(inputs[3], pointer_source.data(), pointer_source.size());
    ggml_tensor * results[4];
    for (int i = 0; i < 4; i++) {
        results[i] = ggml_cast(graph.ctx(), inputs[i], GGML_TYPE_F32);
        graph.output(results[i]);
    }
    std::fill(vector_source.begin(), vector_source.end(), 99.0f);
    std::fill(pointer_source.begin(), pointer_source.end(), -99.0f);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    if (!allocator) throw std::runtime_error("cannot create the graph-input allocator");
    graph.compute(backend, allocator.get());
    for (int i = 0; i < 4; i++) checks.equal(Graph::read(results[i]), i < 2 ? first : first_half, "input owns its copied data");

    for (int i = 0; i < 4; i++) {
        std::vector<float> temporary = second;
        if (i % 2) graph.set(inputs[i], temporary.data(), temporary.size());
        else graph.set(inputs[i], temporary);
        std::fill(temporary.begin(), temporary.end(), -999.0f);
    }
    graph.compute_again(backend);
    for (int i = 0; i < 4; i++) checks.equal(Graph::read(results[i]), second, "set owns its copied data after the source expires");

    Graph invalid(64), foreign(64);
    auto * f32 = invalid.input(first, 4);
    auto * f16 = invalid.half_input(first, 4, 1);
    auto * integer = invalid.input(std::vector<int32_t>(4), 4);
    auto * other = foreign.input(first, 4);
    const std::vector<float> short_data(first.begin(), first.end() - 1);
    const auto vector_shape = rejection([&] { invalid.input(short_data, 4); });
    const auto pointer_shape = rejection([&] { invalid.input(first.data(), first.size() - 1, 4); });
    checks.expect(vector_shape == pointer_shape && vector_shape.first == "runtime_error", "input rejects the same incorrect count");
    for (ggml_tensor * input : {f32, f16}) {
        const auto a = rejection([&] { invalid.set(input, short_data); });
        const auto b = rejection([&] { invalid.set(input, first.data(), first.size() - 1); });
        checks.expect(a == b && a.first == "runtime_error", "set rejects the same incorrect count");
    }
    for (ggml_tensor * input : {integer, other}) {
        const auto a = rejection([&] { invalid.set(input, first); });
        const auto b = rejection([&] { invalid.set(input, first.data(), first.size()); });
        checks.expect(a == b && a.first == "logic_error", "set rejects the same incorrect type or foreign input");
    }
}

}  // namespace buffer_check
