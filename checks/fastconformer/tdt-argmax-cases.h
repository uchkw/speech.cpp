#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fastconformer/decoder.h"
#include "fastconformer/tdt-argmax.h"

namespace argmax_check {

using namespace fastconformer;

struct Checks {
    size_t total = 0, failed = 0;
    void expect(bool ok, const std::string & what) {
        ++total;
        if (!ok) { ++failed; std::printf("FAIL: %s\n", what.c_str()); }
    }
    void equal(const std::vector<float> & a, const std::vector<float> & b, const std::string & what) {
        const bool finite = std::all_of(a.begin(), a.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(b.begin(), b.end(), [](float x) { return std::isfinite(x); });
        expect(finite && a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0), what);
    }
};

void setting(const char * value) {
#ifdef _WIN32
    const int status = _putenv_s("SPEECH_HAAS_TDT_DEVICE_ARGMAX", value ? value : "");
#else
    const int status = value ? setenv("SPEECH_HAAS_TDT_DEVICE_ARGMAX", value, 1) : unsetenv("SPEECH_HAAS_TDT_DEVICE_ARGMAX");
#endif
    if (status) throw std::runtime_error("cannot set the TDT argmax selector");
}

using Pattern = std::pair<std::string, std::vector<float>>;

/** Rejects shapes that would not describe one complete TDT decision before a backend sees them. */
void contract_cases(ggml_backend_t backend, Checks & checks) {
    auto refused = [&](Graph & graph, ggml_tensor * tensor, int blank, const char * name) {
        bool failed = false;
        try { const TdtArgmax argmax(graph, tensor, blank, backend); }
        catch (const std::logic_error &) { failed = true; }
        checks.expect(failed, name);
    };
    Graph graph(32);
    auto * row = graph.input(std::vector<float>(6), 6);
    refused(graph, row, 2, "unretained logits are rejected");
    graph.output(row);
    refused(graph, row, -1, "negative blank is rejected");
    refused(graph, row, 5, "empty duration range is rejected");
    refused(graph, row, 6, "blank beyond logits is rejected");
    auto * matrix = graph.input(std::vector<float>(6), 3, 2);
    graph.output(matrix);
    refused(graph, matrix, 1, "multiple joint rows are rejected");
    auto * half = graph.half_input(std::vector<float>(6), 6, 1);
    graph.output(half);
    refused(graph, half, 2, "non-F32 logits are rejected");
    bool failed = false;
    try { (void) TdtArgmax().read(); }
    catch (const std::logic_error &) { failed = true; }
    checks.expect(failed, "reading absent decisions is rejected");
}

std::vector<Pattern> patterns(int n, bool adversarial) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    constexpr float sub = std::numeric_limits<float>::denorm_min();
    constexpr float twice_sub = 2.0f * sub;
    std::vector<Pattern> out;
    auto add = [&](const char * name, float fill, const std::vector<std::pair<int, float>> & edits) {
        std::vector<float> v((size_t) n, fill);
        for (auto e : edits) if (e.first < n) v[(size_t) e.first] = e.second;
        out.push_back({name, std::move(v)});
    };
    if (!adversarial) {
        add("all-negative", -4.0f, {});
        add("first", -4.0f, {{0, 3.0f}});
        add("last", -4.0f, {{n - 1, 3.0f}});
        add("middle", -4.0f, {{n / 2, 3.0f}});
        add("tie-first-last", -4.0f, {{0, 3.0f}, {n - 1, 3.0f}});
        add("tie-lanes", -4.0f, {{n - 1, 3.0f}, {31, 3.0f}, {32, 3.0f}, {64, 3.0f}});
        add("signed-zero", -0.0f, {{n - 1, 0.0f}});
        add("adjacent-f32", 1.0f, {{n - 1, std::nextafter(1.0f, 2.0f)}});
        add("finite-limits", std::numeric_limits<float>::lowest(), {{n - 1, std::numeric_limits<float>::max()}});
    } else {
        add("first-nan", -4.0f, {{n - 1, 3.0f}, {0, nan}});
        add("later-nan", -4.0f, {{n - 1, nan}, {0, 2.0f}});
        add("interior-nan", -4.0f, {{n / 2, nan}, {n - 1, 3.0f}});
        add("all-nan", nan, {});
        add("all-negative-inf", -inf, {});
        add("negative-inf-nan", -inf, {{n - 1, nan}});
        add("positive-inf-tie", -inf, {{n / 2, inf}, {n - 1, inf}});
        add("positive-inf-nan", -4.0f, {{n - 1, nan}, {0, inf}});
        add("first-nan-positive-inf", -inf, {{n - 1, inf}, {0, nan}});
        add("zero-subnormal", 0.0f, {{n - 1, sub}});
        add("negative-subnormal-zero", -sub, {{n - 1, 0.0f}});
        add("subnormal-order", sub, {{n - 1, twice_sub}});
    }
    return out;
}

/** Keeps the full row only for comparison; the production reader obtains 16 bytes through four gets. */
void raw_cases(ggml_backend_t backend, bool adversarial, Checks & checks) {
    size_t cases = 0;
    for (int labels : {1, 5, 31, 32, 33, 63, 64, 65, 3073}) {
        constexpr int durations = 5;
        Graph graph(32);
        auto * logits = graph.input(std::vector<float>((size_t) labels + durations), labels + durations);
        graph.output(logits);
        const TdtArgmax pick(graph, logits, labels - 1, backend);
        std::vector<ggml_tensor *> reductions;
        int other = 0;
        auto * nodes = graph.cgraph();
        for (int i = 0; i < ggml_graph_n_nodes(nodes); ++i) {
            auto * node = ggml_graph_node(nodes, i);
            if (node->op == GGML_OP_ARGMAX) reductions.push_back(node);
            else if (node->op != GGML_OP_VIEW) ++other;
        }
        checks.expect(reductions.size() == 2 && other == 0 && (logits->flags & GGML_TENSOR_FLAG_OUTPUT),
                      "two ARGMAX operations and retained logits");
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        if (!allocator) throw std::runtime_error("cannot create the raw argmax allocator");
        graph.compute(backend, allocator.get());
        for (int range : {0, 1}) {
            for (const auto & pattern : patterns(range ? durations : labels, adversarial)) {
                std::vector<float> row((size_t) labels + durations, 1000000.0f);
                std::copy(pattern.second.begin(), pattern.second.end(), row.begin() + (range ? labels : 0));
                graph.set(logits, row);
                for (auto * reduction : reductions) {
                    const int32_t poison = -12345;
                    ggml_backend_tensor_set(reduction, &poison, 0, sizeof(poison));
                }
                graph.compute_again(backend);
                const auto got = pick.read();
                const int want_label = first_argmax(row.data(), labels);
                const int want_duration = first_argmax(row.data() + labels, durations);
                const std::string name = std::string(adversarial ? "adversarial " : "finite ") + pattern.first +
                                         " width=" + std::to_string(labels) + " range=" + std::to_string(range);
                checks.expect(got.label == want_label && got.duration_index == want_duration, name);
                if (got.label != want_label || got.duration_index != want_duration) {
                    std::printf("  got=(%d,%d), want=(%d,%d)\n", got.label, got.duration_index, want_label, want_duration);
                }
                const auto retained = Graph::read(logits);
                checks.expect(retained.size() == row.size() && std::memcmp(retained.data(), row.data(), row.size() * sizeof(float)) == 0,
                              name + " leaves source bits unchanged");
                ++cases;
            }
        }
    }
    std::printf("raw %s cases: %zu\n", adversarial ? "nonfinite/subnormal" : "finite", cases);
}

}  // namespace argmax_check
