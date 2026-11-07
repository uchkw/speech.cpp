#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "compare.h"
#include "error.h"
#include "graph.h"

/**
 * Computes `g` for a check: the graph's memory is allocated, the result regions `outputs` name are poisoned with
 * NaNs, and only then are the inputs uploaded and the graph computed. A value the compute leaves unwritten then
 * holds a NaN rather than an earlier graph's result on the same memory, so it cannot pass for this graph's own. The
 * inputs and the weights are not touched: the inputs are uploaded again by Graph::compute_again(), and the weights
 * live in the model file's own buffer. Each graph of a check gets this, and its own allocator, so that no graph's
 * compute_again() runs after another graph has reused its memory.
 */
inline void compute_poisoned(Graph & g, ggml_backend_t backend, ggml_gallocr_t allocr, const std::vector<ggml_tensor *> & outputs) {
    if (!ggml_gallocr_alloc_graph(allocr, g.cgraph())) {
        throw Error(Fault::OutOfMemory, "cannot allocate the memory of a check graph on the device");
    }
    for (ggml_tensor * t : outputs) {
        if (t->type != GGML_TYPE_F32) throw std::logic_error("a check's result is not float32");
        const std::vector<uint8_t> nan(ggml_nbytes(t), 0xFF);
        ggml_backend_tensor_set(t, nan.data(), 0, nan.size());
    }
    g.compute_again(backend);
}

/** Whether two float32 results hold the same bits element for element, which is the differential checks' pass condition. */
inline bool same_bits(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

/** The index of the first value of `v` that is not finite, or -1 when every one is. */
inline int64_t first_not_finite(const std::vector<float> & v) {
    for (size_t i = 0; i < v.size(); i++) {
        if (!std::isfinite(v[i])) return (int64_t) i;
    }
    return -1;
}

/**
 * Reports the comparison of one stage: `got` against `want`, by name `what`. Every value of both must be finite and
 * the two must hold the same bits, which a departure from either fails; the largest difference and the SNR are said
 * either way, as a measurement of the departure and not a tolerance. Says whether the comparison passed.
 */
inline bool stage_report(const std::string & what, const std::vector<float> & got, const std::vector<float> & want) {
    const int64_t bad_got = first_not_finite(got), bad_want = first_not_finite(want);
    if (got.size() != want.size()) {
        std::printf("FAIL %s: %zu and %zu values\n", what.c_str(), got.size(), want.size());
        return false;
    }
    const Diff d = compare(got, want);
    if (bad_got >= 0 || bad_want >= 0) {
        std::printf("FAIL %s: value %lld of %s is not finite\n", what.c_str(), (long long) (bad_got >= 0 ? bad_got : bad_want),
                    bad_got >= 0 ? "the batched result" : "the baseline");
        return false;
    }
    if (!same_bits(got, want)) {
        int64_t first = -1;
        for (size_t i = 0; i < got.size(); i++) {
            if (std::memcmp(&got[i], &want[i], sizeof(float)) != 0) {
                first = (int64_t) i;
                break;
            }
        }
        print_diff("FAIL " + what + " (first differing value " + std::to_string(first) + ")", d);
        return false;
    }
    print_diff("ok   " + what, d);
    return true;
}
