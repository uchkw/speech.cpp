#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "compare.h"
#include "fastconformer/rnnt-greedy.h"
#include "fastconformer/tdt.h"
#include "constants-fixture.h"

namespace constant_check {

using namespace fastconformer;

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & what) {
        total++;
        if (!ok) {
            failed++;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    void equal(const std::vector<float> & got, const std::vector<float> & want, const std::string & what) {
        const bool finite = std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(want.begin(), want.end(), [](float x) { return std::isfinite(x); });
        const bool same = got.size() == want.size() && (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
        expect(finite && same, what);
        if (!finite || !same) print_diff(what, compare(got, want));
    }
};

std::vector<float> frames(int hidden, int count, int seed) {
    std::vector<float> out((size_t) hidden * count);
    for (size_t i = 0; i < out.size(); i++) out[i] = std::sin(0.031f * (float) (i + 37 * seed));
    return out;
}

/** Compares the complete token path and the exact points at which cancellation can be observed. */
Decoding decode_pair(const TdtDecoder & host, const TdtDecoder & cached, ggml_backend_t backend,
                     const std::vector<float> & projected, size_t allowed, Checks & checks, const Boost * boost = nullptr) {
    std::vector<double> a_progress, b_progress;
    auto a_callback = [&](double done) { a_progress.push_back(done); return a_progress.size() <= allowed; };
    auto b_callback = [&](double done) { b_progress.push_back(done); return b_progress.size() <= allowed; };
    const auto a = boost ? host.decode(projected, backend, a_callback, *boost) : host.decode(projected, backend, a_callback);
    const auto b = boost ? cached.decode(projected, backend, b_callback, *boost) : cached.decode(projected, backend, b_callback);
    checks.expect(a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs,
                  "TDT tokens, frames, durations and graph count");
    checks.expect(a_progress == b_progress, "TDT progress and cancellation points");
    return b;
}

void decoding_cases(const ModelFile & model, const PredictionNetwork & plain, const PredictionNetwork & prepared, const Joint & joint,
                    ggml_backend_t backend, bool synthetic, Checks & checks) {
    const TdtDecoder host(model, plain, joint), cached(model, prepared, joint);
    const size_t unlimited = std::numeric_limits<size_t>::max();
    decode_pair(host, cached, backend, {}, unlimited, checks);
    for (int count : {1, 7, 19}) {
        const auto projected = frames(joint.hidden(), count, count);
        decode_pair(host, cached, backend, projected, 0, checks);
        decode_pair(host, cached, backend, projected, 1, checks);
        decode_pair(host, cached, backend, projected, unlimited, checks);
        decode_pair(host, cached, backend, projected, unlimited, checks);
    }
    const BoostingTree tree({{1, 2}, {3, 4}}, host.blank(), 10.0, 1.0, 0.0f);
    const Boost boost{tree, 20.0f};
    decode_pair(host, cached, backend, frames(joint.hidden(), 7, 23), unlimited, checks, &boost);
    if (!synthetic) return;

    const int labels[] = {1, constant_check::kBlank, constant_check::kBlank, 9, 2, constant_check::kBlank};
    const int durations[] = {0, 0, 2, 1, 1, 4};
    std::vector<float> projected((size_t) 6 * joint.hidden(), 0.0f);
    for (size_t t = 0; t < 6; t++) {
        projected[t * joint.hidden() + labels[t]] = 100.0f;
        projected[t * joint.hidden() + constant_check::kBlank + 1 + durations[t]] = 50.0f;
    }
    const auto stopped = decode_pair(host, cached, backend, projected, 2, checks);
    checks.expect(stopped.ids == std::vector<int32_t>({1, 1}) && stopped.graphs == 2, "cancellation stops before the next prediction");
    const auto complete = decode_pair(host, cached, backend, projected, unlimited, checks);
    checks.expect(complete.ids == std::vector<int32_t>({1, 1, 2}) && complete.frames == std::vector<int64_t>({0, 0, 4}) &&
                  complete.durations == std::vector<int32_t>({0, 0, 1}) && complete.graphs == 6,
                  "TDT zero duration, blank chain, max symbols and final blank");
    decode_pair(host, cached, backend, projected, unlimited, checks, &boost);
    const auto repeated = decode_pair(host, cached, backend, projected, unlimited, checks);
    checks.expect(repeated.ids == complete.ids && repeated.frames == complete.frames, "request after boosted or cancelled decoding keeps the token path");

    std::vector<float> last((size_t) joint.hidden(), 0.0f);
    last[3] = 100.0f;
    last[constant_check::kBlank + 1 + 4] = 50.0f;
    const auto end = decode_pair(host, cached, backend, last, unlimited, checks);
    checks.expect(end.ids == std::vector<int32_t>({3}) && end.durations == std::vector<int32_t>({4}), "token whose duration passes the final frame");

    const int chain_labels[] = {constant_check::kBlank, constant_check::kBlank, 1, constant_check::kBlank, constant_check::kBlank,
                                2, constant_check::kBlank, constant_check::kBlank, 3, constant_check::kBlank};
    std::vector<float> chains((size_t) 10 * joint.hidden(), 0.0f);
    for (size_t t = 0; t < 10; t++) {
        chains[t * joint.hidden() + chain_labels[t]] = 100.0f;
        chains[t * joint.hidden() + constant_check::kBlank + 2] = 50.0f;
    }
    const auto cancelled_chain = decode_pair(host, cached, backend, chains, 2, checks);
    checks.expect(cancelled_chain.ids == std::vector<int32_t>({1, 2}) && cancelled_chain.frames == std::vector<int64_t>({2, 5}) &&
                  cancelled_chain.graphs == 6, "cancellation follows blank chains and multiple prediction steps");
    const auto multiple = decode_pair(host, cached, backend, chains, unlimited, checks);
    checks.expect(multiple.ids == std::vector<int32_t>({1, 2, 3}) && multiple.frames == std::vector<int64_t>({2, 5, 8}) &&
                  multiple.durations == std::vector<int32_t>({1, 1, 1}) && multiple.graphs == 10,
                  "separate blank chains preserve the joint after different prediction steps");

    const RnntGreedyDecoder rnnt(model, plain, joint), rnnt_cached(model, prepared, joint);
    const auto r = rnnt.decode(projected, backend, {});
    const auto c = rnnt_cached.decode(projected, backend, {});
    checks.expect(r.ids == c.ids && r.frames == c.frames && r.durations == c.durations && r.graphs == c.graphs, "RNN-T constant preparation preserves the path");
    checks.expect(r.ids == std::vector<int32_t>({1, 1, 9, 9, 2, 2}) &&
                  r.frames == std::vector<int64_t>({0, 0, 3, 3, 4, 4}) && r.durations.empty(), "RNN-T retains its host-state path");
}

}  // namespace constant_check
