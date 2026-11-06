// Compares Haas position projections with 32 MiB, 72 MiB and no retained cache, then independently compares the
// ordinary final joint input and greedy TDT path. Only the projection diagnostic marks intermediate outputs.
//
// usage: fastconformer-position-cache-check <model.gguf> <cpu|gpu|device> positions
//        fastconformer-position-cache-check <model.gguf> <cpu|gpu|device> outputs <audio.npy> [audio.npy ...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "error.h"
#include "fastconformer/recognizer.h"
#include "npy.h"

using namespace fastconformer;

namespace {

void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

bool identical(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

/** Reports every numeric difference without replacing the exact comparison with an error threshold. */
bool compare_values(const std::string & name, const std::vector<float> & candidate, const std::vector<float> & baseline) {
    require(candidate.size() == baseline.size(), name + ": different shapes");
    const bool exact = finite(candidate) && finite(baseline) && identical(candidate, baseline);
    const Diff diff = compare(candidate, baseline);
    std::printf("%s exact=%s finite=%s values=%zu max_abs=%.9g snr_db=%.9g\n", name.c_str(), exact ? "true" : "false",
                finite(candidate) && finite(baseline) ? "true" : "false", candidate.size(), diff.max_abs, diff.snr_db);
    return exact;
}

void select_cache(const char * value) {
#ifdef _WIN32
    const int status = _putenv_s("SPEECH_HAAS_POSITION_CACHE", value ? value : "");
#else
    const int status = value ? setenv("SPEECH_HAAS_POSITION_CACHE", value, 1) : unsetenv("SPEECH_HAAS_POSITION_CACHE");
#endif
    require(status == 0, "cannot set SPEECH_HAAS_POSITION_CACHE for the check");
}

void check_selection(const ModelFile & model, ggml_backend_t backend) {
    for (const char * setting : {static_cast<const char *>(nullptr), "0", "1"}) {
        select_cache(setting);
        const Encoder encoder(model, backend);
        const int64_t expected = setting && std::string(setting) == "1" ? 383 : 170;
        require(encoder.cached_position_span() == expected, "the environment switch selected the wrong position span");
        require(encoder.position_cache_bytes() == 0, "constructing an encoder eagerly allocated its positions");
    }
    select_cache("invalid");
    bool refused = false;
    try {
        const Encoder encoder(model, backend);
    } catch (const Error & error) {
        refused = error.fault() == Fault::InvalidArgument;
    }
    select_cache(nullptr);
    require(refused, "the position-cache switch accepted a value other than 0 or 1");
    std::printf("selection: unset=32MiB 0=32MiB 1=72MiB invalid=refused constructor_cache_bytes=0\n");
}

struct Projection {
    std::vector<std::vector<float>> layers;
    std::vector<float> positions;
};

/** One encoder and allocator survive all lengths, as they do in one loaded recognizer. */
struct Run {
    const ModelFile & model;
    ggml_backend_t backend;
    const int budget;
    Encoder encoder;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator;
    std::vector<ggml_tensor *> position_weights;
    std::vector<const ggml_tensor *> cached_sources;
    size_t retained_bytes = 0;

    Run(const ModelFile & model, ggml_backend_t backend, int budget)
        : model(model), backend(backend), budget(budget), encoder(model, backend, budget),
          allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {
        require(allocator != nullptr, "cannot create a position-cache check allocator");
        require(encoder.position_cache_bytes() == 0, "an encoder starts with a nonempty position cache");
        for (int layer = 0; layer < 24; ++layer) position_weights.push_back(model.tensor("blk." + std::to_string(layer) + ".attn_pos.weight"));
    }

    void record(const char * phase, int64_t frames, const Graph & graph) {
        const int64_t t = encoder.subsampled_frames(frames), span = encoder.cached_position_span();
        const bool cached = t - 1 <= span;
        const size_t bytes = encoder.position_cache_bytes();
        const bool created = retained_bytes == 0 && bytes != 0;
        if (cached) {
            const size_t payload = (size_t) (2 * span + 1) * encoder.d_model() * sizeof(float) * 24;
            require(bytes >= payload, "the retained cache is smaller than its projection tensors");
        } else if (retained_bytes == 0) {
            require(bytes == 0, "a cache-ineligible input created a position cache");
        }
        if (retained_bytes != 0) require(bytes == retained_bytes, "the retained cache allocation changed across requests");
        retained_bytes = bytes;
        const int nodes = ggml_graph_n_nodes(graph.cgraph());
        int products = 0;
        for (int i = 0; i < nodes; ++i) {
            const auto * node = ggml_graph_node(graph.cgraph(), i);
            if (node->op == GGML_OP_MUL_MAT && std::find(position_weights.begin(), position_weights.end(), node->src[0]) != position_weights.end()) ++products;
        }
        require(products == (cached ? 0 : 24), "the utterance graph has the wrong number of position products");
        std::printf("%s budget_mib=%d feature_frames=%lld T=%lld cached=%s cache_created=%s nodes=%d position_products=%d retained_buffer_bytes=%zu\n",
                    phase, budget, (long long) frames, (long long) t, cached ? "true" : "false", created ? "true" : "false", nodes, products, bytes);
    }

    Projection project(int64_t t) {
        const int64_t frames = (t - 1) * encoder.subsampling_factor() + 1;
        const int width = encoder.d_model();
        Graph graph;
        EncoderStages stages;
        encoder.build(graph, std::vector<float>((size_t) frames * model.u32("fastconformer.frontend.n_mels"), 0.0f), frames, &stages);
        require(encoder.subsampled_frames(frames) == t && stages.positions.size() == 24, "unexpected projection shape or layer count");
        // Graph uploads every input, so the diagnostic also allocates the feature input through the subsampling.
        graph.output(stages.subsampled);
        const bool cached = t - 1 <= encoder.cached_position_span();
        if (cached && cached_sources.empty()) cached_sources.resize(stages.positions.size(), nullptr);
        ggml_tensor * raw = nullptr;
        for (size_t layer = 0; layer < stages.positions.size(); ++layer) {
            ggml_tensor * p = stages.positions[layer];
            require(p->type == GGML_TYPE_F32 && p->ne[0] == width && p->ne[1] == 2 * t - 1 && ggml_is_contiguous(p),
                    "a position projection has the wrong type, extent or stride");
            if (cached) {
                const int64_t span = encoder.cached_position_span();
                require(p->op == GGML_OP_VIEW && p->view_src && p->view_src->ne[0] == width && p->view_src->ne[1] == 2 * span + 1 &&
                        p->view_offs == (size_t) (span - (t - 1)) * p->nb[1], "a cached projection selects the wrong central positions");
                if (cached_sources[layer]) require(p->view_src == cached_sources[layer], "a layer's retained projection was replaced");
                cached_sources[layer] = p->view_src;
            } else {
                require(p->op == GGML_OP_MUL_MAT && p->src[0] == position_weights[layer],
                        "an uncached projection reads the wrong layer's weight");
                if (raw) require(p->src[1] == raw, "the layers do not share the same positional input");
                raw = p->src[1];
            }
            graph.output(p);
        }
        graph.compute(backend, allocator.get());
        Projection result;
        for (const auto * p : stages.positions) {
            result.layers.push_back(Graph::read(p));
            require(finite(result.layers.back()), "a position projection contains a nonfinite value");
        }
        if (raw) result.positions = Graph::read(raw);
        record("projections", frames, graph);
        return result;
    }

    std::vector<float> encode(const Joint & joint, const std::vector<float> & features, int64_t frames) {
        Graph graph;
        auto * out = joint.project_encoder(graph.ctx(), encoder.build(graph, features, frames));
        // The normal recognizer marks only this tensor. Intermediate diagnostics use a different graph.
        graph.output(out);
        graph.compute(backend, allocator.get());
        record("ordinary", frames, graph);
        auto values = Graph::read(out);
        require(finite(values), "the ordinary encoder output contains a nonfinite value");
        return values;
    }
};

bool compare_projections(const std::string & name, const Projection & a, const Projection & b) {
    require(a.layers.size() == b.layers.size(), "different projection layer counts");
    bool exact = true;
    for (size_t layer = 0; layer < a.layers.size(); ++layer) {
        exact = compare_values(name + " layer=" + std::to_string(layer), a.layers[layer], b.layers[layer]) && exact;
    }
    return exact;
}

bool check_positions(const ModelFile & model, ggml_backend_t backend) {
    Run original(model, backend, 32), extended(model, backend, 72), uncached(model, backend, 0);
    require(original.encoder.cached_position_span() == 170 && extended.encoder.cached_position_span() == 383 &&
            uncached.encoder.cached_position_span() == -1, "unexpected cache budgets for this Haas model");
    std::vector<float> widest_positions;
    Projection short_original, short_extended;
    bool ok = true;
    for (const int64_t t : {385, 1, 170, 171, 172, 375, 383, 384, 385, 171, 1, 384, 1}) {
        const Projection a = original.project(t), b = extended.project(t), zero = uncached.project(t);
        const std::string label = "T=" + std::to_string(t);
        if (widest_positions.empty()) widest_positions = zero.positions;
        const size_t begin = (size_t) (385 - t) * model.u32("fastconformer.encoder.d_model");
        const size_t count = (size_t) (2 * t - 1) * model.u32("fastconformer.encoder.d_model");
        require(begin + count <= widest_positions.size(), "the positional reference slice is outside its input");
        const std::vector<float> slice(widest_positions.begin() + begin, widest_positions.begin() + begin + count);
        ok = compare_values("position-input " + label, zero.positions, slice) && ok;
        if (!a.positions.empty()) ok = compare_values("original-position-input " + label, a.positions, slice) && ok;
        if (!b.positions.empty()) ok = compare_values("extended-position-input " + label, b.positions, slice) && ok;
        ok = compare_projections("gate 72-vs-32 " + label, b, a) && ok;
        // Changing the GEMM width can change rounding even for the existing cache. These are diagnostics, not passes.
        compare_projections("diagnostic 32-vs-0 " + label, a, zero);
        compare_projections("diagnostic 72-vs-0 " + label, b, zero);
        if (t == 1) {
            if (short_original.layers.empty()) {
                short_original = a;
                short_extended = b;
            } else {
                ok = compare_projections("gate original-long-to-short", a, short_original) && ok;
                ok = compare_projections("gate extended-long-to-short", b, short_extended) && ok;
            }
        }
    }
    return ok;
}

bool same_path(const Decoding & a, const Decoding & b) {
    return a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs;
}

bool check_outputs(Recognizer & recognizer, ggml_backend_t backend, std::vector<std::string> paths) {
    Run original(recognizer.model(), backend, 32), extended(recognizer.model(), backend, 72);
    std::vector<float> first_original, first_extended;
    Decoding first_path_original, first_path_extended;
    paths.push_back(paths.front());
    bool ok = true;
    for (const int64_t frames : {3073, 3000}) {
        std::vector<float> features((size_t) frames * recognizer.frontend().mels());
        for (size_t i = 0; i < features.size(); ++i) features[i] = 0.2f * std::sin((float) (i % 997) * 0.031f);
        std::printf("synthetic-feature-boundary frames=%lld T=%lld not_a_quality_sample=true\n", (long long) frames,
                    (long long) original.encoder.subsampled_frames(frames));
        const auto a = original.encode(recognizer.joint(), features, frames);
        const auto b = extended.encode(recognizer.joint(), features, frames);
        ok = compare_values("gate boundary-final-joint-input", b, a) && ok;
        const auto da = recognizer.decoding(a), db = recognizer.decoding(b);
        const bool path_exact = same_path(da, db);
        std::printf("gate boundary-emitted-path exact=%s\n", path_exact ? "true" : "false");
        ok = path_exact && ok;
    }
    for (size_t i = 0; i < paths.size(); ++i) {
        const Npy audio = read_npy(paths[i]);
        require(audio.shape.size() == 1 && !audio.f32.empty() && finite(audio.f32), "expected finite mono float32 audio in " + paths[i]);
        const int64_t frames = recognizer.frontend().frames(audio.f32.size());
        require(frames > 0, "audio is shorter than one frontend hop: " + paths[i]);
        const auto features = recognizer.frontend().features(audio.f32);
        std::printf("input=%s samples=%zu seconds=%.9g\n", paths[i].c_str(), audio.f32.size(),
                    (double) audio.f32.size() / recognizer.sample_rate());
        const auto a = original.encode(recognizer.joint(), features, frames);
        const auto b = extended.encode(recognizer.joint(), features, frames);
        ok = compare_values("gate ordinary-final-joint-input", b, a) && ok;
        const auto da = recognizer.decoding(a), db = recognizer.decoding(b);
        const std::string ta = recognizer.detokenizer().text(da.ids), tb = recognizer.detokenizer().text(db.ids);
        const bool path_exact = same_path(da, db) && ta == tb;
        std::printf("gate emitted-path exact=%s original_tokens=%zu extended_tokens=%zu original_graphs=%zu extended_graphs=%zu\n",
                    path_exact ? "true" : "false", da.ids.size(), db.ids.size(), da.graphs, db.graphs);
        std::printf("original_text=%s\nextended_text=%s\n", ta.c_str(), tb.c_str());
        ok = path_exact && ok;
        if (i == 0) {
            first_original = a;
            first_extended = b;
            first_path_original = da;
            first_path_extended = db;
        } else if (i + 1 == paths.size()) {
            ok = compare_values("gate original-return-to-first", a, first_original) && ok;
            ok = compare_values("gate extended-return-to-first", b, first_extended) && ok;
            ok = same_path(da, first_path_original) && same_path(db, first_path_extended) && ok;
        }
    }
    return ok;
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    if (args.size() < 4 || (args[3] != "positions" && args[3] != "outputs") ||
        (args[3] == "positions" && args.size() != 4) || (args[3] == "outputs" && args.size() < 5)) {
        std::fprintf(stderr, "usage: %s <model.gguf> <cpu|gpu|device> positions|outputs [audio.npy ...]\n", args[0].c_str());
        return 2;
    }
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[2]), 4), ggml_backend_free);
        Recognizer recognizer(args[1], backend.get(), 4);
        const auto & model = recognizer.model();
        require(model.str("general.name") == "Haas-ja-v1.0" && model.str("fastconformer.decoder.prediction") == "kda" &&
                model.str("fastconformer.encoder.attention") == "rel_pos" && model.u32("fastconformer.encoder.num_layers") == 24 &&
                model.u32("fastconformer.encoder.d_model") == 1024 && recognizer.encoder().subsampling_factor() == 8,
                "this check requires Haas-ja-v1.0 with 24 global-attention layers of width 1024 and subsampling by 8");
        std::printf("backend=%s threads=4 mode=%s no_official_reference=true\n", ggml_backend_name(backend.get()), args[3].c_str());
        check_selection(model, backend.get());
        const int64_t frames = recognizer.frontend().frames((size_t) 30 * recognizer.sample_rate());
        require(frames == 3000 && recognizer.encoder().subsampled_frames(frames) == 375, "the 30-second input does not fit the expected cache span");
        std::printf("30-second sample count: feature_frames=%lld T=%lld\n", (long long) frames,
                    (long long) recognizer.encoder().subsampled_frames(frames));
        const bool ok = args[3] == "positions" ? check_positions(model, backend.get()) :
            check_outputs(recognizer, backend.get(), std::vector<std::string>(args.begin() + 4, args.end()));
        std::printf("%s exact gate: %s\n", args[3].c_str(), ok ? "pass" : "fail");
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
