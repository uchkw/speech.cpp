// Compares packed and separate KDA Q/K/V projections, their bytes and operands, and recurrent decoder steps.
// The synthetic F32 fixture and paired model runs are differential checks, not official NeMo or FLA references.
// Usage: fastconformer-kda-qkv-check <model.gguf> [cpu|gpu|device name] [threads]
//        fastconformer-kda-qkv-check --synthetic <work dir> [cpu|gpu|device name] [threads]

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
#include "compare.h"
#include "error.h"
#include "fastconformer/decoder.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "ggml-cpu.h"
#include "state-fixture.h"

using namespace fastconformer;

namespace {

using Products = std::map<std::string, ggml_tensor *>;
using Snapshot = std::map<std::string, std::vector<float>>;

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

void setting(const char * value) {
#ifdef _WIN32
    require(_putenv_s("SPEECH_HAAS_KDA_QKV", value) == 0, "cannot set the KDA QKV switch");
#else
    require(setenv("SPEECH_HAAS_KDA_QKV", value, 1) == 0, "cannot set the KDA QKV switch");
#endif
}

struct Checks {
    size_t total = 0, failed = 0, logits_differ = 0, argmax_differ = 0;

    void expect(bool ok, const std::string & what) {
        total++;
        if (!ok) {
            failed++;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    bool equal(const std::vector<float> & got, const std::vector<float> & want, const std::string & what) {
        const bool finite = std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(want.begin(), want.end(), [](float x) { return std::isfinite(x); });
        const bool exact = finite && got.size() == want.size() &&
                           (got.empty() || std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0);
        expect(exact, what);
        if (!exact) print_diff(what, compare(got, want));
        return exact;
    }
};

std::vector<float> frame(int hidden, int seed) {
    std::vector<float> result((size_t) hidden);
    for (size_t i = 0; i < result.size(); ++i) result[i] = std::sin(0.031f * (float) (i + 37 * seed));
    return result;
}

std::vector<uint8_t> bytes(const ggml_tensor * tensor) {
    std::vector<uint8_t> result(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size());
    return result;
}

/** The same three retained outputs as the actual DecoderStep graph. */
Snapshot evaluate(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend,
                  const ModelFile & model, bool instrumented, bool packed) {
    Graph graph(512);
    const auto step = prediction.build(graph, 1, prediction.initial_state());
    auto * projected = joint.project_prediction(graph.ctx(), step.output);
    auto * logits = joint.build(graph.ctx(), graph.input(frame(joint.hidden(), 7), joint.hidden()), projected);
    graph.output(logits);
    graph.output(projected);
    graph.output(step.state);
    std::map<std::string, ggml_tensor *> retained{{"state", step.state}, {"prediction", projected}, {"logits", logits}};
    if (instrumented) {
        retained["prediction_network"] = step.output;
        const int layers = model.count("fastconformer.decoder.prediction_layers");
        for (int layer = 0; layer < layers; ++layer) {
            const std::string prefix = "pred.kda." + std::to_string(layer) + ".";
            for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
                auto * node = ggml_graph_node(graph.cgraph(), i);
                if (node->op != GGML_OP_MUL_MAT) continue;
                const std::string name = ggml_get_name(node->src[0]);
                if (packed && name == prefix + "qkv.weight") {
                    const int64_t width = node->ne[0] / 3;
                    for (int part = 0; part < 3; ++part) {
                        retained[prefix + std::to_string(part)] = ggml_view_1d(graph.ctx(), node, width, (size_t) part * width * sizeof(float));
                    }
                } else if (!packed) {
                    int part = 0;
                    for (const char * label : {"q", "k", "v"}) {
                        if (name == prefix + label + ".weight") retained[prefix + std::to_string(part)] = node;
                        ++part;
                    }
                }
            }
        }
        require(retained.size() == (size_t) (4 + 3 * layers), "the instrumented graph misses a QKV product");
        for (const auto & item : retained) graph.output(item.second);
    }
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(bool(allocator), "cannot allocate the prediction-check graph");
    graph.compute(backend, allocator.get());
    Snapshot result;
    for (const auto & item : retained) result[item.first] = Graph::read(item.second);
    return result;
}

/** Counts every named model operand, and the products of model and packed weights. */
std::map<std::string, size_t> operands(const Graph & graph, const ModelFile & model, const std::set<std::string> & names,
                                     Products & products) {
    std::map<std::string, size_t> result;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        auto * node = ggml_graph_node(graph.cgraph(), i);
        for (auto * src : node->src) {
            if (!src) continue;
            const std::string name = ggml_get_name(src);
            if (names.count(name)) {
                require(src == model.tensor(name), "a graph replaces an original model operand: " + name);
                result[name]++;
            }
        }
        if (node->op == GGML_OP_MUL_MAT) {
            const std::string name = ggml_get_name(node->src[0]);
            require(products.emplace(name, node).second, "a weight is multiplied twice: " + name);
        }
    }
    return result;
}

void graph_contract(const ModelFile & model, const Layout & layout, const PredictionNetwork & ordinary,
                    const PredictionNetwork & packed, ggml_backend_t backend) {
    Graph left(512), right(512);
    const auto a = ordinary.build(left, 1, ordinary.initial_state()), b = packed.build(right, 1, packed.initial_state());
    left.output(a.output); left.output(a.state);
    right.output(b.output); right.output(b.state);
    std::set<std::string> names;
    for (const auto & spec : layout.tensors(model)) if (decoding_tensor(spec.name)) names.insert(spec.name);
    Products before, after;
    auto reads = operands(left, model, names, before);
    const auto packed_reads = operands(right, model, names, after);
    const int layers = model.count("fastconformer.decoder.prediction_layers");
    require(after.size() + 2 * (size_t) layers == before.size(), "three QKV products do not become one per KDA layer");
    size_t retained_bytes = 0;
    for (int layer = 0; layer < layers; ++layer) {
        const std::string prefix = "pred.kda." + std::to_string(layer) + ".";
        auto * product = after.at(prefix + "qkv.weight");
        const auto * weight = product->src[0];
        const auto * q = model.tensor(prefix + "q.weight");
        require(weight->type == q->type && weight->ne[0] == q->ne[0] && weight->ne[1] == 3 * q->ne[1] &&
                    ggml_is_matrix(weight) && ggml_is_contiguous(weight), "the packed weight changes its type or row layout");
        require(ggml_backend_buffer_get_type(weight->buffer) == ggml_backend_buffer_get_type(q->buffer) &&
                    ggml_backend_buffer_get_usage(weight->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS,
                "the packed weights change their buffer type or usage");
        size_t offset = 0;
        for (const char * part : {"q", "k", "v"}) {
            const std::string name = prefix + part + ".weight";
            const auto original = bytes(model.tensor(name));
            std::vector<uint8_t> actual(original.size());
            ggml_backend_tensor_get(weight, actual.data(), offset, actual.size());
            require(actual == original && before.count(name) && !after.count(name), "packed bytes or selected products differ: " + name);
            offset += actual.size();
            reads.erase(name);
        }
        require(offset == ggml_nbytes(weight), "the packed matrix does not contain exactly three original matrices");
        retained_bytes += offset;
        std::array<bool, 3> views{};
        for (int i = 0; i < ggml_graph_n_nodes(right.cgraph()); ++i) {
            const auto * node = ggml_graph_node(right.cgraph(), i);
            if (node->op != GGML_OP_VIEW || node->view_src != product) continue;
            const size_t width_bytes = (size_t) q->ne[1] * sizeof(float);
            require(node->view_offs % width_bytes == 0 && node->view_offs / width_bytes < 3 &&
                        ggml_nelements(node) == q->ne[1] && ggml_is_contiguous(node), "a production QKV view has the wrong range");
            const size_t part = node->view_offs / width_bytes;
            require(!views[part], "a production QKV view is duplicated");
            views[part] = true;
        }
        require(std::all_of(views.begin(), views.end(), [](bool value) { return value; }), "a production QKV view is unused");
        std::printf("packed layer %d: %s [%lld,%lld], %zu unchanged bytes\n", layer, ggml_type_name(weight->type),
                    (long long) weight->ne[0], (long long) weight->ne[1], offset);
    }
    require(reads == packed_reads, "the graph changes an unselected model operand");
    for (int i = 0; i < ggml_graph_n_nodes(right.cgraph()); ++i) {
        require(ggml_backend_supports_op(backend, ggml_graph_node(right.cgraph(), i)), "the backend refuses a packed prediction node");
    }
    std::printf("graph contract: %d packed products, %d removed products, %zu additional weight bytes\n", layers, 2 * layers, retained_bytes);
}

void refusal_contract(const ModelFile & model) {
    auto refused = [&] {
        bool rejected = false;
        try { PredictionNetwork invalid(model); }
        catch (const Error & error) { rejected = error.fault() == Fault::InvalidArgument; }
        require(rejected, "an invalid KDA QKV setting or source layout is not refused");
    };
    setting("invalid");
    refused();
    auto * key = model.tensor("pred.kda.0.k.weight");
    const ggml_tensor saved = *key;
    for (int variant = 0; variant < 3; ++variant) {
        try {
            if (variant == 0) key->type = saved.type == GGML_TYPE_F32 ? GGML_TYPE_F16 : GGML_TYPE_F32;
            if (variant == 1) key->ne[1]--;
            if (variant == 2) key->nb[1] += sizeof(float);
            setting("0");
            const PredictionNetwork disabled(model);
            setting("1");
            refused();
        } catch (...) { *key = saved; throw; }
        *key = saved;
    }
    std::printf("refusal contract: invalid switch, mixed type, shape and stride refused; original descriptor restored\n");
}

struct Steps {
    DecoderStep ordinary, packed;
    PredictionState a_state, b_state;
    std::vector<float> last_prediction, last_logits;
    int hidden, blank, durations;
    size_t changed = 0;

    Steps(const ModelFile & model, const PredictionNetwork & a, const PredictionNetwork & b, const Joint & joint, ggml_backend_t backend)
        : ordinary(a, joint, backend, 1), packed(b, joint, backend, 1), a_state(a.initial_state()), b_state(b.initial_state()),
          hidden(joint.hidden()), blank((int) model.u32("fastconformer.decoder.blank_id")),
          durations((int) model.i32_array("fastconformer.decoder.tdt.durations").size()) {}

    void advance(int32_t label, int seed, Checks & checks) {
        const auto projected = frame(hidden, seed);
        ordinary.compute(label, a_state, projected);
        packed.compute(label, b_state, projected);
        const auto previous = a_state;
        a_state = ordinary.state(); b_state = packed.state();
        checks.equal(b_state, a_state, "all recurrent and convolution state values");
        last_prediction = packed.prediction();
        checks.equal(last_prediction, ordinary.prediction(), "prediction after an independent state update");
        std::vector<float> expected;
        ordinary.logits(expected); packed.logits(last_logits);
        checks.logits_differ += !checks.equal(last_logits, expected, "joint logits after an independent state update");
        require(expected.size() == (size_t) (blank + 1 + durations) && last_logits.size() == expected.size(), "joint output width differs");
        const bool finite = std::all_of(last_logits.begin(), last_logits.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(expected.begin(), expected.end(), [](float x) { return std::isfinite(x); });
        const bool decisions = finite && first_argmax(last_logits.data(), blank + 1) == first_argmax(expected.data(), blank + 1) &&
            first_argmax(last_logits.data() + blank + 1, durations) == first_argmax(expected.data() + blank + 1, durations);
        checks.argmax_differ += !decisions;
        checks.expect(decisions, "joint label and duration argmax");
        changed += previous != a_state;
    }
};

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        const bool synthetic = args.size() > 1 && args[1] == "--synthetic";
        const size_t device_at = synthetic ? 3 : 2;
        require(args.size() >= device_at && args.size() <= device_at + 2,
                "usage: fastconformer-kda-qkv-check <model.gguf>|--synthetic <work dir> [device] [threads]");
        const std::string device = args.size() > device_at ? args[device_at] : "cpu";
        const int threads = args.size() > device_at + 1 ? std::stoi(args[device_at + 1]) : 4;
        require(threads > 0, "threads must be positive");
        std::string path = args[1];
        if (synthetic) {
            const auto directory = std::filesystem::u8path(args[2]);
            std::filesystem::create_directories(directory);
            path = (directory / "synthetic-kda-qkv-F32.gguf").u8string();
            state_check::write_fixture(path);
        }
        const Layout & selected = synthetic ? state_check::kLayout : layout;
        const ModelFile metadata(path, selected);
        setting("0");
        const bool quantized = quantized_decoding(metadata);
        setting("1");
        require(quantized_decoding(metadata) == quantized, "metadata-only decoding inspection changes with packing enabled");
        std::printf("metadata-only quantized decoding: unchanged, no tensor load\n");
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(nullptr, ggml_threadpool_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(device), threads), ggml_backend_free);
        if (ggml_backend_is_cpu(backend.get())) {
            auto params = ggml_threadpool_params_default(threads);
            pool.reset(ggml_threadpool_new(&params));
            require(bool(pool), "cannot create the CPU check thread pool");
            ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        }
        const ModelFile model(path, backend.get(), selected, decoding_tensor);
        require(model.str("fastconformer.decoder.prediction") == "kda" && model.str("fastconformer.decoder.kind") == "tdt", "this check needs KDA TDT weights");
        std::map<std::string, std::vector<uint8_t>> originals;
        for (int layer = 0; layer < model.count("fastconformer.decoder.prediction_layers"); ++layer) {
            for (const char * part : {"q", "k", "v"}) {
                const std::string name = "pred.kda." + std::to_string(layer) + "." + part + ".weight";
                originals[name] = bytes(model.tensor(name));
            }
        }
        refusal_contract(model);
        setting("0");
        const PredictionNetwork ordinary(model);
        const Joint joint(model);
        Checks checks;
        const auto original = evaluate(ordinary, joint, backend.get(), model, false, false);
        {
            setting("1");
            const PredictionNetwork packed(model);
            graph_contract(model, selected, ordinary, packed, backend.get());
            const auto normal = evaluate(packed, joint, backend.get(), model, false, true);
            const auto a = evaluate(ordinary, joint, backend.get(), model, true, false);
            const auto b = evaluate(packed, joint, backend.get(), model, true, true);
            for (const auto & item : original) {
                checks.equal(a.at(item.first), item.second, "ordinary actual/instrumented " + item.first);
                checks.equal(b.at(item.first), normal.at(item.first), "packed actual/instrumented " + item.first);
                checks.equal(normal.at(item.first), item.second, "ordinary/packed actual " + item.first);
            }
            for (const auto & item : a) checks.equal(b.at(item.first), item.second, "ordinary/packed instrumented " + item.first);
            Steps first(model, ordinary, packed, joint, backend.get()), second(model, ordinary, packed, joint, backend.get());
            const int blank = (int) model.u32("fastconformer.decoder.blank_id");
            const int count = std::max(16, (int) model.width("pred.kda.0.q_conv.weight", 0) + 4);
            for (int i = 0; i < count; ++i) {
                first.advance(i ? (7 * i) % blank : blank, i, checks);
                second.advance(i ? (11 * i + 1) % blank : blank, i + 29, checks);
            }
            checks.expect(first.changed > 1 && second.changed > 1, "both independent histories evolve");
            first.advance(1, 101, checks); second.advance(1, 101, checks);
            if (synthetic) checks.expect(first.last_prediction != second.last_prediction && first.last_logits != second.last_logits,
                                         "the same label and frame retain different history-dependent outputs");
        }
        setting("0");
        const auto after = evaluate(ordinary, joint, backend.get(), model, false, false);
        for (const auto & item : original) checks.equal(after.at(item.first), item.second, "original after packed owner destruction " + item.first);
        for (const auto & item : originals) checks.expect(bytes(model.tensor(item.first)) == item.second, "original tensor bytes remain unchanged: " + item.first);
        std::printf("backend: %s, threads: %d, state: %lld F32 values, synthetic: %s\n", ggml_backend_name(backend.get()), threads,
                    (long long) ordinary.state_size(), synthetic ? "yes" : "no");
        std::printf("KDA QKV: %zu checks, %zu failed; strict step logits differ: %zu; label/duration argmax differ: %zu\n",
                    checks.total, checks.failed, checks.logits_differ, checks.argmax_differ);
        std::printf("official NeMo/FLA reference and complete TDT decision traces: not checked\n");
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
