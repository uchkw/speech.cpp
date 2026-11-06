// Compares backend-local KDA constants with the original EXP/NEG nodes, then prediction states, logits and TDT paths.
// The synthetic fixture checks F32 and cancellation; real models additionally check metadata, load and backend ownership.
// Usage: fastconformer-constants-check model.gguf [device] [threads]
//        fastconformer-constants-check --synthetic work-directory [device] [threads]
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "constants-cases.h"
#include "error.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "ggml-cpu.h"

using namespace fastconformer;
using namespace constant_check;

namespace {

struct Environment {
    const char * key = "SPEECH_HAAS_KDA_CONSTANTS";
    bool present;
    std::string original;

    Environment() : present(std::getenv(key) != nullptr), original(present ? std::getenv(key) : "") {}
    void set(const char * value) {
#ifdef _WIN32
        if (_putenv_s(key, value ? value : "") != 0) throw std::runtime_error("cannot set the check's constant switch");
#else
        if ((value ? setenv(key, value, 1) : unsetenv(key)) != 0) throw std::runtime_error("cannot set the check's constant switch");
#endif
    }
    ~Environment() {
#ifdef _WIN32
        _putenv_s(key, present ? original.c_str() : "");
#else
        if (present) setenv(key, original.c_str(), 1);
        else unsetenv(key);
#endif
    }
};

size_t unary_nodes(const Graph & graph, ggml_unary_op op) {
    size_t count = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); i++) {
        const auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op == GGML_OP_UNARY && ggml_get_unary_op(node) == op) ++count;
    }
    return count;
}

void original_constants(const ModelFile & model, const PredictionConstants & cached, ggml_backend_t backend, Checks & checks) {
    const size_t layers = model.u32("fastconformer.decoder.prediction_layers");
    const int64_t heads = model.u32("fastconformer.decoder.kda.num_heads");
    checks.expect(cached.preparations == 1 && cached.values.size() == layers && cached.backend == backend,
                  "one completed preparation belongs to this backend");
    if (cached.values.size() != layers) return;
    checks.expect(cached.allocation_bytes >= layers * heads * sizeof(float), "constant allocation contains the F32 payload");
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    if (!allocator) throw std::runtime_error("cannot create the constant reference allocator");
    for (size_t l = 0; l < layers; l++) {
        Graph graph(32);
        auto * input = ggml_reshape_2d(graph.ctx(), model.tensor("pred.kda." + std::to_string(l) + ".a_log"), 1, heads);
        auto * expected = ggml_neg(graph.ctx(), ggml_exp(graph.ctx(), input));
        graph.output(expected);
        graph.compute(backend, allocator.get());
        const auto * value = cached.values[l];
        checks.expect(value->op == GGML_OP_NONE && value->type == GGML_TYPE_F32 && value->ne[0] == 1 && value->ne[1] == heads,
                      "cached constant preserves the original F32 [1,heads] layout");
        checks.equal(Graph::read(value), Graph::read(expected), "cached EXP then NEG is bitwise the original backend expression");
    }
}

void initial_graphs(const PredictionNetwork & plain, const PredictionNetwork & prepared, ggml_backend_t backend,
                    size_t layers, Checks & checks) {
    Graph a(512), b(512);
    checks.expect(plain.constants().preparations == 0 && prepared.constants().preparations == 0,
                  "constructing the prediction networks executes no preparation");
    const auto x = plain.build(a, 1, plain.initial_state());
    const auto y = prepared.build(b, 1, prepared.initial_state());
    a.output(x.output); a.output(x.state);
    b.output(y.output); b.output(y.state);
    checks.expect(unary_nodes(a, GGML_UNARY_OP_EXP) == layers && unary_nodes(a, GGML_UNARY_OP_NEG) == layers,
                  "the original step executes one EXP and NEG per layer");
    checks.expect(unary_nodes(b, GGML_UNARY_OP_EXP) == 0 && unary_nodes(b, GGML_UNARY_OP_NEG) == 0,
                  "the prepared step has no constant EXP or NEG nodes");
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    if (!allocator) throw std::runtime_error("cannot create the prediction comparison allocator");
    a.compute(backend, allocator.get());
    const auto output = Graph::read(x.output), state = Graph::read(x.state);
    b.compute(backend, allocator.get());
    checks.equal(Graph::read(y.output), output, "first prediction output");
    checks.equal(Graph::read(y.state), state, "first prediction state");
}

struct Steps {
    DecoderStep plain, prepared;
    PredictionState a, b;
    std::vector<float> output, logits;
    int hidden;
    size_t changed = 0;

    Steps(const PredictionNetwork & original, const PredictionNetwork & cached, const Joint & joint, ggml_backend_t backend)
        : plain(original, joint, backend, 1), prepared(cached, joint, backend, 1),
          a(original.initial_state()), b(cached.initial_state()), hidden(joint.hidden()) {}

    void advance(int32_t label, int seed, Checks & checks) {
        const auto frame = frames(hidden, 1, seed);
        plain.compute(label, a, frame);
        prepared.compute(label, b, frame);
        const auto next_a = plain.state(), next_b = prepared.state();
        checks.equal(next_b, next_a, "all recurrent and convolution state values");
        checks.equal(prepared.prediction(), plain.prediction(), "reused prediction graph output");
        std::vector<float> reference;
        plain.logits(reference);
        prepared.logits(logits);
        checks.equal(logits, reference, "joint logits from the reused graph");
        changed += next_a != a;
        a = next_a; b = next_b;
        output = prepared.prediction();
    }
};

void aborted_preparation(const ModelFile & model, ggml_backend_t backend, Checks & checks) {
    if (!ggml_backend_is_cpu(backend)) return;
    PredictionNetwork prediction(model, backend, true);
    struct Abort { size_t calls = 0; } aborted;
    ggml_backend_cpu_set_abort_callback(backend, [](void * context) {
        ++static_cast<Abort *>(context)->calls;
        return true;
    }, &aborted);
    bool threw = false;
    try {
        Graph graph(512);
        prediction.build(graph, 1, prediction.initial_state());
    } catch (const Error &) {
        threw = true;
    } catch (...) {
        ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
        throw;
    }
    ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
    checks.expect(threw && aborted.calls > 0 && prediction.constants().preparations == 0 && prediction.constants().values.empty(),
                  "aborted preparation publishes neither a cache nor a success counter");
    Graph retry(512);
    prediction.build(retry, 1, prediction.initial_state());
    original_constants(model, prediction.constants(), backend, checks);
}

void separate_model(const ModelFile & model, const PredictionConstants & first, ggml_backend_t backend, Checks & checks) {
    std::vector<std::vector<float>> before;
    for (const auto * value : first.values) before.push_back(Graph::read(value));
    ModelFile copy(model, backend, decoding_tensor);
    for (size_t l = 0; l < first.values.size(); l++) {
        auto * tensor = copy.tensor("pred.kda." + std::to_string(l) + ".a_log");
        auto values = Graph::read(tensor);
        for (float & value : values) value += 0.0625f * (float) (l + 1);
        ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
    }
    PredictionNetwork prediction(copy, backend, true);
    Graph graph(512);
    prediction.build(graph, 1, prediction.initial_state());
    const auto other = prediction.constants();
    original_constants(copy, other, backend, checks);
    checks.expect(other.values != first.values, "separate models own separate constant tensors on the same backend");
    for (size_t l = 0; l < first.values.size(); l++) {
        checks.equal(Graph::read(first.values[l]), before[l], "preparing a different model does not modify existing constants");
        checks.expect(Graph::read(other.values[l]) != before[l], "each model prepares constants from its own weights");
    }
}

void recognizer_lifecycle(const std::string & path, ggml_backend_t backend, int threads, Environment & environment, Checks & checks) {
    environment.set("0");
    Recognizer plain(path, backend, threads);
    environment.set("1");
    Recognizer prepared(path, backend, threads);
    environment.set("0");
    for (DecodeOn on : {DecodeOn::Device, DecodeOn::Cpu}) {
        checks.expect(prepared.prediction_constants(on).preparations == 0 && prepared.prediction_constants(on).values.empty(),
                      "pure model load executes no constant preparation or lazy CPU preparation");
    }
    const auto input = frames(plain.joint().hidden(), 7, 31);
    PredictionConstants first, first_cpu;
    for (DecodeOn on : {DecodeOn::Device, DecodeOn::Cpu, DecodeOn::Device, DecodeOn::Cpu}) {
        const auto a = plain.decoding(input, plain.decodings().front(), on);
        const auto b = prepared.decoding(input, prepared.decodings().front(), on);
        checks.expect(a.ids == b.ids && a.frames == b.frames && a.durations == b.durations && a.graphs == b.graphs,
                      "model-owned prepared decoding preserves tokens, frames, durations and graphs");
        checks.expect(plain.detokenizer().text(a.ids) == prepared.detokenizer().text(b.ids), "model-owned prepared text");
        const auto got = prepared.prediction_constants(on);
        checks.expect(got.preparations == 1 && plain.prediction_constants(on).preparations == 0,
                      "the load-time switch snapshot survives environment changes and repeated requests");
        if (first.values.empty()) {
            first = got;
            original_constants(prepared.model(), first, backend, checks);
            if (!prepared.on_cpu()) checks.expect(prepared.prediction_constants(DecodeOn::Cpu).preparations == 0,
                                                 "device prediction does not create the CPU constant copy");
        }
        if (on == DecodeOn::Device || prepared.on_cpu()) {
            checks.expect(first.values == got.values && first.backend == got.backend, "the device cache retains its original allocation");
        } else {
            checks.expect(got.backend != first.backend && got.values != first.values && ggml_backend_is_cpu(got.backend),
                          "the lazy CPU decoder owns an independent cache on its own backend");
            if (first_cpu.values.empty()) first_cpu = got;
            checks.expect(first_cpu.values == got.values && first_cpu.backend == got.backend &&
                          first_cpu.allocation_bytes == got.allocation_bytes,
                          "later CPU requests retain their original constant allocation");
        }
    }
    environment.set(nullptr);
    Recognizer default_off(path, backend, threads);
    default_off.decoding(input, default_off.decodings().front(), DecodeOn::Device);
    checks.expect(default_off.prediction_constants(DecodeOn::Device).preparations == 0, "an unset switch preserves the original path");
}

}  // namespace

int main(int argc, char ** argv) {
    const auto args = utf8_args(argc, argv);
    const bool synthetic = args.size() > 1 && args[1] == "--synthetic";
    const size_t device_at = synthetic ? 3 : 2;
    if (args.size() < device_at || args.size() > device_at + 2) return 2;
    try {
        Environment environment;
        Checks checks;
        const std::string device = args.size() > device_at ? args[device_at] : "cpu";
        const int threads = args.size() > device_at + 1 ? std::stoi(args[device_at + 1]) : 4;
        if (threads < 1) throw std::runtime_error("threads must be positive");
        std::string path = args[1];
        if (synthetic) {
            const auto work = std::filesystem::u8path(args[2]);
            std::filesystem::create_directories(work);
            path = (work / "synthetic-kda-constants-F32.gguf").u8string();
            if (std::filesystem::exists(std::filesystem::u8path(path))) throw std::runtime_error("the synthetic fixture already exists; choose a new work directory");
            write_fixture(path);
        }
        const Layout & selected = synthetic ? kLayout : layout;
        environment.set("1");
        {
            ModelFile metadata(path, selected);
            auto layers = make_prediction_layers(metadata);
            const auto info = layers->constants();
            (void) quantized_decoding(metadata);
            checks.expect(info.preparations == 0 && info.values.empty() && !info.backend,
                          "metadata-only construction and default-option inspection execute no preparation");
        }
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(nullptr, ggml_threadpool_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(device), threads), ggml_backend_free);
        if (ggml_backend_is_cpu(backend.get())) {
            auto params = ggml_threadpool_params_default(threads);
            pool.reset(ggml_threadpool_new(&params));
            if (!pool) throw std::runtime_error("cannot create the check's CPU pool");
            ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        }
        const ModelFile model(path, backend.get(), selected, decoding_tensor);
        if (model.str("fastconformer.decoder.prediction") != "kda") throw std::runtime_error("this check requires KDA prediction weights");
        if (!synthetic && model.str("general.name") != "Haas-ja-v1.0") throw std::runtime_error("the model lifecycle check requires Haas-ja-v1.0");
        const PredictionNetwork plain(model), prepared(model, backend.get(), true);
        const Joint joint(model);
        initial_graphs(plain, prepared, backend.get(), model.u32("fastconformer.decoder.prediction_layers"), checks);
        const auto first = prepared.constants();
        original_constants(model, first, backend.get(), checks);
        {
            Steps a(plain, prepared, joint, backend.get()), b(plain, prepared, joint, backend.get());
            const int blank = (int) model.u32("fastconformer.decoder.blank_id");
            for (int i = 0; i < 16; i++) {
                a.advance(i == 0 ? blank : (7 * i) % blank, i, checks);
                b.advance(i == 0 ? blank : (11 * i + 1) % blank, i + 29, checks);
            }
            checks.expect(a.changed > 1 && b.changed > 1 && a.a != b.a, "interleaved request histories evolve independently");
            a.advance(1, 55, checks); b.advance(1, 55, checks);
            checks.expect(a.output != b.output && a.logits != b.logits, "the same label and frame preserve dependence on history");
        }
        decoding_cases(model, plain, prepared, joint, backend.get(), synthetic, checks);
        checks.expect(prepared.constants().preparations == 1 && prepared.constants().values == first.values &&
                      prepared.constants().allocation_bytes == first.allocation_bytes && plain.constants().preparations == 0,
                      "later graphs, resets, cancellation and boosting retain the same single preparation");
        aborted_preparation(model, backend.get(), checks);
        if (synthetic) separate_model(model, first, backend.get(), checks);
        if (!synthetic) recognizer_lifecycle(path, backend.get(), threads, environment, checks);
        std::printf("backend=%s threads=%d preparations=%zu payload_bytes=%zu allocation_bytes=%zu checks=%zu failed=%zu synthetic=%s\n",
                    ggml_backend_name(backend.get()), threads, first.preparations,
                    first.values.size() * model.u32("fastconformer.decoder.kda.num_heads") * sizeof(float),
                    first.allocation_bytes, checks.total, checks.failed, synthetic ? "yes" : "no");
        std::printf("Official NeMo/FLA reference: not checked; this is a same-backend differential check.\n");
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
