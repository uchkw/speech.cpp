// Checks single-row CPU affine normalization dispatch on ordinary KDA decoder graphs and saves exact comparison data.
// Retained-intermediate graphs are differential references, not official NeMo or FLA stage references.
// Usage: fastconformer-kda-norm-fusion-check <model.gguf|--synthetic> <output dir> <cpu|gpu> <threads> <fusions per step>
// GGML_CPU_NORM_MUL_ADD_FUSION_TRACE must be 1. Each invocation fixes its feature flags before ggml starts.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
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

using Allocator = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;
std::atomic<size_t> fusion_events{0};

void log_event(ggml_log_level, const char * text, void *) {
    constexpr char prefix[] = "ggml_cpu_norm_mul_add_fusion:";
    if (std::strncmp(text, prefix, sizeof(prefix) - 1) == 0) fusion_events.fetch_add(1, std::memory_order_relaxed);
    std::fputs(text, stderr);
}

void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}

struct Checks {
    size_t total = 0, failed = 0;

    void expect(bool ok, const std::string & what) {
        ++total;
        if (!ok) {
            ++failed;
            std::printf("FAIL: %s\n", what.c_str());
        }
    }

    void equal(const std::vector<float> & got, const std::vector<float> & want, const std::string & what) {
        const bool finite = std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }) &&
                            std::all_of(want.begin(), want.end(), [](float x) { return std::isfinite(x); });
        const bool exact = finite && got.size() == want.size() &&
                           std::memcmp(got.data(), want.data(), got.size() * sizeof(float)) == 0;
        expect(exact, what);
        if (!exact) print_diff(what, compare(got, want));
    }
};

std::vector<float> values(int width, int seed) {
    std::vector<float> result((size_t) width);
    for (size_t i = 0; i < result.size(); ++i) result[i] = std::sin(0.031f * (float) (i + 37 * seed));
    return result;
}

/** The ordinary step's graph with the three normalization intermediates explicitly retained. */
class RetainedStep {
public:
    RetainedStep(const PredictionNetwork & prediction, const Joint & joint, ggml_backend_t backend)
        : allocator_(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free) {
        require(bool(allocator_), "cannot allocate the reference graph allocator");
        label_ = graph_.input(std::vector<int32_t>{0}, 1);
        state_ = graph_.input(prediction.initial_state(), prediction.state_size());
        frames_ = graph_.input(std::vector<float>((size_t) joint.hidden()), joint.hidden());
        step_ = prediction.build(graph_, label_, state_);
        predicted_ = joint.project_prediction(graph_.ctx(), step_.output);
        logits_ = joint.build(graph_.ctx(), frames_, predicted_);
        graph_.output(logits_);
        graph_.output(predicted_);
        graph_.output(step_.state);
        int norms = 0;
        for (int i = 0; i < ggml_graph_n_nodes(graph_.cgraph()); ++i) {
            auto * norm = ggml_graph_node(graph_.cgraph(), i);
            if (norm->op != GGML_OP_NORM) continue;
            require(i + 2 < ggml_graph_n_nodes(graph_.cgraph()), "normalization is missing its affine nodes");
            auto * mul = ggml_graph_node(graph_.cgraph(), i + 1);
            auto * add = ggml_graph_node(graph_.cgraph(), i + 2);
            require(mul->op == GGML_OP_MUL && add->op == GGML_OP_ADD &&
                    (mul->src[0] == norm || mul->src[1] == norm) && (add->src[0] == mul || add->src[1] == mul),
                    "the ordinary KDA normalization is not a consecutive affine chain");
            require(norm->type == GGML_TYPE_F32 && norm->ne[0] == 640 && ggml_is_vector(norm),
                    "this KDA check expects a single row of 640 F32 values");
            graph_.output(norm);
            graph_.output(mul);
            ++norms;
        }
        require(norms == 3, "this KDA check expects two pre-norms and a final norm");
    }

    void compute(ggml_backend_t backend, int32_t label, const PredictionState & state, const std::vector<float> & frames) {
        graph_.set(label_, std::vector<int32_t>{label});
        graph_.set(state_, state);
        graph_.set(frames_, frames);
        if (allocated_) graph_.compute_again(backend);
        else {
            graph_.compute(backend, allocator_.get());
            allocated_ = true;
        }
    }

    PredictionState state() const { return PredictionNetwork::read_state(step_); }
    std::vector<float> prediction() const { return Graph::read(predicted_); }
    std::vector<float> logits() const { return Graph::read(logits_); }

private:
    Allocator allocator_;
    Graph graph_{512};
    ggml_tensor * label_ = nullptr;
    ggml_tensor * state_ = nullptr;
    ggml_tensor * frames_ = nullptr;
    ggml_tensor * predicted_ = nullptr;
    ggml_tensor * logits_ = nullptr;
    PredictionNetwork::Step step_{};
    bool allocated_ = false;
};

struct Abort {
    int calls = 0;
    int stop_at = 0;
    bool fired = false;
};

bool abort_after(void * data) {
    auto & abort = *static_cast<Abort *>(data);
    abort.fired = ++abort.calls == abort.stop_at;
    return abort.fired;
}

/** An installed callback keeps all three callbacks, including when none requests cancellation. */
void abort_contract(ggml_backend_t backend, int expected, Checks & checks) {
    Graph graph(32);
    auto * x = graph.input(values(640, 1), 640);
    auto * w = graph.input(values(640, 2), 640);
    auto * b = graph.input(values(640, 3), 640);
    auto * out = ggml_add(graph.ctx(), ggml_mul(graph.ctx(), ggml_norm(graph.ctx(), x, 1e-5f), w), b);
    graph.output(out);
    Allocator allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(bool(allocator), "cannot allocate the callback graph allocator");
    size_t before = fusion_events.load();
    graph.compute(backend, allocator.get());
    checks.expect(fusion_events.load() - before == (size_t) (expected / 3), "direct normal graph dispatch count");
    for (int stop_at : {0, 1, 2}) {
        Abort abort{0, stop_at, false};
        ggml_backend_cpu_set_abort_callback(backend, abort_after, &abort);
        before = fusion_events.load();
        bool stopped = false;
        try {
            graph.compute_again(backend);
        } catch (const Error &) {
            stopped = true;
        }
        ggml_backend_cpu_set_abort_callback(backend, nullptr, nullptr);
        checks.expect(abort.calls == (stop_at == 0 ? 3 : stop_at), "CPU callback count");
        checks.expect(abort.fired == (stop_at != 0) && stopped == abort.fired, "CPU callback actually requests cancellation");
        checks.expect(fusion_events.load() == before, "callback plan does not use affine dispatch fusion");
    }
    graph.set(x, values(640, 9));
    before = fusion_events.load();
    graph.compute_again(backend);
    const auto recovered = Graph::read(out);
    checks.expect(fusion_events.load() - before == (size_t) (expected / 3), "dispatch recovers after callback removal");
    ggml_backend_cpu_set_use_ref(backend, true);
    before = fusion_events.load();
    graph.compute_again(backend);
    ggml_backend_cpu_set_use_ref(backend, false);
    checks.expect(fusion_events.load() == before, "CPU reference disables affine dispatch fusion");
    checks.equal(recovered, Graph::read(out), "new input after CPU cancellation matches the reference");
}

void append(std::ofstream & file, const std::vector<float> & data) {
    file.write(reinterpret_cast<const char *>(data.data()), (std::streamsize) (data.size() * sizeof(float)));
    require(bool(file), "cannot write the exact comparison snapshot");
}

/** Other row widths exercise the same generic path with the caller's explicit CPU thread count. */
void direct_rows(ggml_backend_t backend, int expected, Checks & checks, std::ofstream & snapshots) {
    const bool cpu = ggml_backend_is_cpu(backend);
    for (int width : {1, 639, 640, 641, 1025}) {
        Graph graph(32);
        auto * x = graph.input(values(width, 11), width);
        auto * w = graph.input(values(width, 12), width);
        auto * b = graph.input(values(width, 13), width);
        auto * out = ggml_add(graph.ctx(), ggml_mul(graph.ctx(), ggml_norm(graph.ctx(), x, 1e-5f), w), b);
        graph.output(out);
        Allocator allocator(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        require(bool(allocator), "cannot allocate the direct row graph allocator");
        size_t before = fusion_events.load();
        graph.compute(backend, allocator.get());
        checks.expect(fusion_events.load() - before == (size_t) (expected / 3), "generic single-row dispatch count");
        const auto got = Graph::read(out);
        checks.expect(std::all_of(got.begin(), got.end(), [](float x) { return std::isfinite(x); }), "direct row is finite");
        append(snapshots, got);
        if (cpu) {
            ggml_backend_cpu_set_use_ref(backend, true);
            before = fusion_events.load();
            graph.compute_again(backend);
            ggml_backend_cpu_set_use_ref(backend, false);
            checks.expect(fusion_events.load() == before, "generic row reference does not fuse");
            checks.equal(got, Graph::read(out), "generic row preserves the original kernel arithmetic");
        }
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        require(args.size() == 6, "usage: fastconformer-kda-norm-fusion-check <model.gguf|--synthetic> <output dir> <device> <threads> <fusions per step>");
        const int threads = std::stoi(args[4]), expected = std::stoi(args[5]);
        require(threads > 0 && (expected == 0 || expected == 3), "threads must be positive and fusions per step must be 0 or 3");
        const char * trace = std::getenv("GGML_CPU_NORM_MUL_ADD_FUSION_TRACE");
        require(trace && std::strcmp(trace, "1") == 0, "set GGML_CPU_NORM_MUL_ADD_FUSION_TRACE=1 to verify actual dispatch");
        const auto directory = std::filesystem::u8path(args[2]);
        std::filesystem::create_directories(directory);
        const bool synthetic = args[1] == "--synthetic";
        std::string path = args[1];
        if (synthetic) {
            path = (directory / "synthetic-kda-F32.gguf").u8string();
            state_check::write_fixture(path);
        }
        start_ggml();
        ggml_log_set(log_event, nullptr);
        std::unique_ptr<ggml_threadpool, decltype(&ggml_threadpool_free)> pool(nullptr, ggml_threadpool_free);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[3]), threads), ggml_backend_free);
        const bool cpu = ggml_backend_is_cpu(backend.get());
        require(cpu || expected == 0, "the GPU is an unchanged zero-fusion control");
        if (cpu) {
            auto params = ggml_threadpool_params_default(threads);
            pool.reset(ggml_threadpool_new(&params));
            require(bool(pool), "cannot create the CPU check thread pool");
            ggml_backend_cpu_set_threadpool(backend.get(), pool.get());
        }
        const ModelFile model(path, backend.get(), synthetic ? state_check::kLayout : layout, decoding_tensor);
        require(model.str("fastconformer.decoder.prediction") == "kda" && model.count("fastconformer.decoder.prediction_layers") == 2 &&
                model.boolean("fastconformer.decoder.kda.final_norm"), "this check needs two KDA layers with a final norm");
        const PredictionNetwork prediction(model);
        const Joint joint(model);
        require(prediction.hidden() == 640, "this check expects a prediction width of 640");
        const int blank = (int) model.u32("fastconformer.decoder.blank_id");
        const int durations = (int) model.i32_array("fastconformer.decoder.tdt.durations").size();
        Checks checks;
        std::ofstream states(directory / "states.f32", std::ios::binary);
        std::ofstream predictions(directory / "predictions.f32", std::ios::binary);
        std::ofstream logits(directory / "logits.f32", std::ios::binary);
        std::ofstream inputs(directory / "inputs.f32", std::ios::binary);
        std::ofstream direct(directory / "direct.f32", std::ios::binary);
        std::ofstream decisions(directory / "decisions.txt");
        require(bool(states) && bool(predictions) && bool(logits) && bool(inputs) && bool(direct) && bool(decisions),
                "cannot open exact comparison snapshots");
        std::array<std::unique_ptr<DecoderStep>, 2> normal;
        std::array<std::unique_ptr<RetainedStep>, 2> retained;
        std::array<PredictionState, 2> normal_state, retained_state;
        for (size_t h = 0; h < normal.size(); ++h) {
            normal[h] = std::make_unique<DecoderStep>(prediction, joint, backend.get(), 1);
            if (cpu) retained[h] = std::make_unique<RetainedStep>(prediction, joint, backend.get());
            normal_state[h] = retained_state[h] = prediction.initial_state();
        }
        int changed = 0;
        for (int step = 0; step < 17; ++step) {
            for (size_t h = 0; h < normal.size(); ++h) {
                const int label = (1 + 7 * (int) h + 3 * step) % blank;
                const auto frames = values(joint.hidden(), step + 13 * (int) h);
                size_t before = fusion_events.load();
                normal[h]->compute(label, normal_state[h], frames);
                checks.expect(fusion_events.load() - before == (size_t) expected, "ordinary DecoderStep has the requested three-or-zero dispatches");
                const auto previous = normal_state[h];
                normal_state[h] = normal[h]->state();
                const auto predicted = normal[h]->prediction();
                std::vector<float> got_logits;
                normal[h]->logits(got_logits);
                require(got_logits.size() == (size_t) (blank + 1 + durations), "joint output width differs");
                checks.expect(std::all_of(got_logits.begin(), got_logits.end(), [](float x) { return std::isfinite(x); }) &&
                              std::all_of(predicted.begin(), predicted.end(), [](float x) { return std::isfinite(x); }) &&
                              std::all_of(normal_state[h].begin(), normal_state[h].end(), [](float x) { return std::isfinite(x); }),
                              "ordinary state, prediction and logits are finite");
                const int token = (int) first_argmax(got_logits.data(), blank + 1);
                const int duration = (int) first_argmax(got_logits.data() + blank + 1, durations);
                if (cpu) {
                    before = fusion_events.load();
                    retained[h]->compute(backend.get(), label, retained_state[h], frames);
                    checks.expect(fusion_events.load() == before, "retained normalization intermediates prevent the new fusion");
                    retained_state[h] = retained[h]->state();
                    const auto want_logits = retained[h]->logits();
                    checks.equal(normal_state[h], retained_state[h], "recurrent and convolution state");
                    checks.equal(predicted, retained[h]->prediction(), "projected prediction");
                    checks.equal(got_logits, want_logits, "joint logits");
                    require(want_logits.size() == got_logits.size(), "reference joint output width differs");
                    checks.expect(token == first_argmax(want_logits.data(), blank + 1) &&
                                  duration == first_argmax(want_logits.data() + blank + 1, durations), "label and duration argmax");
                }
                decisions << h << ' ' << step << ' ' << label << ' ' << token << ' ' << duration << '\n';
                require(bool(decisions), "cannot write label and duration decisions");
                append(states, normal_state[h]);
                append(predictions, predicted);
                append(logits, got_logits);
                append(inputs, frames);
                changed += normal_state[h] != previous;
            }
        }
        checks.expect(changed == 34, "each interleaved history updates its state at every step");
        checks.expect(normal_state[0] != normal_state[1], "different label histories retain different states");
        direct_rows(backend.get(), expected, checks, direct);
        if (cpu) abort_contract(backend.get(), expected, checks);
        states.flush(); predictions.flush(); logits.flush(); inputs.flush(); direct.flush(); decisions.flush();
        require(bool(states) && bool(predictions) && bool(logits) && bool(inputs) && bool(direct) && bool(decisions),
                "cannot finish writing exact comparison snapshots");
        std::printf("backend=%s threads=%d steps=34 expected_fusions_per_step=%d observed_fusions=%zu checks=%zu failed=%zu\n",
                    ggml_backend_name(backend.get()), threads, expected, fusion_events.load(), checks.total, checks.failed);
        std::printf("official KDA/prediction reference and public TDT audio paths: not checked by this executable\n");
        return checks.failed ? 1 : 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
