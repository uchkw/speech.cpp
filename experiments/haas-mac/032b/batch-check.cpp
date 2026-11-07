// Checks the 032b B=2 batched pair encoder against fastconformer::Encoder on the check's own synthetic model file:
// two utterances through the production encoder, one at a time, against the two halves of one batched graph, at the
// subsampling, every conformer layer, the encoder's output and the joint's projection of it, in the order data flows,
// so the first stage that departs is the one reported. The pass condition is bit equality of both utterances' stages,
// whose values must also be finite; the largest difference and the SNR are said as measurements and relax nothing.
// Original-versus-batch equality is all this says: how close the encoder stands to NeMo's own numbers is the family
// checks' question, answered on the real models and their reference dumps. The pairs cover different utterances, a
// swap, a duplicate and a change of one member alone, whose unchanged partner must come out bit for bit as before and
// whose changed one must move; the lengths cover T = 1, the kernel-9 convolution's boundary lengths and 31/32/33,
// 63/64/65, 171 and 375. Unequal frontend frame counts, even where the subsampled lengths agree, features that do not
// hold their frame count, nonpositive frame counts and a model of local attention are all refused before ggml can
// assert.
//
// usage: fastconformer-batch2-check <work dir> [cpu|gpu|device name] [f32|f16|q4_k|q8_0]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "batch-encoder.h"
#include "check-support.h"
#include "error.h"
#include "fastconformer/encoder.h"
#include "fastconformer/transducer.h"
#include "fixture.h"
#include "quantize.h"
#include "storage.h"

using namespace fastconformer;

namespace {

/** One utterance through the production encoder: every stage the batched pair is compared against, read to the host. */
struct Baseline {
    std::vector<float> subsampled, joint;
    std::vector<std::vector<float>> layers;
    int64_t nodes = 0;
};

Baseline run_baseline(const Encoder & encoder, const Joint & joint, ggml_backend_t backend, const std::vector<float> & features,
                      int64_t frames) {
    Graph g;
    EncoderStages stages;
    ggml_tensor * projected = joint.project_encoder(g.ctx(), encoder.build(g, features, frames, &stages));
    std::vector<ggml_tensor *> outputs{projected, stages.subsampled};
    outputs.insert(outputs.end(), stages.layers.begin(), stages.layers.end());
    for (ggml_tensor * t : outputs) g.output(t);
    Baseline out;
    out.nodes = g.cgraph()->n_nodes;
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocr(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    compute_poisoned(g, backend, allocr.get(), outputs);
    out.subsampled = Graph::read(stages.subsampled);
    for (ggml_tensor * t : stages.layers) out.layers.push_back(Graph::read(t));
    out.joint = Graph::read(projected);
    return out;
}

/** The batched pair's stages, both utterances' halves of each, split at each result's own batch stride. */
struct BatchRun {
    std::vector<float> subsampled[2], joint[2];
    std::vector<std::vector<float>> layers[2];
    int64_t nodes = 0;
    std::string shapes;
};

std::string shape_of(const ggml_tensor * t) {
    std::string s = "[" + std::to_string(t->ne[0]);
    for (int i = 1; i < 4; i++) s += ", " + std::to_string(t->ne[i]);
    return s + "]";
}

BatchRun run_batch(const BatchEncoder & encoder, const Joint & joint, ggml_backend_t backend, const std::vector<float> & a,
                   int64_t frames_a, const std::vector<float> & b, int64_t frames_b) {
    Graph g;
    BatchStages stages;
    ggml_tensor * projected = joint.project_encoder(g.ctx(), encoder.build(g, a, frames_a, b, frames_b, &stages));
    std::vector<ggml_tensor *> outputs{projected, stages.subsampled, stages.im2col, stages.depthwise, stages.scores};
    outputs.insert(outputs.end(), stages.layers.begin(), stages.layers.end());
    for (ggml_tensor * t : outputs) g.output(t);
    BatchRun out;
    out.nodes = g.cgraph()->n_nodes;
    out.shapes = "im2col " + shape_of(stages.im2col) + ", depthwise " + shape_of(stages.depthwise) + ", scores " +
                 shape_of(stages.scores) + ", subsampled " + shape_of(stages.subsampled);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocr(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    compute_poisoned(g, backend, allocr.get(), outputs);
    std::pair<std::vector<float>, std::vector<float>> halves = split_batches(stages.subsampled);
    out.subsampled[0] = std::move(halves.first);
    out.subsampled[1] = std::move(halves.second);
    for (ggml_tensor * t : stages.layers) {
        halves = split_batches(t);
        out.layers[0].push_back(std::move(halves.first));
        out.layers[1].push_back(std::move(halves.second));
    }
    halves = split_batches(projected);
    out.joint[0] = std::move(halves.first);
    out.joint[1] = std::move(halves.second);
    return out;
}

/**
 * Compares a batched pair with the two single utterances' baselines, both members, stage by stage in the order data
 * flows, and stops reporting a member at the first stage that departs. `second_of_pair` and `first_of_pair` say which
 * baseline each member is, which a swapped pair exchanges.
 */
bool compare_stages(const std::string & name, const BatchRun & pair, const Baseline & first_of_pair, const Baseline & second_of_pair,
                    int64_t single_nodes) {
    // One graph path carries the pair: the batched graph holds about one single graph's nodes, which two
    // independently built graphs would double.
    const int64_t drift = std::llabs(pair.nodes - single_nodes);
    bool ok = drift <= 16;
    if (!ok) {
        std::printf("FAIL %s: the batched graph holds %lld nodes beside one graph's %lld\n", name.c_str(), (long long) pair.nodes,
                    (long long) single_nodes);
    }
    for (int which = 0; which < 2; which++) {
        const Baseline & baseline = which ? second_of_pair : first_of_pair;
        const char * utterance = which ? "B" : "A";
        bool stage_ok = stage_report(name + " " + utterance + " subsampling", pair.subsampled[which], baseline.subsampled);
        for (size_t l = 0; l < baseline.layers.size() && stage_ok; l++) {
            const std::string stage = l + 1 == baseline.layers.size() ? "encoder output" : "layer " + std::to_string(l);
            stage_ok = stage_report(name + " " + utterance + " " + stage, pair.layers[which][l], baseline.layers[l]);
        }
        if (stage_ok) stage_ok = stage_report(name + " " + utterance + " joint projection", pair.joint[which], baseline.joint);
        ok = ok && stage_ok;
    }
    return ok;
}

/** Two deterministic feature arrays no utterance shares with the other, [frames, mels] row-major. */
std::vector<float> features(int64_t frames, int mels, int which) {
    std::vector<float> f((size_t) (frames * mels));
    for (int64_t i = 0; i < frames * mels; i++) {
        f[(size_t) i] = which ? 0.5f + 0.5f * std::cos(0.13 * (double) i + 0.7)
                              : 0.5f + 0.5f * std::sin(0.11 * (double) i + 0.3);
    }
    return f;
}

const char * fault_name(Fault fault) {
    switch (fault) {
        case Fault::InvalidArgument: return "an invalid argument";
        case Fault::OutOfRange: return "out of range";
        case Fault::File: return "a file error";
        case Fault::Device: return "a device error";
        case Fault::OutOfMemory: return "out of memory";
        case Fault::Io: return "an io error";
    }
    return "an error";
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 2 || args.size() > 4) {
        std::fprintf(stderr, "usage: %s <work dir> [cpu|gpu|device name] [f32|f16|q4_k|q8_0]\n", argv[0]);
        return 2;
    }
    const std::string type_name = args.size() > 3 ? args[3] : "f32";
    ggml_type type = GGML_TYPE_F32;
    if (type_name == "f16") type = GGML_TYPE_F16;
    else if (type_name == "q4_k") type = GGML_TYPE_Q4_K;
    else if (type_name == "q8_0") type = GGML_TYPE_Q8_0;
    else if (type_name != "f32") {
        std::fprintf(stderr, "usage: %s <work dir> [cpu|gpu|device name] [f32|f16|q4_k|q8_0]\n", argv[0]);
        return 2;
    }
    const std::filesystem::path work = std::filesystem::u8path(args[1]);
    try {
        std::filesystem::create_directories(work);
        ggml_backend_t backend = start_device(find_device(args.size() > 2 ? args[2] : "auto"), default_threads());
        std::printf("backend: %s\n", ggml_backend_name(backend));
        const std::string f32_file = (work / "batch2-check-F32.gguf").u8string();
        const std::string file = type == GGML_TYPE_F32 ? f32_file : (work / ("batch2-check-" + type_name + ".gguf")).u8string();
        const std::string local_file = (work / "batch2-check-local.gguf").u8string();
        batch2::write_fixture(f32_file, false);
        batch2::write_fixture(local_file, true);
        if (type != GGML_TYPE_F32) {
            const WeightType * wanted = nullptr;
            for (const WeightType & w : weight_types()) {
                if (w.type == type) wanted = &w;
            }
            if (!wanted) throw std::runtime_error(type_name + " is not a weight type speech quantize writes");
            quantize_model_file(f32_file, file, *wanted, kBatch2Layout);
        }
        bool ok = true;
        bool shapes_said = false;
        {
            const ModelFile model(file, backend, kBatch2Layout);
            const Encoder encoder(model, backend);
            const BatchEncoder batch(model, backend);
            const Joint joint(model);
            const int mels = (int) model.u32("fastconformer.frontend.n_mels");
            std::printf("weights: matrices in %s, kernels norms and biases in F32; both encoders cache the position "
                        "projections of a span of %lld on either side of zero\n",
                        tensor_type_text(model.stored_type("blk.0.ff1_up.weight")).c_str(), (long long) batch.cached_span());

            // Runs one pair and its two baselines, and says the shapes of the batched stages the first time.
            auto run_pair = [&](const std::string & name, int64_t t) {
                const int64_t frames = batch2::frames_of(t);
                if (encoder.subsampled_frames(frames) != t) {
                    std::printf("FAIL %s: %lld frames give %lld after subsampling, not %lld\n", name.c_str(), (long long) frames,
                                (long long) encoder.subsampled_frames(frames), (long long) t);
                    return false;
                }
                std::printf("%s: T %lld from %lld frames\n", name.c_str(), (long long) t, (long long) frames);
                const std::vector<float> a = features(frames, mels, 0), b = features(frames, mels, 1);
                const Baseline one = run_baseline(encoder, joint, backend, a, frames);
                const Baseline two = run_baseline(encoder, joint, backend, b, frames);
                const BatchRun pair = run_batch(batch, joint, backend, a, frames, b, frames);
                if (!shapes_said) {
                    std::printf("  one graph's nodes %lld, the pair's %lld; %s\n", (long long) one.nodes, (long long) pair.nodes,
                                pair.shapes.c_str());
                    shapes_said = true;
                }
                return compare_stages(name, pair, one, two, one.nodes);
            };

            // T = 1 with its single shifted row, the lengths the kernel-9 convolution pads at, and the lengths around
            // the powers of two that the subsampling and the attention's views cut.
            for (int64_t t : {1, 2, 3, 5, 9, 31, 32, 33, 63, 64, 65, 171, 375}) {
                ok = run_pair("pair", t) && ok;
            }
            for (int64_t t : {32, 171}) {
                const int64_t frames = batch2::frames_of(t);
                const std::vector<float> a = features(frames, mels, 0), b = features(frames, mels, 1);
                const Baseline one = run_baseline(encoder, joint, backend, a, frames);
                const Baseline two = run_baseline(encoder, joint, backend, b, frames);
                const BatchRun swapped = run_batch(batch, joint, backend, b, frames, a, frames);
                const BatchRun duplicated = run_batch(batch, joint, backend, a, frames, a, frames);
                std::printf("swap and duplicate: T %lld\n", (long long) t);
                ok = compare_stages("swap", swapped, two, one, one.nodes) && ok;
                ok = compare_stages("duplicate", duplicated, one, one, one.nodes) && ok;
            }

            // A change of the second member alone: the first member's result must stay the same bits, the second's
            // must move, and two utterances that cannot be told apart can show neither.
            {
                const int64_t t = 32, frames = batch2::frames_of(t);
                const std::vector<float> a = features(frames, mels, 0), b = features(frames, mels, 1);
                const Baseline one = run_baseline(encoder, joint, backend, a, frames);
                const Baseline two = run_baseline(encoder, joint, backend, b, frames);
                const BatchRun before = run_batch(batch, joint, backend, a, frames, a, frames);
                const BatchRun after = run_batch(batch, joint, backend, a, frames, b, frames);
                std::printf("change of B alone: T %lld\n", (long long) t);
                if (same_bits(one.joint, two.joint)) {
                    std::printf("FAIL: the two utterances' joint projections are the same bits, which no real change can show\n");
                    ok = false;
                }
                if (!same_bits(after.joint[0], before.joint[0])) {
                    std::printf("FAIL: the unchanged utterance's joint projection changed with the other's\n");
                    ok = false;
                } else {
                    std::printf("ok   the unchanged utterance's joint projection kept its bits\n");
                }
                if (same_bits(after.joint[1], before.joint[1])) {
                    std::printf("FAIL: the changed utterance's joint projection did not move, as a constant model's would\n");
                    ok = false;
                } else {
                    std::printf("ok   the changed utterance's joint projection moved\n");
                }
                ok = stage_report("changed B joint projection", after.joint[1], two.joint) && ok;
            }

            // What the builder refuses before ggml can assert, each as an Error of the caller naming the input.
            Graph g;
            auto refuses = [&](const std::string & what, Fault fault, const std::string & input, const std::function<void()> & body) {
                try {
                    body();
                    std::printf("FAIL %s: built where it must refuse\n", what.c_str());
                    return false;
                } catch (const Error & e) {
                    if (e.fault() == fault && e.input() == input) {
                        std::printf("ok   %s: %s\n", what.c_str(), e.what());
                        return true;
                    }
                    std::printf("FAIL %s: refused as %s naming %s where %s naming %s is wanted: %s\n", what.c_str(),
                                fault_name(e.fault()), e.input().c_str(), fault_name(fault), input.c_str(), e.what());
                    return false;
                }
            };
            const int64_t frames = batch2::frames_of(32);
            const std::vector<float> a = features(frames, mels, 0), b = features(frames, mels, 1);
            // 249 and 250 frontend frames both give 32 frames after the subsampling; the counts still differ.
            const std::vector<float> longer = features(frames + 1, mels, 1);
            const std::vector<float> shorter(a.begin(), a.end() - 1);
            ok = refuses("unequal frame counts of one subsampled length", Fault::InvalidArgument, "frames",
                         [&] { batch.build(g, a, frames, longer, frames + 1); }) && ok;
            ok = refuses("features short of their frame count", Fault::InvalidArgument, "features",
                         [&] { batch.build(g, shorter, frames, b, frames); }) && ok;
            ok = refuses("features past their frame count", Fault::InvalidArgument, "features",
                         [&] { batch.build(g, a, frames, longer, frames); }) && ok;
            ok = refuses("no frames at all", Fault::InvalidArgument, "frames", [&] { batch.build(g, a, 0, b, 0); }) && ok;
            ok = refuses("negative frame counts", Fault::InvalidArgument, "frames", [&] { batch.build(g, a, -4, b, -4); }) && ok;
        }
        {
            // A model of local attention, which the batched pair encoder refuses on construction.
            const ModelFile local(local_file, backend, kBatch2Layout);
            try {
                const BatchEncoder batch(local, backend);
                std::printf("FAIL local attention: constructed where it must refuse\n");
                ok = false;
            } catch (const Error & e) {
                if (e.fault() == Fault::File) {
                    std::printf("ok   local attention: %s\n", e.what());
                } else {
                    std::printf("FAIL local attention: refused as %s naming %s where a file error is wanted: %s\n",
                                fault_name(e.fault()), e.input().c_str(), e.what());
                    ok = false;
                }
            }
        }
        if (ok) {
            std::filesystem::remove(std::filesystem::u8path(f32_file));
            std::filesystem::remove(std::filesystem::u8path(file));
            std::filesystem::remove(std::filesystem::u8path(local_file));
            std::printf("ok\n");
        } else {
            std::printf("FAIL\n");
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
