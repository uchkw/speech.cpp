// Runs the 032b B=2 batched pair encoder on a real model file and two recordings, against the same model through
// the production encoder, and times the two side by side on the machine that has the model and the recordings; each
// recording must be at the model's own sample rate, for nothing here resamples it. The differential comes first and
// stops everything when it fails: each half of the batched pair's joint projection must be the baseline's own bits,
// and each half, decoded on the CPU by the existing decoder with a state of its own, must give the baseline's token
// ids, frames, durations, graph count and text. Only then does the timing run: one untimed warm round of both
// methods, and then rounds that run both methods each — the two production encodes one after the other, and the
// batched pair — in an order that alternates with the round index, each method timed between its own wall-clock
// bounds, with the build, the allocation, the compute and the readback said apart, and the pair's makespan said
// apart from the first result's availability and the second utterance's wait. Every timing figure covers the encoder
// and the joint projection alone, so none of them is an utterance's final latency, and no benefit is claimed for a
// single request on its own, which the batched pair's first result can only wait for. The timing graphs keep the
// final joint projection as their one result, without poisoning: the synthetic check keeps every stage and poisons
// it, this differential retains the final joint projection and poisons it, and the timing graphs retain that alone,
// which is a difference in what they allocate and is said so.
//
// usage: fastconformer-batch2-model <model.gguf> <a.wav> <b.wav> [gpu|cpu|device name] [decode threads] [rounds]

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "args.h"
#include "backend.h"
#include "batch-encoder.h"
#include "check-support.h"
#include "error.h"
#include "fastconformer/encoder.h"
#include "fastconformer/recognizer.h"
#include "storage.h"
#include "wav.h"

using namespace fastconformer;

namespace {

using Clock = std::chrono::steady_clock;

double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return n % 2 ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

/** One request's phases, each between two of its own timestamps. */
struct Phases {
    double build = 0, allocate = 0, compute = 0, read = 0;
};

/** One run of a method, its phases and the wall-clock bounds it began and ended within. */
struct Run {
    Phases phases;
    Clock::time_point start, end;
};

/**
 * One utterance through the production encoder and the joint, as Recognizer::encode() runs them, timed phase by
 * phase: the graph built, its memory allocated, the inputs uploaded and it computed, the result read back into the
 * caller's own vector. Nothing else is kept and nothing is poisoned, as a request's own graph is. The phases come
 * back with the run's own bounds, so a method's figures come from its own events whichever order it ran in.
 */
Run run_single(const Encoder & encoder, const Joint & joint, ggml_backend_t backend, ggml_gallocr_t allocr,
               const std::vector<float> & features, int64_t frames, std::vector<float> & values) {
    const auto t0 = Clock::now();
    Graph g;
    ggml_tensor * projected = joint.project_encoder(g.ctx(), encoder.build(g, features, frames));
    g.output(projected);
    const auto t1 = Clock::now();
    if (!ggml_gallocr_alloc_graph(allocr, g.cgraph())) {
        throw Error(Fault::OutOfMemory, "cannot allocate the memory of a graph on the device");
    }
    const auto t2 = Clock::now();
    g.compute_again(backend);
    const auto t3 = Clock::now();
    Graph::read(projected, values);
    const auto t4 = Clock::now();
    return {{ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4)}, t0, t4};
}

/**
 * The pair batched in one graph, timed phase by phase the same way. The result is read back and split into the
 * caller's two vectors before the readback's span ends, so the transfer into them is part of the timing and no copy
 * of the halves falls outside it.
 */
Run run_batched(const BatchEncoder & batch, const Joint & joint, ggml_backend_t backend, ggml_gallocr_t allocr,
                const std::vector<float> & a, const std::vector<float> & b, int64_t frames, std::vector<float> & first,
                std::vector<float> & second) {
    const auto t0 = Clock::now();
    Graph g;
    ggml_tensor * projected = joint.project_encoder(g.ctx(), batch.build(g, a, frames, b, frames));
    g.output(projected);
    const auto t1 = Clock::now();
    if (!ggml_gallocr_alloc_graph(allocr, g.cgraph())) {
        throw Error(Fault::OutOfMemory, "cannot allocate the memory of a graph on the device");
    }
    const auto t2 = Clock::now();
    g.compute_again(backend);
    const auto t3 = Clock::now();
    std::pair<std::vector<float>, std::vector<float>> halves = split_batches(projected);
    first = std::move(halves.first);
    second = std::move(halves.second);
    const auto t4 = Clock::now();
    return {{ms(t0, t1), ms(t1, t2), ms(t2, t3), ms(t3, t4)}, t0, t4};
}

/** Says whether the batched half `got` is the baseline `want`, bit for bit, without relaxing anything. */
bool half_report(const std::string & what, const std::vector<float> & got, const std::vector<float> & want) {
    if (first_not_finite(got) >= 0 || first_not_finite(want) >= 0) {
        std::printf("FAIL %s: a value is not finite\n", what.c_str());
        return false;
    }
    if (!same_bits(got, want)) {
        print_diff("FAIL " + what, compare(got, want));
        return false;
    }
    std::printf("ok   %s: the baseline's own bits\n", what.c_str());
    return true;
}

/** Says whether the two decodings agree on every field and their text. */
bool decoding_report(const std::string & what, const Decoding & got, const Decoding & want, const std::string & got_text,
                     const std::string & want_text) {
    const bool equal = got.ids == want.ids && got.frames == want.frames && got.durations == want.durations &&
                       got.graphs == want.graphs && got_text == want_text;
    if (equal) {
        std::printf("ok   %s: %zu tokens, %zu graphs, text \"%s\"\n", what.c_str(), got.ids.size(), got.graphs, got_text.c_str());
        return true;
    }
    std::printf("FAIL %s: ids %s, frames %s, durations %s, graphs %zu against %zu, text \"%s\" against \"%s\"\n", what.c_str(),
                got.ids == want.ids ? "equal" : "DIFFER", got.frames == want.frames ? "equal" : "DIFFER",
                got.durations == want.durations ? "equal" : "DIFFER", got.graphs, want.graphs, got_text.c_str(), want_text.c_str());
    return false;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 4 || args.size() > 7) {
        std::fprintf(stderr, "usage: %s <model.gguf> <a.wav> <b.wav> [gpu|cpu|device name] [decode threads] [rounds]\n", argv[0]);
        return 2;
    }
    int threads = args.size() > 5 ? std::stoi(args[5]) : 4;
    int rounds = args.size() > 6 ? std::stoi(args[6]) : 10;
    if (threads < 1 || rounds < 1) {
        std::fprintf(stderr, "usage: %s <model.gguf> <a.wav> <b.wav> [gpu|cpu|device name] [decode threads] [rounds]\n"
                             "the decode threads and the rounds must be positive\n",
                     argv[0]);
        return 2;
    }
    try {
        // The Mac condition: the model on the GPU, decoding on the CPU with four threads. The threads are the
        // caller's to choose for a check, and the CPU's own count when the model itself is there.
        ggml_backend_t backend = start_device(find_device(args.size() > 4 ? args[4] : "auto"), threads);
        std::printf("backend: %s, decoding on the cpu with %d threads, %d rounds\n", ggml_backend_name(backend), threads, rounds);
        Recognizer recognizer(args[1], backend, threads);
        const Wav wav_a = read_wav(args[2]), wav_b = read_wav(args[3]);
        // The frontend takes the model's own sample rate alone: a recording at another rate is refused here, even
        // when the two share a sample count and a frontend frame count, and nothing is resampled.
        if (wav_a.sample_rate != recognizer.sample_rate() || wav_b.sample_rate != recognizer.sample_rate()) {
            std::fprintf(stderr, "error: %s is at %d Hz and %s is at %d Hz, and the model takes %d Hz; give recordings at "
                                 "the model's rate, for nothing here resamples them\n",
                         args[2].c_str(), wav_a.sample_rate, args[3].c_str(), wav_b.sample_rate, recognizer.sample_rate());
            return 2;
        }
        const std::vector<float> mono_a = wav_a.mono(), mono_b = wav_b.mono();
        const std::vector<float> features_a = recognizer.frontend().features(mono_a), features_b = recognizer.frontend().features(mono_b);
        const int64_t frames_a = recognizer.frontend().frames(mono_a.size()), frames_b = recognizer.frontend().frames(mono_b.size());
        if (frames_a != frames_b) {
            std::fprintf(stderr, "error: the two recordings give %lld and %lld frontend frames; the batched pair encoder takes one "
                                 "frontend frame count, so give two recordings of the same length\n",
                         (long long) frames_a, (long long) frames_b);
            return 2;
        }
        const int64_t frames = frames_a;
        const Encoder & encoder = recognizer.encoder();
        const Joint & joint = recognizer.joint();
        const BatchEncoder batch(recognizer.model(), backend);
        const int64_t t = batch.subsampled_frames(frames);
        std::printf("%s: matrices in %s; %.2f s and %.2f s of audio, %lld frames, %lld after subsampling; both encoders "
                    "cache the position projections of a span of %lld on either side of zero\n",
                    args[1].c_str(), tensor_type_text(recognizer.model().stored_type("blk.0.ff1_up.weight")).c_str(),
                    (double) mono_a.size() / recognizer.sample_rate(), (double) mono_b.size() / recognizer.sample_rate(),
                    (long long) frames, (long long) t, (long long) batch.cached_span());

        // The differential: the production encoder on each utterance alone, then the batched pair, both halves of
        // every result against their own baseline's bits. A failure here stops the timing; nothing falls back.
        const std::vector<float> baseline_a = recognizer.encode(features_a, frames);
        const std::vector<float> baseline_b = recognizer.encode(features_b, frames);
        std::vector<float> half_a, half_b;
        {
            std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocr(
                ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
            Graph g;
            ggml_tensor * projected = joint.project_encoder(g.ctx(), batch.build(g, features_a, frames, features_b, frames));
            g.output(projected);
            compute_poisoned(g, backend, allocr.get(), {projected});
            const std::pair<std::vector<float>, std::vector<float>> halves = split_batches(projected);
            half_a = halves.first;
            half_b = halves.second;
        }
        bool ok = half_report("joint projection of A", half_a, baseline_a);
        ok = half_report("joint projection of B", half_b, baseline_b) && ok;
        if (!ok) {
            std::printf("FAIL: the batched pair departs from the single encoder, so no timing follows\n");
            return 1;
        }

        // Each half decoded on the CPU by the existing decoder, with a state of its own, against the baseline's own
        // decoding of its own projection.
        const std::string & decoding_name = recognizer.decodings().front();
        const Decoding decoded_a = recognizer.decoding(baseline_a, decoding_name, DecodeOn::Cpu);
        const Decoding decoded_b = recognizer.decoding(baseline_b, decoding_name, DecodeOn::Cpu);
        const Decoding pair_a = recognizer.decoding(half_a, decoding_name, DecodeOn::Cpu);
        const Decoding pair_b = recognizer.decoding(half_b, decoding_name, DecodeOn::Cpu);
        const Detokenizer & detokenizer = recognizer.detokenizer();
        ok = decoding_report("decoding of A", pair_a, decoded_a, detokenizer.text(pair_a.ids), detokenizer.text(decoded_a.ids)) && ok;
        ok = decoding_report("decoding of B", pair_b, decoded_b, detokenizer.text(pair_b.ids), detokenizer.text(decoded_b.ids)) && ok;
        if (!ok) {
            std::printf("FAIL: the batched pair decodes apart from the single encoder, so no timing follows\n");
            return 1;
        }

        // The timing: one untimed warm round of both methods, so Metal's pipelines and both position caches are ready
        // before anything is measured, then rounds that run both methods each — the two production encodes one after
        // the other, and the batched pair — in an order that alternates with the round index, so neither method keeps
        // the first or the last place. Each method is timed between its own wall-clock bounds, whichever order it ran
        // in, with both inputs' features ready at its start, and all four outputs are compared with the differential's
        // baselines once both timed spans have ended. The two methods' allocators are each their own, reused round to
        // round as a request's own is.
        std::printf("timing: every figure below covers the encoder and the joint projection alone, without the frontend "
                    "and the decoding, so none of them is an utterance's final latency\n");
        std::printf("the graphs hold the final joint projection as their one result: the synthetic check keeps every stage "
                    "and poisons it, the strict differential above retains the final joint projection and poisons it, and "
                    "these timing graphs retain that alone and poison nothing, which is a difference in what they allocate\n");
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> single_allocr(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> batch_allocr(
            ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
        std::vector<double> single_makespan, single_first, single_wait;
        std::vector<double> batched_makespan, batched_first, batched_wait;
        std::vector<Phases> single_phases, batched_phases;
        std::vector<float> values_a, values_b, pair_first, pair_second;
        for (int round = -1; round < rounds; round++) {
            const bool singles_first = round % 2 == 0;
            Run single_a, single_b, batched;
            if (singles_first) {
                single_a = run_single(encoder, joint, backend, single_allocr.get(), features_a, frames, values_a);
                single_b = run_single(encoder, joint, backend, single_allocr.get(), features_b, frames, values_b);
                batched = run_batched(batch, joint, backend, batch_allocr.get(), features_a, features_b, frames, pair_first,
                                      pair_second);
            } else {
                batched = run_batched(batch, joint, backend, batch_allocr.get(), features_a, features_b, frames, pair_first,
                                      pair_second);
                single_a = run_single(encoder, joint, backend, single_allocr.get(), features_a, frames, values_a);
                single_b = run_single(encoder, joint, backend, single_allocr.get(), features_b, frames, values_b);
            }
            if (round < 0) {
                std::printf("warm round done; the timing starts with the next round\n");
                continue;
            }
            // All four outputs against the differential's baselines, after both methods' timed spans have ended: a round
            // that departs from them is a failure of the measurement itself.
            if (!same_bits(values_a, baseline_a) || !same_bits(values_b, baseline_b) || !same_bits(pair_first, baseline_a) ||
                !same_bits(pair_second, baseline_b)) {
                std::printf("FAIL round %d: a timed result departs from the differential's baseline\n", round);
                return 1;
            }
            // The two single encodes: the pair runs from the first encode's start to the second's end, the first result
            // is the caller's when its own readback ends, and the second utterance waits from the pair's start until
            // its encode begins, which is after the first is done.
            single_makespan.push_back(ms(single_a.start, single_b.end));
            single_first.push_back(ms(single_a.start, single_a.end));
            single_wait.push_back(ms(single_a.start, single_b.start));
            single_phases.push_back({0.5 * (single_a.phases.build + single_b.phases.build),
                                     0.5 * (single_a.phases.allocate + single_b.phases.allocate),
                                     0.5 * (single_a.phases.compute + single_b.phases.compute),
                                     0.5 * (single_a.phases.read + single_b.phases.read)});
            // The batched pair: both halves are read from one graph, so the first result is the caller's only when the
            // whole pair is, and the second utterance's encode begins with the pair's, so it waits nothing.
            batched_makespan.push_back(ms(batched.start, batched.end));
            batched_first.push_back(ms(batched.start, batched.end));
            batched_wait.push_back(0.0);
            batched_phases.push_back(batched.phases);
        }
        auto say = [](const char * what, const std::vector<double> & v) {
            std::printf("  %-34s median %8.2f ms\n", what, median(v));
        };
        auto phase_medians = [](const std::vector<Phases> & phases) {
            std::vector<double> build, allocate, compute, read;
            for (const Phases & p : phases) {
                build.push_back(p.build);
                allocate.push_back(p.allocate);
                compute.push_back(p.compute);
                read.push_back(p.read);
            }
            return std::array<double, 4>{median(build), median(allocate), median(compute), median(read)};
        };
        std::printf("two single encodes, one after the other:\n");
        say("pair makespan", single_makespan);
        say("first result available", single_first);
        say("second utterance's wait", single_wait);
        const std::array<double, 4> sp = phase_medians(single_phases);
        std::printf("  %-34s build %8.2f ms, allocate %8.2f ms, compute %8.2f ms, read %8.2f ms\n", "phases, per utterance", sp[0],
                    sp[1], sp[2], sp[3]);
        std::printf("the batched pair in one graph:\n");
        say("pair makespan", batched_makespan);
        say("first result available", batched_first);
        say("second utterance's wait", batched_wait);
        const std::array<double, 4> bp = phase_medians(batched_phases);
        std::printf("  %-34s build %8.2f ms, allocate %8.2f ms, compute %8.2f ms, read %8.2f ms\n", "phases, per pair", bp[0], bp[1],
                    bp[2], bp[3]);
        std::printf("  both halves of the pair come from one graph, so a request on its own gains nothing here; the first "
                    "result of a batched pair waits for the whole pair, and its second utterance's wait is zero because "
                    "its encode begins with the pair's\n");
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
