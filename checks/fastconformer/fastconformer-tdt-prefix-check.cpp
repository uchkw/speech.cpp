// Checks the bounded KDA prediction prefix cache of FastConformer's plain greedy TDT decoding: a cold run records
// the prefix of fed labels it computes with the joint's choice at each, and a warm run reuses an entry wherever the
// whole fed-label prefix and the choices before it still match, computing the joint alone. Every array this check
// compares — the complete KDA state, the projected prediction and all the token, blank and duration logits — must be
// the same number of finite float values with every byte equal; the ratio, the largest difference and the argmaxes
// are printed only to explain a failure, and a difference the argmax agrees with fails all the same, for no rounding
// between the joint-only and the combined graph has been measured that could excuse one.
//
// It runs two ways. With a model and the dumps of reference/fastconformer/dump.py, and with any projected rows given
// as .npy files, it compares the plain, cold and warm runs for equal tokens, frames, durations, graph counts and
// progress sequences, and the recorded entries byte for byte with a fresh replay of the prediction network, wherever
// the weights a place decodes with are reachable, which the model's device alone is: the CPU copy of a GPU model's
// decoding tensors is not, and a decoding there is checked by its output. The inputs are the joint's projected
// encoder rows [T, joint.hidden] — a dump's joint_encoded.npy, the official joint.enc output, read as it lies. With
// --synthetic and a work directory, it writes deterministic synthetic decoder-only weights there and runs the
// controlled cases no dump can be relied on for — the initial blank, blank chains with the duration-0 correction,
// first-argmax ties, the max-symbols frame advance, the last token emitted and never fed, the divergence later
// agreement does not undo, changed frames altering the fresh joint decisions, the entry cap at 63, 64 and more
// predictions, cancellation before the first prediction and after some, and empty input and the reset of a new
// request — on the plain and the caching decoder alike, with expected tokens, frames, durations, graphs and
// progress. Those weights are the check's own data and no model: nothing measured on them is evidence of what
// NeMo's or flash-linear-attention's code computes. The plain run of the same weights is the whole reference
// either way: NeMo's dumps of Haas's KDA prediction and TDT joint do not exist. The cache's own publication rules
// are checked on tiny synthetic states with no model in reach. The CPU computes with the threads given, 4 by
// default, and the run says which count it used.
//
// usage: fastconformer-tdt-prefix-check <model.gguf> <reference out dir> [projected rows .npy...] [gpu|cpu|device name] [threads=N]
//        fastconformer-tdt-prefix-check --synthetic <work dir> [gpu|cpu|device name] [threads=N]

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "npy.h"
#include "reference-dumps.h"
#include "tdt-prefix-cases.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "fastconformer/tdt-prefix.h"

using namespace fastconformer;
using tdt_prefix_check::Progress;
using tdt_prefix_check::exactly_equal;

namespace {

/** A decoding that decodes nothing, standing for a decoder's identity where the check drives the cache by hand. */
struct IdleDecoder : Decoder {
    Decoding decode(const std::vector<float> &, ggml_backend_t, const DecodingProgress &) const override { return {}; }
    int blank() const override { return 0; }
};

/** What the tests of one input need of the model's TDT decoding. */
struct Spec {
    int hidden;
    int outputs;
    int blank;
    std::vector<int32_t> durations;
    uint32_t max_symbols;
};

/** The frame each prediction step of a decoding started on, and how often the max-symbols rule advanced one frame
 * further, replayed from the tokens the decoding emitted with their frames and durations. */
struct StepStarts {
    std::vector<int64_t> at;
    size_t max_symbol_advances = 0;
};

StepStarts starts_of(const Decoding & d, const Spec & spec, size_t steps) {
    StepStarts s;
    s.at.push_back(0);
    int64_t last_token_frame = -1;
    uint32_t tokens_on_frame = 0;
    for (size_t k = 0; k + 1 < steps && k < d.frames.size(); k++) {
        tokens_on_frame = d.frames[k] == last_token_frame ? tokens_on_frame + 1 : 1;
        last_token_frame = d.frames[k];
        const bool advance = tokens_on_frame >= spec.max_symbols && d.durations[k] == 0;
        s.max_symbol_advances += advance;
        s.at.push_back(d.frames[k] + d.durations[k] + (advance ? 1 : 0));
    }
    return s;
}

/** Says whether two decodings agree, naming the first part that differs. */
bool same_decoding(const Decoding & got, const Decoding & want, const std::string & what) {
    if (got.ids == want.ids && got.frames == want.frames && got.durations == want.durations && got.graphs == want.graphs)
        return true;
    if (got.ids != want.ids)
        std::printf("    %s: the token ids differ (%zu vs %zu)\n", what.c_str(), got.ids.size(), want.ids.size());
    else if (got.frames != want.frames)
        std::printf("    %s: the token frames differ\n", what.c_str());
    else if (got.durations != want.durations)
        std::printf("    %s: the token durations differ\n", what.c_str());
    else
        std::printf("    %s: %zu graphs vs %zu\n", what.c_str(), got.graphs, want.graphs);
    return false;
}

/**
 * The entries a plain decoding published against its tokens: the first fed label is the blank every TDT decoding
 * starts with, every later fed label is the token emitted before it, and the last entry holds the last token where
 * the audio ran out under it and the blank where the decoding emitted one to end. A decoding that reached the
 * entry cap is checked by its labels alone, its last choice lying inside the cap.
 */
bool entries_match_tokens(const std::vector<TdtPrefixCache::Entry> & entries, const Decoding & plain, int blank) {
    if (entries.empty()) {
        std::printf("    the decoding published no entry, not even the initial blank\n");
        return false;
    }
    bool ok = true;
    if (entries.front().label != blank) {
        std::printf("    the first entry holds fed label %d, not the blank\n", entries.front().label);
        ok = false;
    }
    for (size_t k = 1; k < entries.size() && k - 1 < plain.ids.size(); k++) {
        if (entries[k].label != plain.ids[k - 1]) {
            std::printf("    entry %zu holds fed label %d, not the token %d emitted before it\n", k, entries[k].label,
                        plain.ids[k - 1]);
            ok = false;
            break;
        }
    }
    if (entries.size() < TdtPrefixCache::kEntries) {
        int32_t want = -1;
        if (entries.size() == plain.ids.size() && !plain.ids.empty())
            want = plain.ids.back();
        else if (entries.size() == plain.ids.size() + 1)
            want = blank;
        else {
            std::printf("    %zu entries after %zu tokens, which is neither the tokens nor one more\n", entries.size(),
                        plain.ids.size());
            ok = false;
        }
        if (want >= 0 && entries.back().next_label != want) {
            std::printf("    the last entry holds next label %d; %zu entries after %zu tokens want %d\n",
                        entries.back().next_label, entries.size(), plain.ids.size(), want);
            ok = false;
        }
    }
    return ok;
}

/**
 * One input on one place: the plain decoding, the cold run that records its prefix and the warm run that reuses
 * it, the entries against the tokens and, where the weights the place decodes with are reachable, against a fresh
 * replay of the prediction network with the joint-only graph of a hit against the combined graph of a step, both
 * compared byte for byte. A `deep` input adds the token a shorter input emitted but never fed, the entry cap on an
 * input long enough to reach it, a forged next label, a cancelled decoding and a request that announces shorter
 * audio.
 */
bool one_input(Recognizer & recognizer, const std::string & name, DecodeOn on, ggml_backend_t backend, ggml_gallocr_t allocr,
               const Spec & spec, const std::vector<float> & rows, bool deep, const std::string & what) {
    bool ok = true;
    const int64_t frames = (int64_t) (rows.size() / (size_t) spec.hidden);
    auto frame = [&](int64_t t) {
        return std::vector<float>(rows.begin() + t * spec.hidden, rows.begin() + (t + 1) * spec.hidden);
    };
    // The audio length the cache sees, in samples, growing with the frames as the engine's own request counts do.
    auto samples_of = [](int64_t f) { return (size_t) f * 320; };
    const DecodingProgress go = [](double) { return true; };

    Progress plain_progress;
    const Decoding plain = recognizer.decoding(rows, name, on, [&](double done) { return plain_progress.step(done); });

    TdtPrefixCache cache;
    {
        TdtPrefixCache::Reading reading = cache.begin(samples_of(frames));
        Progress cold_progress;
        const Decoding cold =
            recognizer.decoding(rows, name, on, [&](double done) { return cold_progress.step(done); }, reading);
        ok = same_decoding(cold, plain, what + ": the cold run differs from the plain one") && ok;
        if (cold_progress.seen != plain_progress.seen) {
            std::printf("    %s: the cold run reported %zu progress callbacks where the plain one reported %zu\n",
                        what.c_str(), cold_progress.seen.size(), plain_progress.seen.size());
            ok = false;
        }
        if (!reading.complete() || reading.hits() != 0) {
            std::printf("    %s: the cold run %s with %zu reused entries\n", what.c_str(),
                        reading.complete() ? "finished" : "did not finish", reading.hits());
            ok = false;
        }
        cache.publish(std::move(reading));
    }
    const size_t steps = cache.entries().size();
    ok = entries_match_tokens(cache.entries(), plain, spec.blank) && ok;

    {
        TdtPrefixCache::Reading reading = cache.begin(samples_of(frames));
        Progress warm_progress;
        const Decoding warm =
            recognizer.decoding(rows, name, on, [&](double done) { return warm_progress.step(done); }, reading);
        ok = same_decoding(warm, plain, what + ": the warm run differs from the plain one") && ok;
        if (warm_progress.seen != plain_progress.seen) {
            std::printf("    %s: the warm run reported %zu progress callbacks where the plain one reported %zu\n",
                        what.c_str(), warm_progress.seen.size(), plain_progress.seen.size());
            ok = false;
        }
        if (!reading.complete() || reading.hits() != steps) {
            std::printf("    %s: the warm run reused %zu of %zu entries and %s\n", what.c_str(), reading.hits(), steps,
                        reading.complete() ? "finished" : "did not finish");
            ok = false;
        }
        cache.publish(std::move(reading));
    }

    // The weights a place decodes with are reachable on the model's device alone; on the CPU copy of a GPU model
    // they are not, and the runs' agreement is all that place is checked by.
    if (on == DecodeOn::Device || recognizer.on_cpu()) {
        if (steps <= plain.ids.size() + 1) {
            const StepStarts starts = starts_of(plain, spec, steps);
            DecoderStep step(recognizer.prediction(), recognizer.joint(), backend, 1);
            PredictionState state = recognizer.prediction().initial_state();
            size_t graphs = 0;
            for (size_t k = 0; k < steps; k++) {
                const std::string entry = what + ": entry " + std::to_string(k);
                step.compute(cache.entries()[k].label, state, frame(starts.at[k]));
                state = step.state();
                const std::vector<float> predicted = step.prediction();
                std::vector<float> combined;
                step.logits(combined);
                ok = exactly_equal(state, cache.entries()[k].state, entry + " replays its state") && ok;
                ok = exactly_equal(predicted, cache.entries()[k].prediction, entry + " replays its prediction") && ok;
                // The joint alone, as a hit computes it from the entry's projected prediction, against the same
                // step's combined graph, on the whole token, blank and duration logits.
                if (k < 4 || k + 4 >= steps) {
                    Graph joint_graph(64);
                    ggml_tensor * logits = recognizer.joint().build(joint_graph.ctx(),
                                                                    joint_graph.input(frame(starts.at[k]), spec.hidden),
                                                                    joint_graph.input(predicted, spec.hidden));
                    joint_graph.output(logits);
                    joint_graph.compute(backend, allocr);
                    const std::vector<float> alone = Graph::read(logits);
                    graphs++;
                    ok = exactly_equal(alone, combined, entry + "'s joint-only logits against the combined graph's",
                                       spec.blank + 1, spec.outputs) &&
                         ok;
                }
            }
            const bool capped = plain.ids.size() >= TdtPrefixCache::kEntries;
            const std::string blank_chain_joints =
                capped ? "" : " (" + std::to_string(plain.graphs - steps) + " blank-chain joints)";
            std::printf("  %s: %zu tokens in %zu steps, %zu graphs%s, %zu max-symbol advances, %zu joint-only graphs "
                        "compared exactly\n",
                        what.c_str(), plain.ids.size(), steps, plain.graphs, blank_chain_joints.c_str(),
                        starts.max_symbol_advances, graphs);
        } else {
            std::printf("  %s: %zu steps after %zu tokens\n", what.c_str(), steps, plain.ids.size());
            ok = false;
        }
    } else {
        std::printf("  %s: %zu tokens in %zu steps, %zu graphs\n", what.c_str(), plain.ids.size(), steps, plain.graphs);
    }

    if (!deep) return ok;

    // A forged next label stops reuse at one chosen step, and the steps after it, whose labels still agree, never
    // resume it. No joint emits a negative label, so the first mismatch is certain whatever the weights.
    if (on == DecodeOn::Device || recognizer.on_cpu()) {
        const size_t forged_at = steps > 1 ? 1 : 0;
        TdtPrefixCache forged;
        {
            TdtPrefixCache::Reading filling = forged.begin(samples_of(frames));
            filling.bind(&recognizer.decoder(name), backend);
            for (size_t k = 0; k < steps; k++) {
                const TdtPrefixCache::Entry & e = cache.entries()[k];
                filling.append(e.label, e.state, e.prediction, k == forged_at ? int32_t(-1) : e.next_label);
            }
            filling.finish();
            forged.publish(std::move(filling));
        }
        TdtPrefixCache::Reading reading = forged.begin(samples_of(frames));
        const Decoding d = recognizer.decoding(rows, name, on, go, reading);
        ok = same_decoding(d, plain, what + ": the run after a forged next label differs") && ok;
        if (reading.hits() != forged_at + 1) {
            std::printf("    %s: %zu entries were reused around the forged next label at %zu, want %zu\n", what.c_str(),
                        reading.hits(), forged_at, forged_at + 1);
            ok = false;
        }
    }

    // A shorter input's last token, which it emitted but never fed, is computed by a longer one rather than reused:
    // the shorter input's entries all match until its audio ran out, and no entry feeds that token.
    if (frames >= 8) {
        const int64_t cut = frames / 2;
        const std::vector<float> shorter(rows.begin(), rows.begin() + cut * spec.hidden);
        const Decoding plain_short = recognizer.decoding(shorter, name, on, go);
        TdtPrefixCache shorter_cache;
        {
            TdtPrefixCache::Reading reading = shorter_cache.begin(samples_of(cut));
            const Decoding cold_short = recognizer.decoding(shorter, name, on, go, reading);
            ok = same_decoding(cold_short, plain_short, what + ": the cold run on shorter audio differs") && ok;
            shorter_cache.publish(std::move(reading));
        }
        const size_t short_steps = shorter_cache.entries().size();
        TdtPrefixCache::Reading reading = shorter_cache.begin(samples_of(frames));
        const Decoding longer = recognizer.decoding(rows, name, on, go, reading);
        ok = same_decoding(longer, plain, what + ": the longer audio after a shorter input differs") && ok;
        // The shorter input ended by a token it never fed where its steps equal its tokens, and then the longer
        // audio reuses every entry; where it ended by a blank, the longer audio reuses all but that last step,
        // or that one too where its blank chain reaches the end and emits the same blank.
        const bool ended_by_token = short_steps == plain_short.ids.size();
        if (ended_by_token ? reading.hits() != short_steps
                           : (reading.hits() + 1 < short_steps || reading.hits() > short_steps)) {
            std::printf("    %s: the longer audio reused %zu of the shorter input's %zu steps, which %s by a token it "
                        "never fed\n",
                        what.c_str(), reading.hits(), short_steps, ended_by_token ? "ended" : "did not end");
            ok = false;
        }
    }

    // The entry cap: an input that repeats this one until it needs more predictions than the cache holds retains
    // the cap's worth of entries, computes the rest and reuses exactly what it kept. Where the model's own audio
    // makes too few tokens, the cap itself is the synthetic cases' to check.
    {
        const int64_t repeats =
            std::min<int64_t>(40, (int64_t) (TdtPrefixCache::kEntries + 8) / (int64_t) std::max<size_t>(steps, 1) + 2);
        std::vector<float> tiled;
        tiled.reserve(rows.size() * (size_t) repeats);
        for (int64_t r = 0; r < repeats; r++) tiled.insert(tiled.end(), rows.begin(), rows.end());
        const Decoding tiled_plain = recognizer.decoding(tiled, name, on, go);
        TdtPrefixCache capped;
        {
            TdtPrefixCache::Reading reading = capped.begin(samples_of(frames * repeats));
            const Decoding tiled_cold = recognizer.decoding(tiled, name, on, go, reading);
            ok = same_decoding(tiled_cold, tiled_plain, what + ": the cold run on tiled audio differs") && ok;
            capped.publish(std::move(reading));
        }
        if (tiled_plain.ids.size() >= TdtPrefixCache::kEntries) {
            if (capped.entries().size() != TdtPrefixCache::kEntries) {
                std::printf("    %s: %zu entries after %zu tokens, want the %zu the cache holds at most\n", what.c_str(),
                            capped.entries().size(), tiled_plain.ids.size(), TdtPrefixCache::kEntries);
                ok = false;
            }
            TdtPrefixCache::Reading reading = capped.begin(samples_of(frames * repeats));
            const Decoding tiled_warm = recognizer.decoding(tiled, name, on, go, reading);
            ok = same_decoding(tiled_warm, tiled_plain, what + ": the warm run on tiled audio differs") && ok;
            if (reading.hits() != TdtPrefixCache::kEntries) {
                std::printf("    %s: the warm run on tiled audio reused %zu entries, want the %zu kept\n", what.c_str(),
                            reading.hits(), TdtPrefixCache::kEntries);
                ok = false;
            }
        } else {
            std::printf("    %s: tiling %lld times made %zu tokens, too few for the entry cap here; the cap's boundary "
                        "is the synthetic cases' to check\n",
                        what.c_str(), (long long) repeats, tiled_plain.ids.size());
        }
    }

    // A decoding cancelled before its first prediction and after some publishes nothing, and the published prefix
    // still serves the next request whole.
    for (const size_t stop : {size_t(1), size_t(3)}) {
        if (stop > steps) continue;
        TdtPrefixCache::Reading reading = cache.begin(samples_of(frames));
        Progress cancel;
        cancel.stop_after = stop;
        recognizer.decoding(rows, name, on, [&](double done) { return cancel.step(done); }, reading);
        if (reading.complete()) {
            std::printf("    %s: a decoding cancelled by its %zu. progress callback is complete\n", what.c_str(), stop);
            ok = false;
        }
        bool refused = false;
        try {
            cache.publish(std::move(reading));
        } catch (const std::logic_error &) {
            refused = true;
        }
        if (!refused) {
            std::printf("    %s: a cancelled decoding's candidate was published\n", what.c_str());
            ok = false;
        }
        TdtPrefixCache::Reading again = cache.begin(samples_of(frames));
        const Decoding redone = recognizer.decoding(rows, name, on, go, again);
        ok = same_decoding(redone, plain, what + ": the run after a cancelled one differs") && ok;
        if (again.hits() != steps) {
            std::printf("    %s: the run after a cancelled one reused %zu of %zu entries\n", what.c_str(), again.hits(), steps);
            ok = false;
        }
    }

    // A request that announces shorter audio decodes as if no cache existed, the published prefix dropped whole.
    {
        TdtPrefixCache::Reading reading = cache.begin(samples_of(frames) - 1);
        if (!cache.entries().empty()) {
            std::printf("    %s: shorter audio kept the published prefix\n", what.c_str());
            ok = false;
        }
        const Decoding d = recognizer.decoding(rows, name, on, go, reading);
        ok = same_decoding(d, plain, what + ": the run after shorter announced audio differs") && ok;
        if (reading.hits() != 0) {
            std::printf("    %s: the run after shorter announced audio reused %zu entries\n", what.c_str(), reading.hits());
            ok = false;
        }
    }
    return ok;
}

/**
 * The cache's own rules, on tiny synthetic states with no model in reach: publication of an unbound, an incomplete
 * or another cache's candidate, the fed-label induction, the first divergence no later agreement undoes, appends
 * past the source's end, the entry cap, the identity of decoder and backend, longer and shorter audio.
 */
bool cache_rules(ggml_backend_t backend, ggml_backend_t other) {
    bool ok = true;
    auto refused = [](auto && body) {
        try {
            body();
        } catch (const std::logic_error &) {
            return true;
        }
        return false;
    };
    const PredictionState state{1.0f, 2.0f, 3.0f};
    const std::vector<float> prediction{0.25f, 0.5f};
    IdleDecoder mine, yours;
    // The chain the tests walk: the blank fed first, two tokens, and the blank that ends an utterance.
    const int32_t chain[] = {0, 5, 7};
    auto choice = [](int32_t label) { return label == 0 ? 5 : label == 5 ? 7 : 0; };

    TdtPrefixCache cache;
    if (!refused([&] { cache.publish(cache.begin(100)); })) {
        std::printf("an unbound candidate was published\n");
        ok = false;
    }
    if (!refused([&] {
            auto reading = cache.begin(100);
            reading.bind(&mine, backend);
            cache.publish(std::move(reading));
        })) {
        std::printf("an incomplete candidate was published\n");
        ok = false;
    }
    {
        // A finished candidate over no entry is the zero-input case and publishes.
        auto reading = cache.begin(100);
        reading.bind(&mine, backend);
        reading.finish();
        cache.publish(std::move(reading));
        if (!cache.entries().empty()) {
            std::printf("an empty candidate published %zu entries\n", cache.entries().size());
            ok = false;
        }
    }
    {
        auto reading = cache.begin(200);
        reading.bind(&mine, backend);
        for (const int32_t label : chain) {
            if (reading.find(label) != nullptr) {
                std::printf("an empty cache's prefix was reused\n");
                ok = false;
            }
            reading.append(label, state, prediction, choice(label));
        }
        reading.finish();
        cache.publish(std::move(reading));
    }
    {
        auto reading = cache.begin(300);
        reading.bind(&mine, backend);
        for (const int32_t label : chain) {
            if (reading.find(label) == nullptr) {
                std::printf("the matching label %d was not reused\n", label);
                ok = false;
            }
            reading.append(label, state, prediction, choice(label));
        }
        reading.finish();
        if (reading.hits() != 3 || reading.size() != 3) {
            std::printf("a matching walk reused %zu of %zu entries\n", reading.hits(), reading.size());
            ok = false;
        }
        cache.publish(std::move(reading));
    }
    {
        auto reading = cache.begin(300);
        reading.bind(&mine, backend);
        if (reading.find(chain[0]) == nullptr) {
            std::printf("the matching label %d was not reused before a divergence\n", chain[0]);
            ok = false;
        }
        // The joint chose 9 where the published entry recorded 5: the labels agree from here on, but the first
        // divergence ended reuse for good.
        reading.append(chain[0], state, prediction, 9);
        for (size_t k = 1; k < 3; k++) {
            if (reading.find(chain[k]) != nullptr) {
                std::printf("the label %d was reused after a divergence\n", chain[k]);
                ok = false;
            }
            reading.append(chain[k], state, prediction, choice(chain[k]));
        }
        reading.finish();
        if (reading.hits() != 1) {
            std::printf("%zu entries were reused around a divergence, want 1\n", reading.hits());
            ok = false;
        }
    }
    {
        // Appends with no lookup between them, through the source's end and past it, end matching rather than read
        // past the source.
        auto reading = cache.begin(300);
        reading.bind(&mine, backend);
        const std::vector<TdtPrefixCache::Entry> entries = cache.entries();
        for (size_t k = 0; k <= entries.size() + 1; k++)
            reading.append(0, state, prediction, k < entries.size() ? entries[k].next_label : 5);
        if (reading.find(0) != nullptr) {
            std::printf("matching went on past the source's end\n");
            ok = false;
        }
    }
    {
        auto reading = cache.begin(300);
        reading.bind(&mine, backend);
        for (size_t k = 0; k < TdtPrefixCache::kEntries + 10; k++)
            reading.append((int32_t) (k % 7), state, prediction, (int32_t) ((k + 1) % 7));
        reading.finish();
        if (reading.size() != TdtPrefixCache::kEntries) {
            std::printf("a candidate kept %zu entries, want the %zu of the cap\n", reading.size(), TdtPrefixCache::kEntries);
            ok = false;
        }
        cache.publish(std::move(reading));
        if (cache.entries().size() != TdtPrefixCache::kEntries) {
            std::printf("the cache kept %zu entries, want the %zu of the cap\n", cache.entries().size(),
                        TdtPrefixCache::kEntries);
            ok = false;
        }
    }
    {
        auto reading = cache.begin(300);
        reading.bind(&yours, backend);
        if (reading.find(0) != nullptr) {
            std::printf("another decoder's prefix was reused\n");
            ok = false;
        }
    }
    {
        auto reading = cache.begin(300);
        reading.bind(&mine, other);
        if (reading.find(0) != nullptr) {
            std::printf("another backend's prefix was reused\n");
            ok = false;
        }
    }
    {
        auto reading = cache.begin(301);
        reading.bind(&mine, backend);
        if (reading.find(0) == nullptr) {
            std::printf("longer audio dropped the prefix\n");
            ok = false;
        }
    }
    if (cache.entries().size() != TdtPrefixCache::kEntries) {
        std::printf("a candidate that was dropped changed the published prefix\n");
        ok = false;
    }
    {
        auto reading = cache.begin(299);
        if (!cache.entries().empty()) {
            std::printf("shorter audio kept the prefix\n");
            ok = false;
        }
        reading.bind(&mine, backend);
        if (reading.find(0) != nullptr) {
            std::printf("a dropped prefix was reused\n");
            ok = false;
        }
    }
    TdtPrefixCache second;
    if (!refused([&] {
            auto reading = second.begin(1);
            reading.bind(&mine, backend);
            reading.finish();
            cache.publish(std::move(reading));
        })) {
        std::printf("another cache's candidate was published\n");
        ok = false;
    }
    std::printf("cache rules: %s\n", ok ? "ok" : "FAIL");
    return ok;
}

/** The projected encoder rows an .npy file holds — [T, joint.hidden], the joint's projection already applied, which
 * the TDT decoding takes as they are and never projects again. The array is checked before anything reads it: a
 * float32 two-dimensional array of at least one frame, of the joint's width. */
std::vector<float> projected_rows(const Npy & npy, int64_t hidden, const std::string & path) {
    if (npy.shape.size() != 2) throw std::runtime_error(path + " is not a two-dimensional array");
    if (npy.shape[0] == 0) throw std::runtime_error(path + " has no frame");
    if (npy.shape[1] != hidden)
        throw std::runtime_error(path + " has " + std::to_string(npy.shape[1]) + " columns where the joint takes " +
                                 std::to_string(hidden) + ", the projected rows the decoding runs on");
    if (npy.f32.size() != (size_t) npy.size()) throw std::runtime_error(path + " is not float32");
    return npy.f32;
}

/**
 * The checks that need the model's own weights and its dumps: every dump of the reference out dir — its
 * joint_encoded.npy read as the official projected rows — and every projected .npy input, each through one_input()
 * on each place the model decodes on. Returns the exit code, 2 where the model is not a KDA TDT one, which this
 * check refuses as it refuses an argument.
 */
int model_checks(const std::string & model_path, const std::string & reference, const std::vector<std::string> & inputs,
                 ggml_backend_t backend, int threads) {
    // The controlled cases of the cap, the blank chains, the ties and the cancellations need no real weights and
    // run with --synthetic alone; this run does not include them.
    std::printf("the controlled synthetic cases run only with --synthetic <work dir>\n");
    bool ok = true;
    Recognizer recognizer(model_path, backend, threads);
    const ModelFile & model = recognizer.model();
    const std::string kind = model.str("fastconformer.decoder.kind");
    const std::string prediction = model.str("fastconformer.decoder.prediction");
    if (kind != "tdt" || prediction != "kda") {
        std::fprintf(stderr,
                     "%s has a %s decoder with a %s prediction network; the prediction prefix cache is of a tdt "
                     "decoder with a kda one\n",
                     model_path.c_str(), kind.c_str(), prediction.c_str());
        return 2;
    }
    const std::string & name = recognizer.decodings().front();
    const Spec spec{recognizer.joint().hidden(), recognizer.joint().outputs(), recognizer.decoder(name).blank(),
                    model.i32_array("fastconformer.decoder.tdt.durations"),
                    model.u32("fastconformer.decoder.tdt.max_symbols")};
    std::vector<DecodeOn> places{DecodeOn::Cpu};
    if (!recognizer.on_cpu()) places.push_back(DecodeOn::Device);
    auto place_name = [](DecodeOn on) { return on == DecodeOn::Cpu ? "cpu" : "device"; };
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));

    // Zero input: no frame is decoded, and the empty candidate publishes and matches nothing.
    {
        TdtPrefixCache empty;
        TdtPrefixCache::Reading reading = empty.begin(0);
        const Decoding d = recognizer.decoding(std::vector<float>(), name, DecodeOn::Device,
                                                [](double) { return true; }, reading);
        if (!d.ids.empty() || d.graphs != 0 || !reading.complete() || reading.hits() != 0) {
            std::printf("zero input: %zu tokens in %zu graphs, %s, %zu reused\n", d.ids.size(), d.graphs,
                        reading.complete() ? "complete" : "incomplete", reading.hits());
            ok = false;
        }
        empty.publish(std::move(reading));
        if (!empty.entries().empty()) {
            std::printf("zero input published %zu entries\n", empty.entries().size());
            ok = false;
        }
    }

    for (const auto & dump : reference_dumps(reference, model)) {
        // The dump's joint_encoded.npy is the official joint.enc output, the projected rows themselves.
        const std::filesystem::path projected = dump / std::filesystem::u8path("joint_encoded.npy");
        const std::vector<float> rows =
            projected_rows(read_npy(projected.u8string()), recognizer.joint().hidden(), projected.u8string());
        std::printf("%s (%lld frames of joint_encoded.npy)\n", dump.filename().u8string().c_str(),
                    (long long) (rows.size() / (size_t) recognizer.joint().hidden()));
        for (const DecodeOn on : places)
            ok = one_input(recognizer, name, on, backend, allocr, spec, rows, true, "on the " + place_name(on)) && ok;
    }
    for (const std::string & path : inputs) {
        const std::vector<float> rows = projected_rows(read_npy(path), recognizer.joint().hidden(), path);
        std::printf("%s (%lld frames)\n", path.c_str(),
                    (long long) (rows.size() / (size_t) recognizer.joint().hidden()));
        for (const DecodeOn on : places)
            ok = one_input(recognizer, name, on, backend, allocr, spec, rows, false, "on the " + place_name(on)) && ok;
    }
    ggml_gallocr_free(allocr);
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    const std::string usage =
        "usage: fastconformer-tdt-prefix-check <model.gguf> <reference out dir> [projected rows .npy...] "
        "[gpu|cpu|device name] [threads=N]\n"
        "       fastconformer-tdt-prefix-check --synthetic <work dir> [gpu|cpu|device name] [threads=N]";
    if (args.size() < 3) {
        std::fprintf(stderr, "%s\n", usage.c_str());
        return 2;
    }
    const bool synthetic = args[1] == "--synthetic";
    std::vector<std::string> inputs;
    std::string device, threads_arg;
    bool threads_given = false;
    for (size_t i = 3; i < args.size(); i++) {
        if (args[i].rfind("threads=", 0) == 0) {
            if (threads_given) {
                std::fprintf(stderr, "%s: threads are named twice, at %s\n", usage.c_str(), args[i].c_str());
                return 2;
            }
            threads_given = true;
            threads_arg = args[i].substr(8);
            continue;
        }
        const bool npy = args[i].size() > 4 && args[i].compare(args[i].size() - 4, 4, ".npy") == 0;
        if (npy && synthetic) {
            std::fprintf(stderr, "%s: --synthetic takes its own synthetic weights and no .npy input\n", usage.c_str());
            return 2;
        }
        if (npy) {
            inputs.push_back(args[i]);
            continue;
        }
        if (device.empty()) {
            device = args[i];
            continue;
        }
        std::fprintf(stderr, "%s: %s is neither an .npy input nor the one device\n", usage.c_str(), args[i].c_str());
        return 2;
    }

    // The threads and the device are settled before any backend is created, so that a refused argument starts
    // nothing.
    int threads = 4;
    if (threads_given) {
        size_t read = 0;
        bool taken = true;
        try {
            threads = std::stoi(threads_arg, &read);
        } catch (const std::exception &) {
            taken = false;
        }
        if (!taken || read != threads_arg.size() || threads < 1 || threads > 1024) {
            std::fprintf(stderr, "%s: threads=%s is not a count from 1 to 1024\n", usage.c_str(), threads_arg.c_str());
            return 2;
        }
    }
    ggml_backend_dev_t chosen = nullptr;
    try {
        chosen = find_device(device.empty() ? "auto" : device);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "%s: %s (%s)\n", usage.c_str(), device.c_str(), e.what());
        return 2;
    }

    try {
        ggml_backend_t backend = start_device(chosen, threads);
        std::printf("backend: %s, cpu threads: %d\n", ggml_backend_name(backend), threads);
        bool ok = true;
        {
            ggml_backend_t other_cpu = start_device(find_device("cpu"), 1);
            ok = cache_rules(backend, other_cpu) && ok;
            ggml_backend_free(other_cpu);
        }
        const int code =
            synthetic ? (tdt_prefix_check::run_cases(args[2], backend) ? 0 : 1)
                      : model_checks(args[1], args[2], inputs, backend, threads);
        ggml_backend_free(backend);
        // The publication rules ran before either mode, so their failure fails whichever mode ran.
        return code == 0 && !ok ? 1 : code;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
