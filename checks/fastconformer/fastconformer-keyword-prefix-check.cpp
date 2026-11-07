// A differential check of the keyword prediction prefix of TDT decoding: the bounded prefix of prediction steps a
// plain decoding records, which a boosted decoding on the same decoder and backend borrows while every token it
// feeds the prediction network matches the recorded one, and never borrows again once one differs. The plain, the
// plain recorded, the ordinary boosted and the prefix-reused decodings of one input run on a backend whose graphs
// are watched (decoding-observer.h), and the reused decoding must decide exactly as the ordinary boosted one: the
// same token ids on the same frames with the same durations, the same graphs, the same progress and cancellation,
// and the same joint logits bytes, for these checked forward paths have no approved floating-point tolerance; an
// argmax that moved fails on its own. Every watched graph of a decoding must leave exactly one logits row of the
// joint's width, all finite, as many rows as the decoding computed graphs, so a missing observation cannot pass; the
// full prediction steps the reuse saves are counted from the graphs the backend itself computed, with the matching
// positions derived from the two ordinary trajectories' token ids alone, so a graph count by itself proves nothing:
// a borrowed position still computes the joint. The recorded entries are replayed through an independent
// DecoderStep sequence fed the same labels from zero, the entry after the last token included, and the prediction
// network must carry its past: the same final label after two different token histories leaves different states,
// joint.pred outputs and joint logits.
//
// A claimed cancellation must reach its cancellation callback and leave less than a full run behind, or it is an
// ordinary completion and proves nothing; the cutoff follows the input's own decoding length, and an input too short
// to cancel part-way is reported as the case it is. The next request after a cancellation must run from zero as
// before.
//
// The synthetic mode writes the all-F32 two-KDA-layer fixture of 003-state-fixture.h into a fresh directory the
// caller gives, which the check refuses to create where anything exists, for its writer overwrites; the fixture
// stays in the caller's directory afterwards, the caller's artifact. The fixture's joint projects the prediction at
// 0.1 on the diagonal and copies the first outputs, so a projected frame chooses the token and the duration each
// step decides by a margin the prediction network's history cannot overturn. Crafted frames cover no divergence, a
// first-token divergence whose initial blank still matches, a middle divergence whose later tokens coincide again
// without the reuse rejoining, runs of blanks, a blank duration of 0 that becomes 1, a nonblank duration of 0 under
// max_symbols, a token whose duration ends the utterance, empty input, and 63, 64 and 65 predictor positions around
// the prefix's capacity of 64, beyond which the ordinary predictor must run again. The model mode runs the same
// differential on the weights of a real Haas-ja-v1.0 file, KDA prediction and TDT decoding, with synthetic frames,
// where the same-weight observations alone are the evidence: official dumps, a quantized file's identity, Vulkan,
// other builds and speed stay unverified by this check. Every backend and Recognizer the check makes computes on
// four CPU threads.
//
// usage: fastconformer-keyword-prefix-check --fixture <fresh-output-directory> [gpu|cpu|device name]
//        fastconformer-keyword-prefix-check <model.gguf> [gpu|cpu|device name]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "decoding-observer.h"
#include "error.h"

#include "fastconformer/003-state-fixture.h"
#include "fastconformer/decoder-step.h"
#include "fastconformer/keyword-prefix.h"
#include "fastconformer/recognizer.h"
#include "fastconformer/tdt.h"

using namespace fastconformer;

namespace {

constexpr int kNever = std::numeric_limits<int>::max();
constexpr int kThreads = 4;

/** The check's verdict, as failures name what failed; every failure keeps its evidence in the output. */
struct Verdict {
    bool ok = true;
    void fail(const std::string & what) {
        std::printf("    FAIL %s\n", what.c_str());
        ok = false;
    }
};

/** A backend this invocation made, freed once every model and observer that uses it has been destroyed. */
struct BackendOwner {
    explicit BackendOwner(ggml_backend_t backend) : backend_(backend) {}
    ~BackendOwner() {
        if (backend_) ggml_backend_free(backend_);
    }
    BackendOwner(const BackendOwner &) = delete;
    BackendOwner & operator=(const BackendOwner &) = delete;
    ggml_backend_t get() const { return backend_; }

private:
    ggml_backend_t backend_;
};

/** The backend of `device` as find_device() reads it, whose CPU computes on the check's four threads. */
ggml_backend_t four_thread_backend(const std::string & device) {
    return start_device(find_device(device.empty() ? "auto" : device), kThreads);
}

/** The first index at which two sequences differ, or -1 when they are equal. */
template <typename T>
int64_t first_difference(const std::vector<T> & a, const std::vector<T> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        if (a[i] != b[i]) return (int64_t) i;
    }
    return a.size() == b.size() ? -1 : (int64_t) n;
}

/** Whether `shorter` is a prefix of `full`. */
bool is_prefix(const std::vector<int32_t> & shorter, const std::vector<int32_t> & full) {
    return shorter.size() <= full.size() && std::equal(shorter.begin(), shorter.end(), full.begin());
}

std::string short_ids(const std::vector<int32_t> & ids) {
    std::string s;
    for (size_t i = 0; i < ids.size() && i < 12; i++) s += (i ? " " : "") + std::to_string(ids[i]);
    if (ids.size() > 12) s += " ... (" + std::to_string(ids.size()) + " ids)";
    return s;
}

/** "" when two decodings decided alike in tokens, frames and durations, else where they part. */
std::string decisions_text(const Decoding & a, const Decoding & b) {
    const int64_t ids = first_difference(a.ids, b.ids);
    if (ids >= 0) return "token ids differ at " + std::to_string(ids) + ": got {" + short_ids(a.ids) + "}, want {" + short_ids(b.ids) + "}";
    const int64_t frames = first_difference(a.frames, b.frames);
    if (frames >= 0) return "emitted frames differ at token " + std::to_string(frames);
    const int64_t durations = first_difference(a.durations, b.durations);
    if (durations >= 0) return "durations differ at token " + std::to_string(durations);
    return "";
}

/** The labels a decoding feeds its prediction network: the blank first, then every token it emits. */
std::vector<int32_t> fed_labels(const Decoding & d, int32_t blank) {
    std::vector<int32_t> fed{blank};
    fed.insert(fed.end(), d.ids.begin(), d.ids.end());
    return fed;
}

/**
 * The positions a reused decoding borrows, derived from the two ordinary trajectories alone: the prefix of fed
 * labels the plain recording and the boosted decoding share, capped by what the recording kept. The first step is
 * fed the blank on both sides, so it matches even where the first emitted token differs.
 */
size_t matching_positions(const KeywordPrefix & prefix, const Decoding & plain, const Decoding & boosted, int32_t blank) {
    const std::vector<int32_t> a = fed_labels(plain, blank), b = fed_labels(boosted, blank);
    size_t m = 0;
    while (m < prefix.entries().size() && m < a.size() && m < b.size() && a[m] == b[m]) m++;
    return m;
}

/** How two float sequences differ: floats whose bytes differ, the largest difference among them, and values that
 * are not finite. */
struct FloatsDiff {
    size_t differing = 0;
    double max_abs = 0;
    bool finite = true;
};

FloatsDiff diff_floats(const std::vector<float> & a, const std::vector<float> & b) {
    FloatsDiff d;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) d.finite = false;
        if (std::memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            d.differing++;
            d.max_abs = std::max(d.max_abs, std::fabs((double) a[i] - (double) b[i]));
        }
    }
    if (a.size() != b.size()) {
        d.differing += std::max(a.size(), b.size()) - n;
        d.max_abs = INFINITY;
    }
    return d;
}

/** Counts a decoding's calls of its progress and answers false on the one call past `cancel_after`. */
struct Progress {
    std::vector<double> calls;
    int cancel_after = kNever;
    bool operator()(double done) {
        calls.push_back(done);
        return (int) calls.size() <= cancel_after;
    }
    /**
     * Whether the run made the call that cancels it; an ordinary completion never does. The sentinel of a
     * non-cancelling run is kept out of the addition, whose int would overflow on it.
     */
    bool cancelled() const { return !calls.empty() && cancel_after != kNever && (int) calls.size() == cancel_after + 1; }
};

/** One family file's prediction network, joint and TDT decoder, on the plain backend and its observing wrapper. */
struct Rig {
    const ModelFile & model;
    const PredictionNetwork & prediction;
    const Joint & joint;
    std::unique_ptr<TdtDecoder> decoder;
    int32_t blank = 0;
    int outputs = 0;
    ggml_backend_t plain = nullptr;
};

/** What one decoding run decided, and what the backend computed for it where the run was watched. */
struct Observed {
    Decoding d;
    std::vector<double> progress;
    bool cancelled_run = false;
    size_t steps = 0, joints = 0, others = 0;
    /** Watched graphs of a decoding kind that left not exactly one logits row of the joint's width. */
    size_t broken_rows = 0;
    std::vector<std::vector<float>> logits;
};

/**
 * One decoding: the plain one, its recorded pass, the boosted one or its prefix-reused pass, watched through
 * `observer` where it is given and alone on the rig's plain backend otherwise.
 */
Observed run(const Rig & rig, observe::Observer * observer, const std::vector<float> & projected, int cancel_after,
             const Boost * boost, KeywordPrefix * record, const KeywordPrefix * reuse) {
    ggml_backend_t backend = observer ? observer->backend() : rig.plain;
    if (observer) observer->clear();
    Progress progress;
    progress.cancel_after = cancel_after;
    DecodingProgress tell = [&](double done) { return progress(done); };
    Decoding d;
    if (reuse) d = rig.decoder->decode_keyword_prefix(projected, backend, tell, *boost, *reuse);
    else if (record) d = rig.decoder->record_keyword_prefix(projected, backend, tell, *record);
    else if (boost) d = rig.decoder->decode(projected, backend, tell, *boost);
    else d = rig.decoder->decode(projected, backend, tell);
    Observed o;
    o.d = d;
    o.progress = progress.calls;
    o.cancelled_run = progress.cancelled();
    if (observer) {
        for (const observe::SeenGraph & g : observer->seen()) {
            if (g.kind == observe::SeenGraph::Kind::Step) o.steps++;
            else if (g.kind == observe::SeenGraph::Kind::Joint) o.joints++;
            else {
                o.others++;
                continue;
            }
            if (g.matched != 1 || g.logits.size() != (size_t) rig.outputs) o.broken_rows++;
            else o.logits.push_back(g.logits);
        }
    }
    return o;
}

/**
 * That a watched run's observation is whole: no graph the observer knows no decoding kind for, no graph of a
 * decoding kind without exactly one logits row of the joint's width, as many rows as prediction and joint-only
 * graphs together, as many graphs as the decoding itself counts, and every value finite.
 */
void validate_observed(Verdict & v, const Rig & rig, const std::string & what, const Observed & o) {
    if (o.others > 0 || o.broken_rows > 0) {
        v.fail(what + ": " + std::to_string(o.others) + " graphs the observer knows no decoding kind for and " +
               std::to_string(o.broken_rows) + " that left not exactly one logits row of " + std::to_string(rig.outputs) +
               " floats; a label-embedding node alone is no decoder step whose decisions were seen");
    }
    if (o.logits.size() != o.steps + o.joints) {
        v.fail(what + ": " + std::to_string(o.logits.size()) + " logits rows for " + std::to_string(o.steps) +
               " prediction and " + std::to_string(o.joints) + " joint-only graphs; a missing observation cannot pass");
    }
    if (o.d.graphs != o.steps + o.joints) {
        v.fail(what + ": the decoding counts " + std::to_string(o.d.graphs) + " graphs, the observer saw " +
               std::to_string(o.steps + o.joints));
    }
    size_t infinite = 0;
    for (const std::vector<float> & row : o.logits) {
        for (float x : row) infinite += !std::isfinite(x);
    }
    if (infinite > 0) v.fail(what + ": the rows hold " + std::to_string(infinite) + " values that are not finite");
}

/** Requires that two runs decided alike, saw the same progress and counted the same graphs. */
void require_equal_runs(Verdict & v, const Observed & got, const Observed & want, const std::string & what) {
    const std::string decisions = decisions_text(got.d, want.d);
    if (!decisions.empty()) v.fail(what + ": " + decisions);
    else if (got.d.graphs != want.d.graphs) {
        v.fail(what + ": " + std::to_string(got.d.graphs) + " graphs where the other run computed " + std::to_string(want.d.graphs));
    } else if (got.progress != want.progress) {
        const int64_t at = std::max(first_difference(got.progress, want.progress), (int64_t) 0);
        v.fail(what + ": the progress calls differ at " + std::to_string(at));
    }
}

/**
 * Requires that two watched runs consumed the same joint logits: the same number of rows, each validated to the
 * joint's width before any argmax is taken, and the same bytes, for these forward paths have no approved
 * floating-point tolerance. The counts and the largest difference are printed for the record, and a row whose
 * argmax of the tokens and the blank, or of the durations, moved fails on its own.
 */
void require_equal_logits(Verdict & v, const Rig & rig, const std::string & what, const Observed & got, const Observed & want) {
    if (got.logits.size() != want.logits.size()) {
        v.fail(what + ": " + std::to_string(got.logits.size()) + " logits rows where the other run consumed " +
               std::to_string(want.logits.size()));
        return;
    }
    for (const std::vector<std::vector<float>> * side : {&got.logits, &want.logits}) {
        for (const std::vector<float> & row : *side) {
            if (row.size() != (size_t) rig.outputs) {
                v.fail(what + ": a logits row of " + std::to_string(row.size()) + " floats, not the joint's " +
                       std::to_string(rig.outputs));
                return;
            }
        }
    }
    const int slots = rig.outputs - (rig.blank + 1);
    size_t byte_differing = 0, floats = 0, label_argmax = 0, duration_argmax = 0;
    double max_abs = 0;
    for (size_t i = 0; i < got.logits.size(); i++) {
        const std::vector<float> & a = got.logits[i], & b = want.logits[i];
        for (size_t j = 0; j < a.size(); j++) {
            if (std::memcmp(&a[j], &b[j], sizeof(float)) != 0) {
                byte_differing++;
                max_abs = std::max(max_abs, std::fabs((double) a[j] - (double) b[j]));
            }
        }
        floats += a.size();
        if (first_argmax(a.data(), rig.blank + 1) != first_argmax(b.data(), rig.blank + 1)) label_argmax++;
        if (slots > 0 && first_argmax(a.data() + rig.blank + 1, slots) != first_argmax(b.data() + rig.blank + 1, slots)) {
            duration_argmax++;
        }
    }
    std::printf("    %s: %zu logits rows, %zu floats, %zu differ in bytes (max |diff| %.2e), the argmax of the tokens"
                " and the blank differs in %zu rows, the durations' in %zu\n",
                what.c_str(), got.logits.size(), floats, byte_differing, max_abs, label_argmax, duration_argmax);
    if (byte_differing > 0) {
        v.fail(what + ": the joint logits are not the same bytes, and no rounding difference is approved here");
    }
    if (label_argmax > 0 || duration_argmax > 0) {
        v.fail(what + ": the joint logits changed what the decoding decides in " + std::to_string(label_argmax + duration_argmax) + " rows");
    }
}

/** The four decodings of one input: the plain control, its recorded pass, the boosted control, the reused pass. */
struct Quartet {
    Observed plain, recorded, boosted, reused;
};

/** What a scenario's crafted input intends, so its case cannot pass vacuously; -1 takes what the runs give. */
struct Expect {
    const std::vector<int32_t> * plain_ids = nullptr;
    const std::vector<int32_t> * boost_ids = nullptr;
    /** The matching positions the two ordinary trajectories must give, where the input is crafted. */
    int64_t matches = -1;
    /** The entries the recording must keep. */
    int64_t entries = -1;
    /** -1 the plain and boosted ids must be equal; >= 0 the first index where they differ, with equal tails. */
    int64_t divergence = -2;
    /** Whether at least the initial blank must be borrowed. */
    bool any_match = false;
    /** Whether to print the prefix's allocated payload. */
    bool payload = false;
};

/** Checks the four watched runs of one input against each other and the intent, and the four unwatched against them. */
void check_quartet(Verdict & v, const Rig & rig, const std::string & name, const Quartet & seen, const KeywordPrefix & prefix,
                   const Quartet & quiet, const Expect & want) {
    validate_observed(v, rig, name + ": the plain pass", seen.plain);
    validate_observed(v, rig, name + ": the recorded pass", seen.recorded);
    validate_observed(v, rig, name + ": the boosted pass", seen.boosted);
    validate_observed(v, rig, name + ": the reused pass", seen.reused);

    if (want.plain_ids && seen.plain.d.ids != *want.plain_ids) {
        v.fail(name + ": the plain decoding gave {" + short_ids(seen.plain.d.ids) + "}, the frames intend {" + short_ids(*want.plain_ids) + "}");
    }
    if (want.boost_ids && seen.boosted.d.ids != *want.boost_ids) {
        v.fail(name + ": the boosted decoding gave {" + short_ids(seen.boosted.d.ids) + "}, the frames intend {" + short_ids(*want.boost_ids) + "}");
    }
    const int64_t diverge = first_difference(seen.plain.d.ids, seen.boosted.d.ids);
    if (want.divergence == -1 && diverge >= 0) {
        v.fail(name + ": the plain and boosted ids differ at " + std::to_string(diverge) + " where none was intended");
    }
    if (want.divergence >= 0) {
        if (diverge != want.divergence) {
            v.fail(name + ": the plain and boosted ids diverge at " + std::to_string(diverge) + ", not at " + std::to_string(want.divergence));
        } else {
            const std::vector<int32_t> tail_a(seen.plain.d.ids.begin() + diverge + 1, seen.plain.d.ids.end());
            const std::vector<int32_t> tail_b(seen.boosted.d.ids.begin() + diverge + 1, seen.boosted.d.ids.end());
            if (tail_a != tail_b) v.fail(name + ": after diverging, the two decodings' ids do not coincide again");
        }
    }

    const size_t matches = matching_positions(prefix, seen.recorded.d, seen.boosted.d, rig.blank);
    const std::string divergence = diverge >= 0 ? ", first different token at " + std::to_string(diverge) : "";
    const std::string capacity = prefix.entries().size() == KeywordPrefix::kCapacity ? " (the capacity)" : "";
    std::printf("  plain %zu ids, boosted %zu ids%s; %zu entries%s, %zu matching positions\n", seen.plain.d.ids.size(),
                seen.boosted.d.ids.size(), divergence.c_str(), prefix.entries().size(), capacity.c_str(), matches);

    require_equal_runs(v, seen.recorded, seen.plain, name + ": the recorded pass differs from the plain one");
    require_equal_logits(v, rig, name + ": the recorded pass against the plain one, which records nothing but copies",
                         seen.recorded, seen.plain);
    require_equal_runs(v, seen.reused, seen.boosted, name + ": the reused pass differs from the ordinary boosted one");
    require_equal_logits(v, rig, name + ": the reused pass against the ordinary boosted one", seen.reused, seen.boosted);

    const std::vector<int32_t> fed = fed_labels(seen.recorded.d, rig.blank);
    if (prefix.entries().size() > fed.size() || prefix.entries().size() > KeywordPrefix::kCapacity) {
        v.fail(name + ": the recording kept " + std::to_string(prefix.entries().size()) + " entries for " + std::to_string(fed.size()) +
               " fed labels");
    }
    for (size_t i = 0; i < prefix.entries().size() && i < fed.size(); i++) {
        if (prefix.entries()[i].label != fed[i]) {
            v.fail(name + ": entry " + std::to_string(i) + " holds label " + std::to_string(prefix.entries()[i].label) +
                   ", the decoding fed " + std::to_string(fed[i]));
            break;
        }
    }
    if (want.entries >= 0 && (int64_t) prefix.entries().size() != want.entries) {
        v.fail(name + ": the recording kept " + std::to_string(prefix.entries().size()) + " entries, expected " + std::to_string(want.entries));
    }
    if (want.matches >= 0 && (int64_t) matches != want.matches) {
        v.fail(name + ": " + std::to_string(matches) + " matching positions, the trajectories give " + std::to_string(want.matches));
    }
    if (want.any_match && matches == 0) v.fail(name + ": nothing was borrowed, not even the initial blank");

    std::printf("  reused pass: steps %zu = %zu - %zu matches, joints %zu = %zu + %zu\n", seen.reused.steps, seen.boosted.steps, matches,
                seen.reused.joints, seen.boosted.joints, matches);
    if (seen.recorded.steps != seen.plain.steps || seen.recorded.joints != seen.plain.joints) {
        v.fail(name + ": recording changed the graphs the plain pass computes");
    }
    if (seen.reused.steps != seen.boosted.steps - matches || seen.reused.joints != seen.boosted.joints + matches) {
        v.fail(name + ": the reused pass computed " + std::to_string(seen.reused.steps) + " full steps and " +
               std::to_string(seen.reused.joints) + " joint-only graphs, where " + std::to_string(seen.boosted.steps) + " - " +
               std::to_string(matches) + " matches and " + std::to_string(seen.boosted.joints) + " + " + std::to_string(matches) + " follow");
    }

    require_equal_runs(v, quiet.plain, seen.plain, name + ": the unwatched plain run differs from the watched one");
    require_equal_runs(v, quiet.recorded, seen.recorded, name + ": the unwatched recorded run differs from the watched one");
    require_equal_runs(v, quiet.boosted, seen.boosted, name + ": the unwatched boosted run differs from the watched one");
    require_equal_runs(v, quiet.reused, seen.reused, name + ": the unwatched reused run differs from the watched one");
    std::printf("  the four unwatched runs on the plain backend decide and progress as the watched ones\n");

    if (want.payload) {
        const size_t least = prefix.entries().size() * (size_t) (rig.prediction.state_size() + rig.joint.hidden()) * sizeof(float);
        const size_t payload = prefix.payload_bytes();
        std::printf("  the prefix's vectors hold %zu bytes for %zu entries (at least %zu of states and predictions alone); "
                    "this is the payload the prefix allocated, not the process's heap\n",
                    payload, prefix.entries().size(), least);
        if (payload < least) v.fail(name + ": the prefix's payload is below its states and predictions alone");
    }
}

/** Runs the four decodings watched and the four unwatched, and checks them; the input's craft is `want`'s claim. */
void scenario(Verdict & v, const Rig & rig, observe::Observer & observer, const std::string & name,
              const std::vector<float> & projected, const Boost & boost, const Expect & want) {
    std::printf("%s (%lld frames)\n", name.c_str(), (long long) (projected.size() / (size_t) rig.joint.hidden()));
    KeywordPrefix prefix, quiet_prefix;
    Quartet seen;
    seen.plain = run(rig, &observer, projected, kNever, nullptr, nullptr, nullptr);
    seen.recorded = run(rig, &observer, projected, kNever, nullptr, &prefix, nullptr);
    seen.boosted = run(rig, &observer, projected, kNever, &boost, nullptr, nullptr);
    seen.reused = run(rig, &observer, projected, kNever, &boost, nullptr, &prefix);
    Quartet quiet;
    quiet.plain = run(rig, nullptr, projected, kNever, nullptr, nullptr, nullptr);
    quiet.recorded = run(rig, nullptr, projected, kNever, nullptr, &quiet_prefix, nullptr);
    quiet.boosted = run(rig, nullptr, projected, kNever, &boost, nullptr, nullptr);
    quiet.reused = run(rig, nullptr, projected, kNever, &boost, nullptr, &quiet_prefix);
    check_quartet(v, rig, name, seen, prefix, quiet, want);
}

/**
 * Compares a plain recording's entries with an independent ordinary DecoderStep sequence fed the same labels from
 * zero, on the plain backend with no observer: the labels, every state-after byte and every joint.pred byte, all
 * finite. Prediction is independent of the audio frame, so the replay runs every entry on one fixed frame of the
 * joint's width, the final entry after the last token, whose emission is the blank or the end of the frames,
 * included.
 */
void recorded_entries_check(Verdict & v, const Rig & rig, const std::vector<float> & projected, const char * name) {
    std::printf("%s: the recorded entries against an independent prediction sequence from zero\n", name);
    KeywordPrefix prefix;
    const Decoding recorded = rig.decoder->record_keyword_prefix(projected, rig.plain, DecodingProgress(), prefix);
    const std::vector<int32_t> fed = fed_labels(recorded, rig.blank);
    const std::vector<KeywordPrefix::Entry> & entries = prefix.entries();
    if (entries.empty()) {
        std::printf("  the input gave the recording no entries\n");
        return;
    }
    if (entries.size() > fed.size() || entries.size() > KeywordPrefix::kCapacity) {
        v.fail("the recording kept more entries than the decoding fed labels or than its capacity");
        return;
    }
    for (size_t i = 0; i < entries.size(); i++) {
        if (entries[i].label != fed[i]) {
            v.fail("entry " + std::to_string(i) + " holds label " + std::to_string(entries[i].label) + ", the decoding fed " +
                   std::to_string(fed[i]));
            break;
        }
    }
    size_t infinite = 0;
    for (const KeywordPrefix::Entry & e : entries) {
        for (float x : e.state) infinite += !std::isfinite(x);
        for (float x : e.prediction) infinite += !std::isfinite(x);
    }
    if (infinite > 0) v.fail("the recording holds " + std::to_string(infinite) + " values that are not finite");

    DecoderStep step(rig.prediction, rig.joint, rig.plain, 1);
    PredictionState state = rig.prediction.initial_state();
    const std::vector<float> frame((size_t) rig.joint.hidden(), 0.0f);
    size_t states_differing = 0, predictions_differing = 0, worst_at = 0;
    double worst = 0;
    for (size_t j = 0; j < entries.size(); j++) {
        step.compute(entries[j].label, state, frame);
        state = step.state();
        const FloatsDiff s = diff_floats(state, entries[j].state);
        const FloatsDiff p = diff_floats(step.prediction(), entries[j].prediction);
        states_differing += s.differing;
        predictions_differing += p.differing;
        if (std::max(s.max_abs, p.max_abs) > worst) {
            worst = std::max(s.max_abs, p.max_abs);
            worst_at = j;
        }
    }
    std::printf("  all %zu entries replayed from zero, the final one included: %zu state floats and %zu joint.pred floats"
                " differ in bytes (max |diff| %.2e at entry %zu)\n",
                entries.size(), states_differing, predictions_differing, worst, worst_at);
    if (states_differing + predictions_differing > 0) {
        v.fail("the independent prediction sequence from zero differs from the recording");
    }
}

/**
 * That the prediction network carries its token past: two independent ordinary DecoderStep histories from zero that
 * differ in one earlier label and end on the same label, on one and the same fixed frame, must leave different
 * states, different joint.pred outputs and different joint logits, each on its own and all finite. A zero difference
 * in any one of the three would let a recorded entry stand in for a step the decoding never took.
 */
void history_sensitivity_check(Verdict & v, const Rig & rig, int32_t first, int32_t second, int32_t last) {
    std::printf("the same final label %d after the earlier labels %d and %d, on one fixed frame\n", (int) last, (int) first, (int) second);
    const std::vector<float> frame((size_t) rig.joint.hidden(), 0.0f);
    DecoderStep a(rig.prediction, rig.joint, rig.plain, 1), b(rig.prediction, rig.joint, rig.plain, 1);
    PredictionState state = rig.prediction.initial_state();
    a.compute(rig.blank, state, frame);
    state = a.state();
    a.compute(first, state, frame);
    state = a.state();
    a.compute(last, state, frame);
    PredictionState other = rig.prediction.initial_state();
    b.compute(rig.blank, other, frame);
    other = b.state();
    b.compute(second, other, frame);
    other = b.state();
    b.compute(last, other, frame);
    std::vector<float> a_out, b_out;
    a.logits(a_out);
    b.logits(b_out);
    const FloatsDiff states = diff_floats(a.state(), b.state());
    const FloatsDiff predictions = diff_floats(a.prediction(), b.prediction());
    const FloatsDiff logits = diff_floats(a_out, b_out);
    std::printf("  the states differ in %zu floats, the joint.pred outputs in %zu, the joint logits in %zu"
                " (max |diff| %.2e)\n",
                states.differing, predictions.differing, logits.differing,
                std::max(std::max(states.max_abs, predictions.max_abs), logits.max_abs));
    if (!states.finite || !predictions.finite || !logits.finite) {
        v.fail("one of the two token histories left values that are not finite");
    }
    if (states.differing == 0) v.fail("two token pasts left the same prediction state");
    if (predictions.differing == 0) v.fail("two token pasts left the same joint.pred output");
    if (logits.differing == 0) v.fail("two token pasts left the same joint logits on one frame");
}

/** The cutoff a cancelled run uses: the one wanted where the input reaches it, else the last it does, labeled. */
int reachable_cutoff(const std::string & what, size_t steps, int wanted) {
    const int reach = (int) std::min<size_t>(steps - 1, (size_t) std::max(wanted, 1));
    if ((size_t) reach < (size_t) wanted) {
        std::printf("  the intended cutoff after %d progress calls is beyond the %zu this input's %s gives; cancelling"
                    " after %d, the deepest call that leaves a partial run\n",
                    wanted, steps, what.c_str(), reach);
    }
    return reach;
}

/** Requires that a claimed cancelled run reached its cancellation callback and left less than a full run behind. */
void require_cancelled(Verdict & v, const std::string & what, const Observed & cancelled, int cancel_after, const Observed & full) {
    if (!cancelled.cancelled_run) {
        v.fail(what + " completed on its own after " + std::to_string(cancelled.progress.size()) +
               " progress calls without reaching the " + std::to_string(cancel_after + 1) + "th that cancels it; an ordinary"
               " early completion is no cancellation");
        return;
    }
    if (cancelled.d.graphs >= full.d.graphs) {
        v.fail(what + " was cancelled yet computed as many graphs as a full run");
    }
}

/**
 * A record that progress cancels, whose partial prefix a full reused decoding must still borrow correctly; then
 * fresh requests from zero, which the cancellation must not have touched. The cutoff follows the plain decoding's
 * own length, for a cancellation an input never reaches proves nothing.
 */
void cancelled_record_check(Verdict & v, const Rig & rig, observe::Observer & observer, const std::vector<float> & projected,
                            const Boost & boost, int wanted) {
    std::printf("a record cancelled by progress, then the full decoding on its partial prefix\n");
    const Observed full_plain = run(rig, nullptr, projected, kNever, nullptr, nullptr, nullptr);
    if (full_plain.progress.size() < 2) {
        std::printf("  the input gives %zu decoding steps, too few to cancel part-way; the cancelled-record case is"
                    " unreached on it\n",
                    full_plain.progress.size());
        return;
    }
    const int cancel_after = reachable_cutoff("plain decoding", full_plain.progress.size(), wanted);
    KeywordPrefix partial;
    const Observed cancelled = run(rig, &observer, projected, cancel_after, nullptr, &partial, nullptr);
    validate_observed(v, rig, "the cancelled record", cancelled);
    require_cancelled(v, "the cancelled record", cancelled, cancel_after, full_plain);
    if (partial.entries().empty()) {
        v.fail("the cancelled record kept no entry; the cutoff came before any recorded prediction step");
    }
    if (partial.entries().size() > fed_labels(cancelled.d, rig.blank).size()) {
        v.fail("the cancelled record kept more entries than it fed labels");
    }
    const Observed boosted = run(rig, &observer, projected, kNever, &boost, nullptr, nullptr);
    const Observed reused = run(rig, &observer, projected, kNever, &boost, nullptr, &partial);
    validate_observed(v, rig, "the full boosted decoding beside a partial record", boosted);
    validate_observed(v, rig, "the reused decoding on a partially recorded prefix", reused);
    const size_t matches = matching_positions(partial, cancelled.d, boosted.d, rig.blank);
    std::printf("  the cancelled record kept %zu entries of a partial plain decoding of %zu tokens; %zu matching"
                " positions against the full boosted decoding\n",
                partial.entries().size(), cancelled.d.ids.size(), matches);
    require_equal_runs(v, reused, boosted, "the full reused decoding on a partially recorded prefix");
    require_equal_logits(v, rig, "the full reused decoding on a partially recorded prefix", reused, boosted);
    if (reused.steps != boosted.steps - matches || reused.joints != boosted.joints + matches) {
        v.fail("the reused decoding on the partial prefix computed graphs the matches do not explain");
    }
    KeywordPrefix fresh;
    const Observed fresh_plain = run(rig, &observer, projected, kNever, nullptr, nullptr, nullptr);
    const Observed fresh_recorded = run(rig, &observer, projected, kNever, nullptr, &fresh, nullptr);
    const Observed fresh_reused = run(rig, &observer, projected, kNever, &boost, nullptr, &fresh);
    validate_observed(v, rig, "a fresh plain pass after a cancelled record", fresh_plain);
    validate_observed(v, rig, "a fresh recorded pass after a cancelled record", fresh_recorded);
    validate_observed(v, rig, "a fresh reused pass after a cancelled record", fresh_reused);
    require_equal_runs(v, fresh_recorded, fresh_plain, "a fresh recorded pass after a cancelled record");
    require_equal_runs(v, fresh_reused, boosted, "a fresh reused pass after a cancelled record");
    require_equal_logits(v, rig, "a fresh reused pass after a cancelled record", fresh_reused, boosted);
    if (!is_prefix(cancelled.d.ids, full_plain.d.ids)) {
        v.fail("the cancelled record emitted tokens the full plain decoding does not");
    } else if (cancelled.d.ids.size() == full_plain.d.ids.size() && partial.entries().size() == fresh.entries().size()) {
        v.fail("the cancelled record kept as many entries and tokens as a full one; the cutoff left nothing partial");
    }
    std::printf("  a fresh request from zero decides as before\n");
}

/** A reuse that progress cancels, which must decide as an ordinary boosted decoding cancelled at the same call; then
 * a fresh request from zero. */
void cancelled_reuse_check(Verdict & v, const Rig & rig, observe::Observer & observer, const std::vector<float> & projected,
                           const Boost & boost, int wanted) {
    std::printf("a reuse cancelled by progress, beside the boosted decoding cancelled with it\n");
    KeywordPrefix prefix;
    const Observed recorded = run(rig, &observer, projected, kNever, nullptr, &prefix, nullptr);
    const Observed plain = run(rig, &observer, projected, kNever, nullptr, nullptr, nullptr);
    const Observed full_boost = run(rig, &observer, projected, kNever, &boost, nullptr, nullptr);
    validate_observed(v, rig, "the plain recording before a cancelled reuse", recorded);
    validate_observed(v, rig, "the plain decoding before a cancelled reuse", plain);
    validate_observed(v, rig, "the full boosted decoding before a cancelled reuse", full_boost);
    if (full_boost.progress.size() < 2) {
        std::printf("  the input gives %zu decoding steps, too few to cancel part-way; the cancelled-reuse case is"
                    " unreached on it\n",
                    full_boost.progress.size());
        return;
    }
    const int cancel_after = reachable_cutoff("boosted decoding", full_boost.progress.size(), wanted);
    const Observed boosted = run(rig, &observer, projected, cancel_after, &boost, nullptr, nullptr);
    const Observed reused = run(rig, &observer, projected, cancel_after, &boost, nullptr, &prefix);
    validate_observed(v, rig, "the cancelled boosted decoding", boosted);
    validate_observed(v, rig, "the cancelled reused decoding", reused);
    require_cancelled(v, "the cancelled boosted decoding", boosted, cancel_after, full_boost);
    require_cancelled(v, "the cancelled reused decoding", reused, cancel_after, full_boost);
    require_equal_runs(v, recorded, plain, "the plain recording beside a cancelled reuse");
    require_equal_logits(v, rig, "the plain recording beside a cancelled reuse", recorded, plain);
    require_equal_runs(v, reused, boosted, "the cancelled reused decoding");
    require_equal_logits(v, rig, "the cancelled reused decoding", reused, boosted);
    const size_t matches = std::min(matching_positions(prefix, recorded.d, boosted.d, rig.blank), boosted.steps);
    std::printf("  %zu of the %zu positions the cancelled runs computed were borrowed\n", matches, boosted.steps);
    if (matches == 0) v.fail("the cancelled reuse borrowed nothing, not even the step fed the blank");
    if (reused.steps != boosted.steps - matches || reused.joints != boosted.joints + matches) {
        v.fail("the cancelled reuse computed graphs the matches do not explain");
    }
    if (!is_prefix(boosted.d.ids, full_boost.d.ids)) {
        v.fail("the cancelled boosted decoding emitted tokens the full one does not");
    }
    const Observed fresh_reuse = run(rig, &observer, projected, kNever, &boost, nullptr, &prefix);
    validate_observed(v, rig, "a fresh full reused pass after a cancelled reuse", fresh_reuse);
    require_equal_runs(v, fresh_reuse, full_boost, "a fresh full reused pass after a cancelled reuse");
    require_equal_logits(v, rig, "a fresh full reused pass after a cancelled reuse", fresh_reuse, full_boost);
    std::printf("  a fresh request from zero decides as the ordinary boosted decoding\n");
}

/** That a prefix of another owner is refused before any graph is computed. */
void expect_refusal(Verdict & v, observe::Observer & observer, const std::function<void()> & call, const char * what) {
    observer.clear();
    try {
        call();
        v.fail(std::string(what) + " was not refused");
    } catch (const std::logic_error & e) {
        std::printf("  %s refused before compute: %s\n", what, e.what());
    }
    if (!observer.seen().empty()) v.fail(std::string(what) + " computed graphs before refusing");
}

/**
 * That a prefix is refused to another decoder on the same backend, and to the right decoder on another backend:
 * one recorded on the plain backend under the observer's wrapper, and, where `foreign` is given, one a decoder on
 * a backend of its own recorded.
 */
void owner_refusals_check(Verdict & v, const Rig & rig, observe::Observer & observer, const std::vector<float> & projected,
                          const Boost & boost, TdtDecoder & other, TdtDecoder * foreign_decoder, const KeywordPrefix * foreign) {
    std::printf("a prefix of another owner\n");
    KeywordPrefix others;
    other.record_keyword_prefix(projected, observer.backend(), DecodingProgress(), others);
    expect_refusal(v, observer, [&] { rig.decoder->decode_keyword_prefix(projected, observer.backend(), DecodingProgress(), boost, others); },
                   "a prefix of another decoder");
    KeywordPrefix here;
    rig.decoder->record_keyword_prefix(projected, rig.plain, DecodingProgress(), here);
    expect_refusal(v, observer, [&] { rig.decoder->decode_keyword_prefix(projected, observer.backend(), DecodingProgress(), boost, here); },
                   "a prefix of another backend");
    if (foreign && foreign_decoder) {
        expect_refusal(v, observer,
                       [&] { foreign_decoder->decode_keyword_prefix(projected, observer.backend(), DecodingProgress(), boost, *foreign); },
                       "a prefix of a second backend's own decoder");
    }
}

/**
 * One projected fixture frame: the joint's logits are ReLU(frame + 0.1 * prediction output), so a large value at a
 * token's output decides the token and one at a duration slot the duration, whatever past the prediction network
 * carries; a frame that intends a divergence sits its winner a small margin above the alternate token, which the
 * tree's bonus for the alternate then overturns in the boosted decoding alone.
 */
std::vector<float> fixture_frame(int64_t hidden, int32_t blank, int32_t token, float token_value, int32_t alternate,
                                 float alternate_value, int duration_slot) {
    std::vector<float> frame((size_t) hidden, 0.0f);
    if (token >= 0) frame[(size_t) token] = token_value;
    if (alternate >= 0 && alternate != token) frame[(size_t) alternate] = alternate_value;
    frame[(size_t) (blank + 1 + duration_slot)] = 4.0f;
    return frame;
}

void append_frame(std::vector<float> & projected, const std::vector<float> & frame) {
    projected.insert(projected.end(), frame.begin(), frame.end());
}

/** Deterministic projected frames of a model's own joint width: a same-weight differential needs no other input. */
std::vector<float> synthetic_frames(int64_t frames, int64_t hidden) {
    std::vector<float> out((size_t) (frames * hidden));
    for (int64_t t = 0; t < frames; t++) {
        for (int64_t i = 0; i < hidden; i++) {
            out[(size_t) (t * hidden + i)] =
                0.6f * std::sin(0.013f * (float) (i + 37 * t)) + 0.3f * std::cos(0.0071f * (float) (i - 11 * t));
        }
    }
    return out;
}

/** The scenarios of the generated fixture, where crafted frames decide every token and duration. */
void fixture_scenarios(Verdict & v, const Rig & rig, observe::Observer & observer) {
    const int64_t hidden = rig.joint.hidden();
    const int32_t blank = rig.blank;
    const int32_t A = 3, B = 7, C = 5, X = 11, D = 9;
    const float win = 5.0f, near = 4.6f;
    std::printf("fixture: joint hidden %lld, prediction state %lld floats, outputs %d, blank %d, durations 0 1 2 3 4, "
                "max_symbols 2\n",
                (long long) hidden, (long long) rig.prediction.state_size(), rig.outputs, (int) blank);

    {
        std::vector<float> projected;
        for (int t = 0; t < 8; t++) append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        const std::vector<int32_t> want(8, A);
        const BoostingTree tree({{A, D}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = e.boost_ids = &want;
        e.matches = 8;
        e.entries = 8;
        e.divergence = -1;
        e.any_match = true;
        scenario(v, rig, observer, "no divergence, the wanted token carried by a large margin", projected, Boost{tree, 1.0f}, e);
    }
    {
        std::vector<float> projected;
        append_frame(projected, fixture_frame(hidden, blank, A, win, B, near, 1));
        for (int t = 1; t < 8; t++) append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        const std::vector<int32_t> plain(8, A);
        std::vector<int32_t> boosted{B};
        boosted.insert(boosted.end(), 7, A);
        const BoostingTree tree({{B, A}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = &plain;
        e.boost_ids = &boosted;
        e.matches = 1;
        e.entries = 8;
        e.divergence = 0;
        e.any_match = true;
        scenario(v, rig, observer, "a first-token divergence, the initial blank still borrowed", projected, Boost{tree, 1.0f}, e);
    }
    {
        std::vector<float> projected;
        for (int t = 0; t < 2; t++) append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        append_frame(projected, fixture_frame(hidden, blank, C, win, X, near, 1));
        for (int t = 3; t < 8; t++) append_frame(projected, fixture_frame(hidden, blank, D, win, -1, 0.0f, 1));
        const std::vector<int32_t> plain{A, A, C, D, D, D, D, D};
        const std::vector<int32_t> boosted{A, A, X, D, D, D, D, D};
        const BoostingTree tree({{X, D}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = &plain;
        e.boost_ids = &boosted;
        e.matches = 3;
        e.entries = 8;
        e.divergence = 2;
        e.any_match = true;
        scenario(v, rig, observer, "a middle divergence whose later tokens coincide without rejoining", projected, Boost{tree, 1.0f}, e);
    }
    {
        std::vector<float> projected;
        for (int t = 0; t < 3; t++) append_frame(projected, fixture_frame(hidden, blank, blank, win, -1, 0.0f, 1));
        append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 2));
        append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        const std::vector<int32_t> want{A};
        const BoostingTree tree({{A, D}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = e.boost_ids = &want;
        e.matches = 1;
        e.entries = 1;
        e.divergence = -1;
        e.any_match = true;
        scenario(v, rig, observer, "three blanks in a row, then a token whose duration ends the utterance", projected,
                 Boost{tree, 1.0f}, e);
    }
    {
        std::vector<float> projected;
        append_frame(projected, fixture_frame(hidden, blank, blank, win, -1, 0.0f, 0));
        append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        const std::vector<int32_t> want{A};
        const BoostingTree tree({{A, D}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = e.boost_ids = &want;
        e.matches = 1;
        e.entries = 1;
        e.divergence = -1;
        e.any_match = true;
        scenario(v, rig, observer, "a blank duration of 0 that becomes 1", projected, Boost{tree, 1.0f}, e);
    }
    {
        std::vector<float> projected;
        for (int t = 0; t < 2; t++) append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 0));
        const std::vector<int32_t> want{A, A, A, A};
        const BoostingTree tree({{A, A}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = e.boost_ids = &want;
        e.matches = 4;
        e.entries = 4;
        e.divergence = -1;
        e.any_match = true;
        scenario(v, rig, observer, "nonblank durations of 0 with max_symbols forcing the frame on", projected, Boost{tree, 1.0f}, e);
    }
    {
        const BoostingTree tree({{A, D}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.matches = 0;
        e.entries = 0;
        e.divergence = -1;
        scenario(v, rig, observer, "empty input", std::vector<float>(), Boost{tree, 1.0f}, e);
    }
    {
        std::vector<float> projected;
        for (int t = 0; t < 2; t++) append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        append_frame(projected, fixture_frame(hidden, blank, blank, win, -1, 0.0f, 1));
        const std::vector<int32_t> want{A, A};
        const BoostingTree tree({{A, D}}, blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.plain_ids = e.boost_ids = &want;
        e.matches = 3;
        e.entries = 3;
        e.divergence = -1;
        e.any_match = true;
        scenario(v, rig, observer, "a blank on the last frame ends the decoding", projected, Boost{tree, 1.0f}, e);
    }
    for (int frames : {63, 64, 65}) {
        std::vector<float> projected;
        for (int t = 0; t < frames; t++) append_frame(projected, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
        const std::vector<int32_t> want((size_t) frames, A);
        const BoostingTree tree({{A, D}}, blank, 2.0, 1.0, 0.0f);
        const int64_t entries = (int64_t) std::min<size_t>((size_t) frames, KeywordPrefix::kCapacity);
        std::string name = "capacity: " + std::to_string(frames) + " predictor positions, one token of duration 1 a frame";
        if (frames == 65) name += ", the last beyond the prefix's capacity";
        Expect e;
        e.plain_ids = e.boost_ids = &want;
        e.matches = entries;
        e.entries = entries;
        e.divergence = -1;
        e.any_match = true;
        e.payload = true;
        scenario(v, rig, observer, name, projected, Boost{tree, 1.0f}, e);
    }

    std::vector<float> middle;
    for (int t = 0; t < 2; t++) append_frame(middle, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
    append_frame(middle, fixture_frame(hidden, blank, C, win, X, near, 1));
    for (int t = 3; t < 8; t++) append_frame(middle, fixture_frame(hidden, blank, D, win, -1, 0.0f, 1));
    const BoostingTree middle_tree({{X, D}}, blank, 2.0, 1.0, 0.0f);
    cancelled_record_check(v, rig, observer, middle, Boost{middle_tree, 1.0f}, 3);
    cancelled_reuse_check(v, rig, observer, middle, Boost{middle_tree, 1.0f}, 5);

    std::vector<float> history;
    for (int t = 0; t < 8; t++) append_frame(history, fixture_frame(hidden, blank, A, win, -1, 0.0f, 1));
    recorded_entries_check(v, rig, history, "the all-same-token recording");
    history_sensitivity_check(v, rig, /* first */ 3, /* second */ 7, /* last */ 5);
}

/** The same differential on a real Haas file's own weights, whose synthetic frames decide what they decide. */
void model_scenarios(Verdict & v, const Rig & rig, observe::Observer & observer) {
    std::printf("model: %s, %s weights, blank %d, outputs %d, joint hidden %lld, prediction state %lld floats\n",
                rig.model.str("general.name").c_str(), read_identity(rig.model).weight_type.c_str(), (int) rig.blank, rig.outputs,
                (long long) rig.joint.hidden(), (long long) rig.prediction.state_size());
    if (rig.blank < 8) {
        v.fail("the model's vocabulary is too small for the trees the check spells");
        return;
    }
    const int64_t hidden = rig.joint.hidden();
    const std::vector<int64_t> lengths{48, 360, 1440};
    for (size_t k = 0; k < lengths.size(); k++) {
        const std::vector<float> projected = synthetic_frames(lengths[k], hidden);
        const std::vector<std::vector<int32_t>> words = k == 0 ? std::vector<std::vector<int32_t>>{{0, 1}}
                                                               : std::vector<std::vector<int32_t>>{{2, 3, 4}};
        const BoostingTree tree(words, rig.blank, 2.0, 1.0, 0.0f);
        Expect e;
        e.any_match = true;
        e.payload = k == lengths.size() - 1;
        scenario(v, rig, observer, "synthetic frames of " + std::to_string(lengths[k]), projected, Boost{tree, 2.0f}, e);
        if (k == 1) {
            recorded_entries_check(v, rig, projected, "the model's own weights");
            history_sensitivity_check(v, rig, /* first */ 0, /* second */ 1, /* last */ 2);
        }
    }
    const std::vector<float> medium = synthetic_frames(360, hidden);
    const BoostingTree tree({{2, 3, 4}}, rig.blank, 2.0, 1.0, 0.0f);
    cancelled_record_check(v, rig, observer, medium, Boost{tree, 2.0f}, 6);
    cancelled_reuse_check(v, rig, observer, medium, Boost{tree, 2.0f}, 9);
    const std::vector<float> few = synthetic_frames(16, hidden);
    TdtDecoder other(rig.model, rig.prediction, rig.joint);
    owner_refusals_check(v, rig, observer, few, Boost{tree, 2.0f}, other, nullptr, nullptr);
}

int finish(Verdict & v, const std::string & what) {
    if (v.ok) {
        std::printf("%s: OK\n", what.c_str());
        return 0;
    }
    std::printf("%s: FAILED\n", what.c_str());
    return 1;
}

/**
 * Creates the caller's fresh output directory, which must not exist, writes the fixture's fixed filename into it and
 * runs the crafted scenarios, the capacity boundaries and the owner refusals. The directory and its fixture stay the
 * caller's; the writer overwrites, so the check refuses any path that exists.
 */
int fixture_mode(const std::string & out, const std::string & device) {
    const std::filesystem::path root = std::filesystem::u8path(out);
    std::error_code ec;
    const bool created = std::filesystem::create_directory(root, ec);
    if (ec || !created) {
        throw Error(Fault::InvalidArgument, "refusing to write the fixture into " + out +
                    ": the directory exists or cannot be created; give a fresh directory that does not exist", "--fixture");
    }
    const std::filesystem::path file = root / "003-state-fixture.gguf";
    state_check::write_fixture(file.u8string());
    BackendOwner plain(four_thread_backend(device));
    std::printf("backend: %s, computing on %d CPU threads where the device is the CPU\n", ggml_backend_name(plain.get()), kThreads);
    Verdict v;
    {
        ModelFile model(file.u8string(), plain.get(), state_check::kLayout);
        PredictionNetwork prediction(model);
        Joint joint(model);
        std::unique_ptr<TdtDecoder> decoder = std::make_unique<TdtDecoder>(model, prediction, joint);
        const int32_t blank = decoder->blank();
        Rig rig{model, prediction, joint, std::move(decoder), blank, joint.outputs(), plain.get()};
        if (rig.blank != state_check::kBlank || rig.outputs != state_check::kOutputs || joint.hidden() != state_check::kHidden ||
            prediction.state_size() != 46720) {
            v.fail("the fixture the check loaded is not the one 003-state-fixture.h writes");
        }
        observe::Observer observer(plain.get(), joint.outputs());
        fixture_scenarios(v, rig, observer);
        {
            BackendOwner second(four_thread_backend(device));
            {
                ModelFile other_model(file.u8string(), second.get(), state_check::kLayout);
                PredictionNetwork other_prediction(other_model);
                Joint other_joint(other_model);
                TdtDecoder foreign_decoder(other_model, other_prediction, other_joint);
                std::vector<float> few;
                for (int t = 0; t < 4; t++) append_frame(few, fixture_frame(joint.hidden(), rig.blank, 3, 5.0f, -1, 0.0f, 1));
                const BoostingTree tree({{3, 9}}, rig.blank, 2.0, 1.0, 0.0f);
                KeywordPrefix foreign;
                foreign_decoder.record_keyword_prefix(few, second.get(), DecodingProgress(), foreign);
                TdtDecoder other(model, prediction, joint);
                owner_refusals_check(v, rig, observer, few, Boost{tree, 1.0f}, other, &foreign_decoder, &foreign);
            }
        }
    }
    return finish(v, "the fixture's keyword prediction prefix");
}

/** Loads the model and runs the differential on its own weights with synthetic frames. */
int model_mode(const std::string & path, const std::string & device) {
    BackendOwner plain(four_thread_backend(device));
    std::printf("backend: %s, computing on %d CPU threads where the device is the CPU\n", ggml_backend_name(plain.get()), kThreads);
    Verdict v;
    {
        Recognizer recognizer(path, plain.get(), kThreads);
        if (recognizer.model().str("general.name") != "Haas-ja-v1.0" ||
            recognizer.model().str("fastconformer.decoder.prediction") != "kda" ||
            recognizer.model().str("fastconformer.decoder.kind") != "tdt") {
            throw Error(Fault::InvalidArgument, path + " is not Haas-ja-v1.0 with a KDA prediction network and a TDT"
                        " decoder, the weights this check's real-model mode is for; the fixture mode covers the rest");
        }
        std::unique_ptr<TdtDecoder> decoder = std::make_unique<TdtDecoder>(recognizer.model(), recognizer.prediction(), recognizer.joint());
        const int32_t blank = decoder->blank();
        Rig rig{recognizer.model(), recognizer.prediction(), recognizer.joint(), std::move(decoder), blank,
                recognizer.joint().outputs(), plain.get()};
        observe::Observer observer(plain.get(), recognizer.joint().outputs());
        model_scenarios(v, rig, observer);
    }
    return finish(v, "the model's keyword prediction prefix");
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    const char * usage = "usage: %s --fixture <fresh-output-directory> [gpu|cpu|device name]\n"
                         "       %s <model.gguf> [gpu|cpu|device name]\n";
    if (args.size() < 2) {
        std::fprintf(stderr, usage, args[0].c_str(), args[0].c_str());
        return 2;
    }
    try {
        if (args[1] == "--fixture") {
            if (args.size() < 3 || args.size() > 4) {
                std::fprintf(stderr, usage, args[0].c_str(), args[0].c_str());
                return 2;
            }
            return fixture_mode(args[2], args.size() > 3 ? args[3] : "");
        }
        if (args.size() > 3) {
            std::fprintf(stderr, usage, args[0].c_str(), args[0].c_str());
            return 2;
        }
        return model_mode(args[1], args.size() > 2 ? args[2] : "");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
