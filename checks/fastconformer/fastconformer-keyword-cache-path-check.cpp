// Checks cached and uncached keyword preparation through real raw decodings, WordsModel and the family engine.
// Counters are local to this check. This is a same-weight differential check, not an official-reference or timing run.
// Usage: fastconformer-keyword-cache-path-check <model.gguf> <audio.wav> [cpu|gpu|device] [cpu|device]

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "args.h"
#include "backend.h"
#include "engine.h"
#include "error.h"
#include "fastconformer/guard.h"
#include "fastconformer/layout.h"
#include "fastconformer/recognizer.h"
#include "keyword-cache-check.h"
#include "resample.h"
#include "wav.h"

using namespace fastconformer;
using namespace keyword_check;

namespace {

const char * const kFlag = "SPEECH_HAAS_KEYWORD_CACHE";
const std::string kA = "Haas=ハース,はーす\nAIモデル";
const std::string kB = "音声認識=おんせいにんしき\n日本語=にほんご";

void setting(const char * value) {
#ifdef _WIN32
    const int status = _putenv_s(kFlag, value ? value : "");
#else
    const int status = value ? setenv(kFlag, value, 1) : unsetenv(kFlag);
#endif
    if (status != 0) throw std::runtime_error("cannot set the keyword cache selector");
}

class Progress : public Run {
public:
    explicit Progress(double limit = 2.0) : limit_(limit) {}
    bool audio(const float *, size_t) override { throw std::logic_error("recognition passed audio"); }
    bool progress(double done) override {
        ++calls;
        if (done >= limit_) cancelled_ = true;
        return !cancelled_;
    }
    bool stopped() override { return cancelled_; }
    size_t calls = 0;
private:
    double limit_;
    bool cancelled_ = false;
};

void recognized(const Recognized & got, const Recognized & want, Checks & check) {
    check.expect(got.text == want.text && got.untimed == want.untimed && got.stop == want.stop &&
                 got.languages == want.languages && got.speakers == want.speakers, "The engine's recognition differs.");
    const auto timed = [&](const std::vector<TimedText> & a, const std::vector<TimedText> & b) {
        check.expect(a.size() == b.size(), "The engine's timed item count differs.");
        for (size_t i = 0; i < a.size(); ++i) {
            check.expect(a[i].text == b[i].text && std::memcmp(&a[i].start, &b[i].start, sizeof(double)) == 0 &&
                         std::memcmp(&a[i].end, &b[i].end, sizeof(double)) == 0, "An engine token or segment differs.");
        }
    };
    timed(got.tokens, want.tokens);
    timed(got.segments, want.segments);
}

void engine_path(const std::string & path, ggml_backend_t backend, const std::vector<float> & audio, DecodeOn on, Checks & check) {
    setting("0");
    auto off = load_fastconformer(path, backend, 4);
    setting("1");
    auto cached = load_fastconformer(path, backend, 4);
    const auto file = std::make_shared<ModelFile>(path, layout);
    RequestValues defaults;
    for (const auto & option : describe_fastconformer(file).options) {
        if (option.default_value) defaults.set(option.option, *option.default_value, false);
    }
    defaults.set(SPEECH_OPT_DECODE_ON_CPU, on == DecodeOn::Cpu, true);
    const auto request = [&](const std::string & raw, bool guard = true, bool timestamps = false) {
        auto values = defaults;
        values.set(SPEECH_OPT_KEYWORDS, raw, true);
        if (!raw.empty()) values.set(SPEECH_OPT_KEYWORDS_GUARD, guard, true);
        values.set(SPEECH_OPT_TIMESTAMPS, timestamps, true);
        return values;
    };
    const auto pair = [&](const RequestValues & values, const std::vector<float> & samples) {
        Progress left, right;
        const auto want = off->transcribe(samples, values, left);
        const auto got = cached->transcribe(samples, values, right);
        check.expect(!left.stopped() && !right.stopped(), "A complete engine request was cancelled.");
        recognized(got, want, check);
    };
    pair(request(kA), audio);
    pair(request(kA), audio);
    pair(request(kA, false), audio);
    std::vector<std::pair<RequestValues, std::string>> invalid;
    auto values = request("");
    values.set(SPEECH_OPT_KEYWORDS_ALPHA, 2.0, true);
    invalid.push_back({values, "keywords_alpha"});
    values = request("");
    values.set(SPEECH_OPT_KEYWORDS_GUARD, false, true);
    invalid.push_back({values, "keywords_guard"});
    invalid.push_back({request(" \n\t\n"), "keywords"});
    invalid.push_back({request("=はーす"), "keywords"});
    invalid.push_back({request(kA, true, true), "timestamps"});
    for (const auto & item : invalid) {
        for (Engine * engine : {off.get(), cached.get()}) {
            Progress progress;
            bool refused = false;
            try { engine->transcribe(audio, item.first, progress); }
            catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument && error.input() == item.second; }
            check.expect(refused && progress.calls == 0, "An invalid request did not retain its original early error.");
        }
    }
    pair(request(kA), audio);
    for (const auto & raw : {kB, kA}) {
        for (Engine * engine : {off.get(), cached.get()}) {
            Progress cancelled(0.75);
            const auto result = engine->transcribe(audio, request(raw), cancelled);
            check.expect(cancelled.stopped() && result.text.empty(), "The keyword request did not cancel through the engine.");
        }
        pair(request(kA), audio);
    }
    pair(request("", true, true), audio);
    pair(request(kA), std::vector<float>(audio.size(), 0.0f));
    setting("invalid");
    bool refused = false;
    try { const auto invalid_engine = load_fastconformer(path, backend, 4); }
    catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
    check.expect(refused, "The Haas engine accepted an invalid cache selector.");
}

void raw_path(Recognizer & recognizer, const std::vector<float> & audio, DecodeOn on, Checks & check) {
    const WordsModel & words = *recognizer.words();
    const auto & frontend = recognizer.frontend();
    const auto projected = recognizer.encode(frontend.features(audio), frontend.frames(audio.size()));
    const auto name = recognizer.decodings().front();
    const auto plain = recognizer.decoding(projected, name, on);
    check.expect(has_speech(audio, words.speech_rms(), (size_t) words.speech_frame()), "Give speech audio for the cache-hit path.");
    struct Case { std::string raw; float alpha; bool guard; };
    const std::vector<Case> cases = {
        {kA, 2.0f, true}, {kA, 2.0f, true}, {kA, 2.0f, false}, {kB, 2.0f, true}, {kA, 2.0f, true},
        {kA, 1.5f, true}, {"Haas=ハース\nHaas=はーす", 2.0f, true},
        {"Haas=はーす\nHaas=ハース", 2.0f, true},
        {"Cafe\xCC\x81=カフェ\nCafé=かふぇ", 2.0f, true}, {kA, 2.0f, true},
    };
    WordsCache cache(true);
    Counters counts;
    for (size_t i = 0; i < cases.size(); ++i) {
        const auto & item = cases[i];
        std::unique_ptr<PreparedWords> pending;
        auto & prepared = prepare(cache, item.raw, item.alpha, pending, counts);
        check.expect(bool(pending) == (i != 1 && i != 2), "The expected cache hit/miss did not occur.");
        const Vocabulary reference(word_entries(item.raw), item.alpha);
        vocabulary(prepared.vocabulary, reference, check);
        const auto want = words.recognize(recognizer, reference, item.guard, audio, projected, name, on, {});
        const bool had_tree = bool(prepared.boosting);
        const auto got = words.recognize(recognizer, prepared, item.guard, audio, projected, name, on, {});
        if (!had_tree && prepared.boosting) ++counts.trees;
        words_text(got, want, check);
        check.expect(bool(prepared.boosting), "The actual keyword path did not construct its tree.");
        const auto reference_tree = words.tree(reference);
        tree(*prepared.boosting, reference_tree, check);
        const auto expected_ids = recognizer.decoding(projected, name, on, {}, Boost{reference_tree, item.alpha});
        const auto cached_ids = recognizer.decoding(projected, name, on, {}, Boost{*prepared.boosting, item.alpha});
        decoding(cached_ids, expected_ids, check);
        decoding(recognizer.decoding(projected, name, on), plain, check);
        std::printf("case %zu: raw_bytes=%zu entries=%zu tree_payload=%zu hit=%s\n", i, prepared.raw.size(),
                    prepared.entry_count, prepared.tree_bytes(), pending ? "no" : "yes");
        const bool kept = publish(cache, pending, counts);
        check.expect(kept == (i != 1 && i != 2), "A completed small preparation was not published as expected.");
        if (i == 2) check.expect(counts.preparations == 1 && counts.trees == 1 && counts.publications == 1,
                                 "A repeat or guard change reconstructed a cached preparation.");
    }
    {
        std::unique_ptr<PreparedWords> pending;
        auto & prepared = prepare(cache, "Cancelled=キャンセル", 2.0f, pending, counts);
        bool cancelled = false;
        const bool had_tree = bool(prepared.boosting);
        words.recognize(recognizer, prepared, true, audio, projected, name, on, [&](double done) {
            if (done >= 0.25) cancelled = true;
            return !cancelled;
        });
        if (!had_tree && prepared.boosting) ++counts.trees;
        check.expect(cancelled && pending, "The cancelled miss did not stay request-owned.");
    }
    for (const double limit : {0.0, 0.5}) {
        std::unique_ptr<PreparedWords> pending;
        auto & prepared = prepare(cache, kA, 2.0f, pending, counts);
        check.expect(!pending && prepared.boosting, "Cancellation did not start from a cached hit.");
        const auto before = counts;
        const auto saved_tree = *prepared.boosting;
        const Vocabulary reference(word_entries(kA), 2.0f);
        const size_t payload = prepared.tree_bytes();
        bool cancelled = false;
        words.recognize(recognizer, prepared, true, audio, projected, name, on, [&](double done) {
            if (done >= limit) cancelled = true;
            return !cancelled;
        });
        check.expect(cancelled, "The cached hit did not cancel at the selected decoding pass.");
        vocabulary(prepared.vocabulary, reference, check);
        tree(*prepared.boosting, saved_tree, check);
        auto & retry = prepare(cache, kA, 2.0f, pending, counts);
        check.expect(!pending && &retry == &prepared && retry.tree_bytes() == payload,
                     "Cancellation replaced or resized the cached entry.");
        const auto want = words.recognize(recognizer, reference, true, audio, projected, name, on, {});
        words_text(words.recognize(recognizer, retry, true, audio, projected, name, on, {}), want, check);
        check.expect(counts.preparations == before.preparations && counts.trees == before.trees &&
                     counts.publications == before.publications, "A cancelled hit or retry reconstructed the cache.");
    }
    const std::vector<float> silent(audio.size(), 0.0f);
    const auto silence = recognizer.encode(frontend.features(silent), frontend.frames(silent.size()));
    for (const auto & raw : {std::string("Silence=サイレンス"), kA}) {
        std::unique_ptr<PreparedWords> pending;
        auto & prepared = prepare(cache, raw, 2.0f, pending, counts);
        const bool hit = !pending, had_tree = bool(prepared.boosting);
        const Vocabulary reference(word_entries(raw), 2.0f);
        const auto want = words.recognize(recognizer, reference, true, silent, silence, name, on, {});
        const auto got = words.recognize(recognizer, prepared, true, silent, silence, name, on, {});
        words_text(got, want, check);
        check.expect(!got.speech && got.boosted.empty() && bool(prepared.boosting) == had_tree,
                     "Silence constructed or used a boosted path.");
        check.expect(hit == (raw == kA), "Cancellation or silence replaced the last completed keyword entry.");
        check.expect(!publish(cache, pending, counts), "A silent miss published a treeless entry.");
    }
    std::printf("raw path: preparations=%zu trees=%zu publications=%zu; counters live only in this check\n",
                counts.preparations, counts.trees, counts.publications);
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        if (args.size() < 3 || args.size() > 5) {
            throw std::runtime_error("usage: fastconformer-keyword-cache-path-check <model.gguf> <audio.wav> [cpu|gpu|device] [cpu|device]");
        }
        const std::string place = args.size() > 4 ? args[4] : "cpu";
        if (place != "cpu" && place != "device") throw std::runtime_error("the decoding place must be cpu or device");
        const DecodeOn on = place == "cpu" ? DecodeOn::Cpu : DecodeOn::Device;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args.size() > 3 ? args[3] : "gpu"), 4),
                                                                         ggml_backend_free);
        Checks check;
        std::vector<float> audio;
        {
            Recognizer recognizer(args[1], backend.get(), 4);
            if (!recognizer.words() || recognizer.model().str("general.name") != "Haas-ja-v1.0" ||
                recognizer.model().str("fastconformer.decoder.prediction") != "kda") {
                throw std::runtime_error("the path check requires Haas KDA with custom vocabulary support");
            }
            const auto wav = read_wav(args[2]);
            audio = Resampler(wav.sample_rate, recognizer.sample_rate())(wav.mono());
            raw_path(recognizer, audio, on, check);
        }
        engine_path(args[1], backend.get(), audio, on, check);
        std::printf("keyword cache path: %zu checks; backend=%s decode=%s threads=4; official reference, original binary, timing and other backends are not checked\n",
                    check.total, ggml_backend_name(backend.get()), place.c_str());
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
