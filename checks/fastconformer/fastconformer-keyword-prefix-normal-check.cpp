// The normal production path of the keyword prediction prefix, as a check: Haas-ASR's custom vocabulary through
// WordsModel::recognize(), exactly as the engine runs it, on a real model file the caller gives and non-silent
// speech, watched from the side by decoding-observer.h. The tool uses only the APIs the original base of this
// experiment already had, so the same source builds there: there the environment selector does nothing, and actual
// use of the prefix is shown by comparing the graph kinds of three processes — an original-base build, this one with
// SPEECH_HAAS_KEYWORD_PREFIX unset or 0, and this one with it 1 — on the same model, audio, keywords and arguments.
// The selector is set (or read, where --prefix is not given) before any model loads, for the runtime reads it once,
// as the model loads. One recognition runs unobserved on the plain backend and one observed through the wrapper, in
// sequential scopes so the two full models never live together; the audio goes through the real frontend and encoder
// each time, and the observer is cleared after the encoder so that only the decoding is counted, the encoder's
// graphs left out of the decoding kinds. The observed run reports no timing, and nothing of the model or the audio
// beyond the recognition outputs the comparison asks for.
//
// The tool writes one artifact file of little-endian sections, robust to truncation: an 8-byte magic "SPKWP028", a
// u32 format version, then sections of a u32 tag, a u64 payload length and that many payload bytes, ending with the
// tag 0 and an empty payload and nothing after it. Tag 1 holds the request's settings as one line of JSON
// (selector, backend, device, threads, decode_on, model, weights, blank, outputs, duration_slots, hidden, alpha,
// guard, the keyword entries, the wav's name, its rate and samples, the projected frames, the decoding's name).
// Tags 2 and 3 hold the observed and the unobserved words: a speech byte and three u32-length UTF-8 strings, the
// plain, the boosted and the formatted text. Tags 4 and 5 hold the observed and the unobserved progress, a u64
// count and that many f64. Tag 6 holds the observed graphs: u64 counts of prediction graphs, joint-only graphs and
// others, and the u64 index of the logits row where the boosted pass begins, the first progress call at a half or
// more, or all-ones where no boosted pass ran. Tag 7 holds the graph kinds in order, a u64 count and one byte a
// graph, 1 a prediction graph and 2 a joint-only one. Tag 8 holds every joint logits row the decoding consumed, in
// order: a u64 row count, then each row's outputs f32 bits unchanged. experiments/haas-mac/028_compare_keyword_prefix.py
// reads this format and fails on any difference the experiment does not approve.
//
// usage: fastconformer-keyword-prefix-normal-check <model.gguf> <speech.wav> <output file> [gpu|cpu|device name]
//                                           [--keywords-file <file>] [--alpha <number>] [--guard 0|1]
//                                           [--decode cpu|device] [--prefix 0|1]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "args.h"
#include "backend.h"
#include "decoding-observer.h"
#include "error.h"
#include "wav.h"

#include "fastconformer/recognizer.h"
#include "fastconformer/words.h"

using namespace fastconformer;

namespace {

constexpr char kMagic[8] = {'S', 'P', 'K', 'W', 'P', '0', '2', '8'};
constexpr uint32_t kFormat = 1;
constexpr uint32_t kTagEnd = 0, kTagMeta = 1, kTagText = 2, kTagQuietText = 3, kTagProgress = 4, kTagQuietProgress = 5,
                  kTagGraphs = 6, kTagKinds = 7, kTagLogits = 8;
constexpr uint64_t kNoBoundary = std::numeric_limits<uint64_t>::max();
constexpr int kThreads = 4;

void put_u32(std::vector<uint8_t> & out, uint32_t v) {
    for (int i = 0; i < 4; i++) out.push_back((uint8_t) (v >> (8 * i)));
}

void put_u64(std::vector<uint8_t> & out, uint64_t v) {
    for (int i = 0; i < 8; i++) out.push_back((uint8_t) (v >> (8 * i)));
}

void put_f32(std::vector<uint8_t> & out, float v) {
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    put_u32(out, bits);
}

void put_f64(std::vector<uint8_t> & out, double v) {
    uint64_t bits = 0;
    std::memcpy(&bits, &v, 8);
    put_u64(out, bits);
}

void put_str(std::vector<uint8_t> & out, const std::string & text) {
    put_u32(out, (uint32_t) text.size());
    out.insert(out.end(), text.begin(), text.end());
}

void write_section(std::ofstream & file, uint32_t tag, const std::vector<uint8_t> & payload) {
    std::vector<uint8_t> head;
    put_u32(head, tag);
    put_u64(head, payload.size());
    file.write((const char *) head.data(), (std::streamsize) head.size());
    file.write((const char *) payload.data(), (std::streamsize) payload.size());
}

/** `text` as a JSON string literal, escaped as JSON escapes it; the bytes above 0x7F are UTF-8 already. */
std::string json_string(const std::string & text) {
    std::string out = "\"";
    char escape[8];
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += (char) c;
        } else if (c < 0x20) {
            std::snprintf(escape, sizeof escape, "\\u%04x", (unsigned) c);
            out += escape;
        } else {
            out += (char) c;
        }
    }
    return out + "\"";
}

std::string json_number(double v) {
    char out[40];
    std::snprintf(out, sizeof out, "%.9g", v);
    return out;
}

/** What one recognition gave, watched or not. */
struct Capture {
    WordsText text;
    std::vector<double> progress;
    size_t steps = 0, joints = 0, others = 0, broken = 0;
    uint64_t boundary = kNoBoundary;
    std::vector<uint8_t> kinds;
    std::vector<std::vector<float>> logits;
};

/** One unobserved recognition, on the backend the recognizer holds, collecting the texts and the progress alone. */
Capture quiet_recognition(Recognizer & r, const Vocabulary & vocabulary, bool guard, const std::vector<float> & samples,
                          const std::vector<float> & projected, const std::string & name, DecodeOn on) {
    Capture cap;
    DecodingProgress tell = [&](double done) {
        cap.progress.push_back(done);
        return true;
    };
    cap.text = r.words()->recognize(r, vocabulary, guard, samples, projected, name, on, tell);
    return cap;
}

/**
 * One observed recognition. The observer must already be cleared of the encoder's graphs: from here on it sees the
 * two decodings alone. The progress callback records where the boosted pass's graphs begin, for WordsModel::recognize
 * maps the plain decoding's calls to [0, 1/2) and the boosted one's to [1/2, 1), so the first call at a half or more
 * stands between them.
 */
Capture observed_recognition(Recognizer & r, const Vocabulary & vocabulary, bool guard, const std::vector<float> & samples,
                             const std::vector<float> & projected, const std::string & name, DecodeOn on,
                             observe::Observer & observer, int64_t outputs) {
    Capture cap;
    DecodingProgress tell = [&](double done) {
        cap.progress.push_back(done);
        if (done >= 0.5 && cap.boundary == kNoBoundary) cap.boundary = observer.seen().size();
        return true;
    };
    cap.text = r.words()->recognize(r, vocabulary, guard, samples, projected, name, on, tell);
    for (const observe::SeenGraph & g : observer.seen()) {
        if (g.kind == observe::SeenGraph::Kind::Step) {
            cap.steps++;
            cap.kinds.push_back(1);
        } else if (g.kind == observe::SeenGraph::Kind::Joint) {
            cap.joints++;
            cap.kinds.push_back(2);
        } else {
            cap.others++;
            continue;
        }
        if (g.matched != 1 || g.logits.size() != (size_t) outputs) cap.broken++;
        else cap.logits.push_back(g.logits);
    }
    return cap;
}

std::vector<uint8_t> text_payload(const WordsText & words) {
    std::vector<uint8_t> out;
    out.push_back(words.speech ? 1 : 0);
    put_str(out, words.plain);
    put_str(out, words.boosted);
    put_str(out, words.text);
    return out;
}

std::vector<uint8_t> progress_payload(const std::vector<double> & progress) {
    std::vector<uint8_t> out;
    put_u64(out, progress.size());
    for (double done : progress) put_f64(out, done);
    return out;
}

/** Reads a whole file as bytes; a path that cannot be opened throws. */
std::string read_file(const std::string & path, const char * option) {
    std::ifstream in(std::filesystem::u8path(path), std::ios::binary);
    if (!in) throw Error(Fault::Io, "cannot open " + path, option);
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (in.bad()) throw Error(Fault::Io, "cannot read " + path, option);
    return text;
}

/** Sets SPEECH_HAAS_KEYWORD_PREFIX to `value` for this process, overwriting what it held; standard C++ has no
 *  environment setter, so the one each platform spells its own is used. */
void set_selector(const std::string & value) {
#if defined(_WIN32)
    const errno_t set = _putenv_s("SPEECH_HAAS_KEYWORD_PREFIX", value.c_str());
#else
    const int set = ::setenv("SPEECH_HAAS_KEYWORD_PREFIX", value.c_str(), 1);
#endif
    if (set != 0) throw Error(Fault::Io, "cannot set SPEECH_HAAS_KEYWORD_PREFIX", "--prefix");
}

struct Options {
    std::string model, wav, out, device = "cpu", keywords_file, decode = "device", prefix;
    double alpha = 0;
    bool guard = false;
};

int normal_mode(const Options & o) {
    std::error_code ec;
    if (std::filesystem::exists(std::filesystem::u8path(o.out), ec) || ec) {
        throw Error(Fault::InvalidArgument, "refusing to write the artifact to " + o.out + ": the path exists or"
                    " cannot be checked; give a fresh path, for the artifact must not overwrite an input file or an"
                    " earlier run's artifact", "output");
    }
    const std::vector<std::string> entries = word_entries(read_file(o.keywords_file, "--keywords-file"));
    if (entries.empty()) {
        throw Error(Fault::InvalidArgument, "the keywords file holds no entry; give one written form=reading a line",
                    "--keywords-file");
    }

    // The runtime reads SPEECH_HAAS_KEYWORD_PREFIX once, as the model loads, so it is set (or read as it stands)
    // before the first Recognizer of this process is constructed.
    std::string selector = o.prefix;
    if (!selector.empty()) {
        set_selector(selector);
    } else {
        const char * given = std::getenv("SPEECH_HAAS_KEYWORD_PREFIX");
        selector = given ? given : "unset";
    }

    start_ggml();
    ggml_backend_t plain = start_device(find_device(o.device), kThreads);
    const bool cpu = ggml_backend_dev_type(ggml_backend_get_device(plain)) == GGML_BACKEND_DEVICE_TYPE_CPU;
    const std::string backend_name = ggml_backend_name(plain), backend_kind = device_kind(plain);
    if (o.decode == "cpu" && !cpu) {
        ggml_backend_free(plain);
        throw Error(Fault::InvalidArgument, "the CPU decoding of a model on another device runs on a backend the"
                    " library makes inside itself, which no check-side observer can watch without changing production"
                    " code; this case belongs to its own entry-point differential run", "--decode");
    }
    std::printf("requested selector: SPEECH_HAAS_KEYWORD_PREFIX=%s\nbackend: %s (%s), decoding on %s, four CPU threads"
                " where the device is the CPU\n",
                selector.c_str(), backend_name.c_str(), backend_kind.c_str(), cpu ? "the CPU" : "the device");

    const Wav wav = read_wav(o.wav);

    Capture quiet;
    std::string model_name, weights, decoding_name;
    int64_t blank = 0, outputs = 0, hidden = 0, frames = 0;
    float alpha = 0;
    {
        Recognizer r(o.model, plain, kThreads);
        if (!r.words()) {
            throw Error(Fault::File, o.model + " offers no Haas-ASR custom vocabulary, which this tool drives", "model");
        }
        if (r.model().str("fastconformer.decoder.kind") != "tdt") {
            throw Error(Fault::File, o.model + " holds no TDT decoder, the only one that boosts words", "model");
        }
        model_name = r.model().str("general.name");
        weights = read_identity(r.model()).weight_type;
        decoding_name = r.decodings().front();
        blank = r.decoder(decoding_name).blank();
        outputs = r.joint().outputs();
        hidden = r.joint().hidden();
        alpha = o.alpha > 0 ? (float) o.alpha : r.words()->alpha();
        if (!(alpha > 0)) {
            throw Error(Fault::InvalidArgument, "the model's own vocabulary alpha is not positive; give --alpha",
                        "--alpha");
        }
        if (selector == "1" && (model_name != "Haas-ja-v1.0" || r.model().str("fastconformer.decoder.prediction") != "kda")) {
            throw Error(Fault::InvalidArgument, "SPEECH_HAAS_KEYWORD_PREFIX takes effect only on Haas-ja-v1.0 with a"
                        " KDA prediction network and a TDT decoder; " + model_name + " is not one, and a candidate-on"
                        " artifact of it would show no borrowing", "--prefix");
        }
        if (wav.sample_rate != r.sample_rate()) {
            throw Error(Fault::InvalidArgument, "the wav plays at " + std::to_string(wav.sample_rate) +
                        " Hz and the model takes " + std::to_string(r.sample_rate()) + " Hz; resample it first", "wav");
        }
        const std::vector<float> samples = wav.mono();
        if (samples.empty()) throw Error(Fault::InvalidArgument, "the wav holds no samples", "wav");
        const Vocabulary vocabulary(entries, alpha);
        const std::vector<float> features = r.frontend().features(samples);
        const std::vector<float> projected = r.encode(features, r.frontend().frames(samples.size()));
        frames = (int64_t) (projected.size() / (size_t) hidden);
        quiet = quiet_recognition(r, vocabulary, o.guard, samples, projected, decoding_name,
                                  o.decode == "cpu" ? DecodeOn::Cpu : DecodeOn::Device);
    }

    Capture seen;
    {
        observe::Observer observer(plain, outputs);
        Recognizer r(o.model, observer.backend(), kThreads);
        const std::vector<float> samples = wav.mono();
        const Vocabulary vocabulary(entries, alpha);
        const std::vector<float> features = r.frontend().features(samples);
        const std::vector<float> projected = r.encode(features, r.frontend().frames(samples.size()));
        observer.clear();
        seen = observed_recognition(r, vocabulary, o.guard, samples, projected, decoding_name,
                                    o.decode == "cpu" ? DecodeOn::Cpu : DecodeOn::Device, observer, outputs);
    }
    ggml_backend_free(plain);

    bool ok = true;
    const auto fail = [&](const std::string & what) {
        std::printf("FAIL %s\n", what.c_str());
        ok = false;
    };
    if (seen.others > 0 || seen.broken > 0) {
        fail(std::to_string(seen.others) + " graphs the observer knows no decoding kind for and " +
             std::to_string(seen.broken) + " that left not exactly one logits row of " + std::to_string(outputs) +
             " floats in the observed decoding window");
    }
    if (seen.logits.size() != seen.steps + seen.joints || seen.kinds.size() != seen.logits.size()) {
        fail("the observed decoding left " + std::to_string(seen.logits.size()) + " logits rows for " +
             std::to_string(seen.steps + seen.joints) + " graphs; a missing observation cannot pass");
    }
    size_t infinite = 0;
    for (const std::vector<float> & row : seen.logits) {
        for (float x : row) infinite += !std::isfinite(x);
    }
    if (infinite > 0) fail("the observed logits rows hold " + std::to_string(infinite) + " values that are not finite");
    if (seen.text.plain != quiet.text.plain || seen.text.boosted != quiet.text.boosted ||
        seen.text.text != quiet.text.text || seen.text.speech != quiet.text.speech) {
        fail("the observed recognition wrote other texts than the unobserved one");
    }
    if (seen.progress != quiet.progress) {
        fail("the observed recognition made " + std::to_string(seen.progress.size()) + " progress calls, the unobserved one " +
             std::to_string(quiet.progress.size()) + ", or different ones");
    }
    if (!seen.text.speech) {
        fail("the wav has no speech by the model's has_speech(); the boosted pass never ran, and no borrowing can be"
             " seen on it; give non-silent speech");
    } else if (seen.boundary == kNoBoundary || seen.boundary >= seen.logits.size()) {
        fail("the boosted pass ran but no progress call marked where its graphs begin");
    }

    std::ofstream file(std::filesystem::u8path(o.out), std::ios::binary | std::ios::trunc);
    if (!file) throw Error(Fault::Io, "cannot write " + o.out, "output");
    file.write(kMagic, 8);
    std::vector<uint8_t> format;
    put_u32(format, kFormat);
    file.write((const char *) format.data(), (std::streamsize) format.size());

    std::string keywords;
    for (const std::string & entry : entries) keywords += (keywords.empty() ? "" : ",") + json_string(entry);
    std::string meta = "{\"selector\":" + json_string(selector) +
                       ",\"backend\":" + json_string(backend_name) +
                       ",\"device\":\"" + backend_kind + "\"" +
                       ",\"threads\":" + std::to_string(kThreads) +
                       ",\"decode_on\":\"" + (o.decode == "cpu" ? "cpu" : "device") + "\"" +
                       ",\"model\":" + json_string(model_name) +
                       ",\"weights\":" + json_string(weights) +
                       ",\"blank\":" + std::to_string(blank) +
                       ",\"outputs\":" + std::to_string(outputs) +
                       ",\"duration_slots\":" + std::to_string(outputs - blank - 1) +
                       ",\"hidden\":" + std::to_string(hidden) +
                       ",\"alpha\":" + json_number(alpha) +
                       ",\"guard\":" + std::string(o.guard ? "true" : "false") +
                       ",\"keywords\":[" + keywords + "]" +
                       ",\"wav\":" + json_string(std::filesystem::u8path(o.wav).filename().u8string()) +
                       ",\"sample_rate\":" + std::to_string(wav.sample_rate) +
                       ",\"samples\":" + std::to_string(wav.samples.size()) +
                       ",\"frames\":" + std::to_string(frames) +
                       ",\"decoding\":" + json_string(decoding_name) + "}";
    std::vector<uint8_t> meta_payload(meta.begin(), meta.end());
    write_section(file, kTagMeta, meta_payload);
    write_section(file, kTagText, text_payload(seen.text));
    write_section(file, kTagQuietText, text_payload(quiet.text));
    write_section(file, kTagProgress, progress_payload(seen.progress));
    write_section(file, kTagQuietProgress, progress_payload(quiet.progress));
    std::vector<uint8_t> graphs;
    put_u64(graphs, seen.steps);
    put_u64(graphs, seen.joints);
    put_u64(graphs, seen.others);
    put_u64(graphs, seen.boundary);
    write_section(file, kTagGraphs, graphs);
    std::vector<uint8_t> kinds;
    put_u64(kinds, seen.kinds.size());
    kinds.insert(kinds.end(), seen.kinds.begin(), seen.kinds.end());
    write_section(file, kTagKinds, kinds);
    std::vector<uint8_t> logits;
    put_u64(logits, seen.logits.size());
    for (const std::vector<float> & row : seen.logits) {
        if (row.size() != (size_t) outputs) {
            throw Error(Fault::File, "an observed logits row of another width than the joint's outputs", "output");
        }
        for (float x : row) put_f32(logits, x);
    }
    write_section(file, kTagLogits, logits);
    write_section(file, kTagEnd, {});
    file.flush();
    if (!file) throw Error(Fault::Io, "cannot write " + o.out, "output");

    std::printf("observed: %zu prediction graphs, %zu joint-only graphs, %zu logits rows, the boosted pass begins at"
                " row %llu of a recognition of %lld frames\n",
                seen.steps, seen.joints, seen.logits.size(), (unsigned long long) seen.boundary, (long long) frames);
    std::printf("plain: %s\nboosted: %s\ntext: %s\n", seen.text.plain.c_str(), seen.text.boosted.c_str(),
                seen.text.text.c_str());
    std::printf("%s\n", ok ? "the normal recognition's artifacts: OK" : "the normal recognition's artifacts: FAILED");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    const char * usage = "usage: %s <model.gguf> <speech.wav> <output file> [gpu|cpu|device name]\n"
                         "                 [--keywords-file <file>] [--alpha <number>] [--guard 0|1]\n"
                         "                 [--decode cpu|device] [--prefix 0|1]\n";
    Options o;
    std::vector<std::string> positional;
    bool have_device = false;
    for (size_t i = 1; i < args.size(); i++) {
        const std::string & arg = args[i];
        const std::string value = i + 1 < args.size() ? args[i + 1] : "";
        if (arg == "--keywords-file" && !value.empty()) {
            o.keywords_file = value;
            i++;
        } else if (arg == "--alpha" && !value.empty()) {
            char * end = nullptr;
            o.alpha = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0') {
                std::fprintf(stderr, "error: --alpha takes a number, not %s\n%s", value.c_str(), usage);
                return 2;
            }
            i++;
        } else if (arg == "--guard" && (value == "0" || value == "1")) {
            o.guard = value == "1";
            i++;
        } else if (arg == "--decode" && (value == "cpu" || value == "device")) {
            o.decode = value;
            i++;
        } else if (arg == "--prefix" && (value == "0" || value == "1")) {
            o.prefix = value;
            i++;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "error: unknown or incomplete option %s\n%s", arg.c_str(), usage);
            return 2;
        } else if (positional.size() < 3) {
            positional.push_back(arg);
        } else if (!have_device) {
            o.device = arg;
            have_device = true;
        } else {
            std::fprintf(stderr, "error: one argument too many: %s\n%s", arg.c_str(), usage);
            return 2;
        }
    }
    if (positional.size() < 3 || o.keywords_file.empty()) {
        std::fprintf(stderr, "%s", usage);
        return 2;
    }
    o.model = positional[0];
    o.wav = positional[1];
    o.out = positional[2];
    try {
        return normal_mode(o);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
