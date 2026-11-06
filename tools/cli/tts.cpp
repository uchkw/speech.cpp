#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "ggml.h"
#include "sentences.h"
#include "terminal.h"
#include "wave-output.h"

// speech tts: speaks the text, or each non-empty line of stdin as a text of its own, into one 16-bit mono WAVE at the
// model's rate (wave-output.h), written as the audio is made so that a player reading stdout starts before the rest is
// made; a model whose request speaks only a short time speaks a text a sentence at a time (sentences.h). It reports on
// stderr where the time went: the load time, and for each text the time to first audio, the whole and the RTFx, the
// seconds of audio over the seconds they took.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/** What the audio callback needs of one text, and the failure to write that stopped it. */
struct Speaking {
    WavWriter & wav;
    Clock::time_point start;
    /** The line that shows the text being spoken, what it says before the audio made, and the model's rate. */
    terminal::Status & status;
    std::string doing;
    double rate;
    uint64_t samples = 0;
    double first_audio = -1;
    std::optional<Failure> failed;
};

/** Writes the audio; a failure to write stops the request, since no exception may cross the C API. */
int on_audio(const float * s, size_t n, void * user_data) {
    Speaking & r = *static_cast<Speaking *>(user_data);
    if (r.first_audio < 0) r.first_audio = seconds_since(r.start);
    try {
        r.wav.write(s, n);
    } catch (const Failure & e) {
        r.failed = e;
        return 1;
    }
    r.samples += n;
    char made[48];
    std::snprintf(made, sizeof made, ", %.1f s of audio", r.samples / r.rate);
    r.status.set(r.doing + made);
    return 0;
}

/** The next line of stdin with text on it, without its line ending or a UTF-8 byte order mark, or false at the end. */
bool next_line(std::string & line, bool & first) {
    while (std::getline(std::cin, line)) {
        if (first && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        first = false;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") != std::string::npos) return true;
    }
    return false;
}

std::string joined(size_t count, const char * (*get)(const speech_model_info *, size_t), const speech_model_info * m) {
    std::string out;
    for (size_t i = 0; i < count; i++) out += (i ? ", " : "") + std::string(get(m, i));
    return out;
}

int run_tts(const CommandLine & line, FILE * out) {
    const std::optional<std::string> output = line.value("-o");
    if (!output) throw UsageError("name the WAVE file to write with -o, or give -o - for stdout");
    const bool verbose = line.has("-v");
    const bool from_stdin = line.args.size() == 1;

    // A model named for the first time is fetched here, before the load is timed.
    const std::string path = model_file(line.args[0]);
    require_options(path, line);
    const auto t0 = Clock::now();
    terminal::Status loading("Loading " + argument_label(line.args[0]));
    const Model model = load_model(path, line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    loading.done(loaded_line({m}, seconds_since(t0)));
    if (verbose) {
        std::fprintf(stderr, "speech.cpp %s, %d Hz, voices: %s; languages: %s\n", speech_version(), speech_model_info_sample_rate(m),
                     joined(speech_model_info_voice_count(m), speech_model_info_voice_name, m).c_str(),
                     joined(speech_model_info_language_count(m), speech_model_info_language, m).c_str());
    }

    Output file(*output, out);
    WavWriter wav(file.file(), speech_model_info_sample_rate(m));
    const double rate = speech_model_info_sample_rate(m);
    std::string text = from_stdin ? "" : line.args[1];
    bool first_line = true, model_limit = false;
    int count = 0;
    uint64_t samples = 0;
    double busy = 0;
    // Nothing cancels a run of the command line; a write that fails stops it through the audio callback.
    Cancellation cancellation;
    while (from_stdin ? next_line(text, first_line) : count == 0) {
        count++;
        // A text is named by its line of stdin, or by the file it goes to.
        const std::string what = from_stdin ? "line " + std::to_string(count) : *output == "-" ? "" : *output;
        const std::string where = what.empty() ? "" : what + ": ";
        const std::string doing = what.empty() ? "Speaking" : "Speaking " + what;
        terminal::Status status(doing);
        Speaking speaking{wav, Clock::now(), status, doing, rate};
        std::optional<Spoken> spoken;
        try {
            spoken = speak_text(model.get(), text, line.options, Split::Sentences, on_audio, nullptr, &speaking, cancellation);
            if (speaking.failed) throw *speaking.failed;
        } catch (const Failure & e) {
            // A failure names the line of stdin at fault; a text given as an argument is the only one.
            throw Failure(e.code(), e.option(), (from_stdin ? where : "") + e.what());
        }
        const uint64_t made = spoken->samples;
        const double total = seconds_since(speaking.start), audio = (double) made / rate;
        const speech_stop stop = spoken->stop;
        const std::string requests = spoken->requests > 1 ? " from " + std::to_string(spoken->requests) + " requests" : "";
        char said[160];
        std::snprintf(said, sizeof said, "%.2f s of audio%s in %.3f s, RTFx %.1f, time to first audio %.3f s", audio, requests.c_str(), total,
                      audio / total, speaking.first_audio);
        std::string details = said;
        if (verbose) details += ", seed " + std::to_string((long long) spoken->seed) + ", " + speech_stop_name(stop);
        status.done(where + terminal::paint(terminal::Color::Gray, details), terminal::Mark::Audio);
        if (stop == SPEECH_STOP_MAX_SECONDS) {
            std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Warning, where + "the speech reached --max-seconds and was stopped there").c_str());
        } else if (stop == SPEECH_STOP_MODEL_LIMIT) {
            std::fprintf(stderr, "%s\n",
                         terminal::marked(terminal::Mark::Warning, where + "the speech reached the longest the model makes and was stopped there").c_str());
            model_limit = true;
        }
        samples += made;
        busy += total;
    }
    if (count == 0) throw UsageError("stdin held no text; give the text as an argument or pipe lines of text in");
    wav.finish();
    file.keep();
    if (from_stdin) {
        char said[160];
        std::snprintf(said, sizeof said, "%d lines, %.2f s of audio in %.3f s, RTFx %.1f", count, samples / rate, busy, samples / rate / busy);
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, std::string(*output == "-" ? "" : *output + ": ") + said).c_str());
    }
    return model_limit ? 3 : 0;
}

}  // namespace

Command tts_command() {
    Command c;
    c.name = "tts";
    c.usage = "tts MODEL -o FILE|- [options] [TEXT]";
    c.summary = "speak text into a WAVE file";
    c.description =
        "Speaks TEXT, or without it each non-empty line of stdin as a text of its own, into one 16-bit mono WAVE at\n"
        "the model's rate, written as the audio comes; -o - writes it to stdout. A model whose request speaks less than\n"
        "a minute, such as Irodori-TTS, speaks a text of several sentences a sentence at a time, and a sentence it\n"
        "cannot speak at once at its commas or spaces. A seed applies to every text, and without one each text draws\n"
        "its own. A text that begins with - follows --. Exits with 3 when a text stopped at the longest speech the\n"
        "model makes.";
    c.flags = {
        {"-o", "FILE", false, "the WAVE file to write, or - for stdout"},
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the model, and each text's seed and stop reason"),
    };
    c.request_options = true;
    c.model = ModelKind::Synthesis;
    c.min_args = 1;
    c.max_args = 2;
    c.run = run_tts;
    return c;
}
