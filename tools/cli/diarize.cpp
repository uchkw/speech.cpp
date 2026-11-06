#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "json.h"
#include "speakers.h"
#include "terminal.h"
#include "wave-input.h"

// speech diarize: finds who speaks where in WAVE files and writes each file's segments to stdout in the order the files
// are given, as text, as one JSON object per file or as RTTM, and reports on stderr the load time and, for each file,
// its seconds of audio, the time to its segments, the RTFx and the speakers it heard.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

int run_diarize(const CommandLine & line, FILE * out) {
    const std::string format = line.value("--format").value_or("text");
    if (format != "text" && format != "json" && format != "rttm") throw UsageError("--format takes text, json or rttm, not \"" + format + "\"");

    // RTTM names a recording by an id without spaces, the file's name without its folder and extension.
    if (format == "rttm") {
        for (size_t i = 1; i < line.args.size(); i++) {
            const std::string id = std::filesystem::u8path(line.args[i]).stem().u8string();
            if (id.empty() || std::any_of(id.begin(), id.end(), [](char c) { return std::isspace((unsigned char) c); })) {
                throw UsageError("--format rttm names each recording by its file's name, and " + line.args[i] +
                                 "'s holds a space or is empty; rename the file, or write text or json");
            }
        }
    }

    // A model named for the first time is fetched here, before the load is timed.
    const std::string path = model_file(line.args[0]);
    require_options(path, line);
    const auto t0 = Clock::now();
    terminal::Status status("Loading " + argument_label(line.args[0]));
    const Model model = load_model(path, line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    status.done(loaded_line({m}, seconds_since(t0)));
    if (line.has("-v")) std::fprintf(stderr, "speech.cpp %s, %d Hz\n", speech_version(), speech_model_info_sample_rate(m));

    double audio_total = 0, busy = 0;
    for (size_t i = 1; i < line.args.size(); i++) {
        const std::string & file = line.args[i];
        std::vector<SpeakerSegment> segments;
        double audio = 0;
        const auto start = Clock::now();
        {
            // The line is erased before the segments reach stdout, which may be the same terminal.
            const terminal::Status diarizing("Diarizing " + file);
            try {
                const Wav wav = read_audio(file);
                const std::vector<float> samples = wav.mono();
                audio = (double) samples.size() / wav.sample_rate;
                const Request request = diarization_request(model.get(), samples, wav.sample_rate, line.options);
                Cancellation never;
                segments = *diarize_segments(request.get(), never);
            } catch (const Failure & e) {
                throw Failure(e.code(), e.option(), file + ": " + e.what());
            }
        }
        const double took = seconds_since(start);
        const std::string id = std::filesystem::u8path(file).stem().u8string();
        std::string json = "{\"file\":" + json_string(file) + ",\"segments\":[";
        std::set<int> speakers;
        for (size_t k = 0; k < segments.size(); k++) {
            const SpeakerSegment & s = segments[k];
            speakers.insert(s.speaker);
            if (format == "json") {
                json += std::string(k ? "," : "") + "{\"start\":" + json_number(s.start) + ",\"end\":" + json_number(s.end) +
                        ",\"speaker\":" + std::to_string(s.speaker) + "}";
            } else if (format == "rttm") {
                std::fprintf(out, "SPEAKER %s 1 %.3f %.3f <NA> <NA> speaker_%d <NA> <NA>\n", id.c_str(), s.start, s.end - s.start, s.speaker);
            } else {
                std::fprintf(out, "%s\t%.3f\t%.3f\t%d\n", file.c_str(), s.start, s.end, s.speaker);
            }
        }
        if (format == "json") std::fprintf(out, "%s]}\n", json.c_str());
        std::fflush(out);
        char said[160];
        std::snprintf(said, sizeof said, "%.2f s of audio in %.3f s, RTFx %.0f, %zu segment%s of %zu speaker%s", audio, took, audio / took, segments.size(),
                      segments.size() == 1 ? "" : "s", speakers.size(), speakers.size() == 1 ? "" : "s");
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, file + ": " + terminal::paint(terminal::Color::Gray, said)).c_str());
        audio_total += audio;
        busy += took;
    }
    if (line.args.size() > 2) {
        char said[128];
        std::snprintf(said, sizeof said, "%zu files, %.2f s of audio in %.3f s, RTFx %.0f", line.args.size() - 1, audio_total, busy, audio_total / busy);
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, said).c_str());
    }
    return 0;
}

}  // namespace

Command diarize_command() {
    Command c;
    c.name = "diarize";
    c.usage = "diarize MODEL [options] AUDIO.wav...";
    c.summary = "write who speaks where in WAVE files";
    c.description =
        "Finds who speaks where in each WAVE file (16-, 24- or 32-bit PCM or 32-bit float, any rate, channels averaged)\n"
        "with a diarization model and writes each speaker's segments to stdout in the order given, in order of their\n"
        "starts: with --format text one line per segment, FILE<TAB>START<TAB>END<TAB>SPEAKER, times in seconds; with\n"
        "--format json one object per file, {\"file\", \"segments\": [{\"start\", \"end\", \"speaker\"}]}; with --format\n"
        "rttm one RTTM line per segment, the file's name without its extension as its id. A speaker is the model's number\n"
        "for a voice, from 0 in the order the voices first speak in the file; segments of two speakers overlap where they\n"
        "speak at once. The model hears each file in the chunks of --streaming-mode, offline (27.2 s) unless given, or of\n"
        "--chunk-len and the other settings of NeMo's streaming; --onset, --offset, --pad-onset, --pad-offset,\n"
        "--min-duration-on and --min-duration-off are the postprocessing of NeMo's diarize(), with its defaults.";
    c.flags = {
        {"--format", "text|json|rttm", false, "text (the default), one JSON object per file and line, or RTTM"},
        device_flag("auto (the default), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release and the sample rate"),
    };
    c.request_options = true;
    c.model = ModelKind::Diarization;
    c.min_args = 2;
    c.max_args = SIZE_MAX;
    c.run = run_diarize;
    return c;
}
