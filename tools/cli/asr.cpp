#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "asr-stream.h"
#include "catalog.h"
#include "commands.h"
#include "enhancement.h"
#include "fetch.h"
#include "json.h"
#include "regions.h"
#include "speakers.h"
#include "terminal.h"
#include "wave-input.h"

// speech asr: recognizes the speech in WAVE files and writes each file's text to stdout in the order the files are
// given, as text or as one JSON object per file, and reports on stderr the load time and, for each file, its seconds
// of audio, the time to its text and the RTFx, the seconds of audio over the seconds they took. With --vad it
// transcribes each file by the regions where a detection model finds that someone speaks, and PCM from stdin (-) or the
// microphone (--live) region by region as the audio arrives (asr-stream.h). With --enhance an enhancement model enhances
// each file whole, or each utterance once it has ended, before it is recognized.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/** The flags and defaults of transcription by regions, for the help: "--threshold 0.5, ... and --max-speech-duration-s 10". */
std::string region_defaults() {
    const std::vector<RequestOption> defaults = region_options({});
    std::string out;
    for (size_t i = 0; i < defaults.size(); i++) {
        const OptionValue & v = defaults[i].value;
        const std::string value = v.index() == 1 ? std::to_string(std::get<int64_t>(v)) : json_number(std::get<double>(v));
        out += (i == 0 ? "" : i + 1 == defaults.size() ? " and " : ", ") + option_flag(defaults[i].option) + " " + value;
    }
    return out;
}

/** The detection model of --vad, and the request options that are its own: those it takes and the recognizer does not. */
struct Detection {
    Model model{nullptr, speech_model_free};
    std::vector<RequestOption> options;
};

/** The diarization model of --diarize, and the request options that are its own. */
struct Diarizer {
    Model model{nullptr, speech_model_free};
    std::vector<RequestOption> options;
};

/** The enhancement model of --enhance, and the request options that are its own. */
struct Enhancer {
    Model model{nullptr, speech_model_free};
    std::vector<RequestOption> options;
};

/** The file of the enhancement model --enhance names, fetched if it is not there yet; none without --enhance. */
std::optional<std::string> enhancement_file(const CommandLine & line) {
    const std::optional<std::string> argument = line.value("--enhance");
    if (!argument) return std::nullopt;
    const std::string path = model_file(*argument);
    const speech_task task = speech_model_info_task(file_info(path).get());
    if (task != SPEECH_TASK_ENHANCEMENT) {
        throw UsageError("--enhance takes a model of speech enhancement, and " + *argument + " is a speech " + task_name(task) + " model");
    }
    return path;
}

/**
 * The enhancement model of `path`, on the recognizer's device, and the options that are its own: those it takes and none
 * of `others`, the other models' files, takes, which leave `options`.
 */
std::optional<Enhancer> enhancer_of(const CommandLine & line, const std::optional<std::string> & path,
                                    const std::vector<const speech_model_info *> & others, std::vector<RequestOption> & options) {
    if (!path) return std::nullopt;
    Enhancer e;
    e.model = load_model(*path, line.loading(false));
    const ModelInfo info = model_info(e.model.get());
    std::vector<RequestOption> rest;
    for (const RequestOption & o : options) {
        bool own = speech_model_info_takes(info.get(), o.option);
        for (const speech_model_info * other : others) own = own && !(other && speech_model_info_takes(other, o.option));
        (own ? e.options : rest).push_back(o);
    }
    options = rest;
    return e;
}

/** The file of the diarization model --diarize names, fetched if it is not there yet; none without --diarize. */
std::optional<std::string> diarization_file(const CommandLine & line) {
    const std::optional<std::string> argument = line.value("--diarize");
    if (!argument) return std::nullopt;
    if (!line.has("--vad")) throw UsageError("--diarize gives each region its speakers; give --vad too, as in --vad silero-vad");
    const std::string path = model_file(*argument);
    const speech_task task = speech_model_info_task(file_info(path).get());
    if (task != SPEECH_TASK_DIARIZATION) {
        throw UsageError("--diarize takes a model of speech diarization, and " + *argument + " is a speech " + task_name(task) + " model");
    }
    return path;
}

/** The file of the detection model --vad names, fetched if it is not there yet; none without --vad. */
std::optional<std::string> detection_file(const CommandLine & line) {
    const std::optional<std::string> argument = line.value("--vad");
    if (!argument) return std::nullopt;
    const std::string path = model_file(*argument);
    const speech_task task = speech_model_info_task(file_info(path).get());
    if (task != SPEECH_TASK_DETECTION) {
        throw UsageError("--vad takes a model of speech detection, such as silero-vad, and " + *argument + " is a speech " + task_name(task) +
                         " model");
    }
    return path;
}

std::optional<Detection> detection_of(const CommandLine & line, const std::optional<std::string> & path, const speech_model_info * recognizer,
                                      std::vector<RequestOption> & options) {
    if (!path) return std::nullopt;
    Detection d;
    d.model = load_model(*path, detection_loading(line.loading(false)));
    const ModelInfo info = model_info(d.model.get());
    std::vector<RequestOption> recognition;
    for (const RequestOption & o : options) {
        const bool own = speech_model_info_takes(info.get(), o.option) && !speech_model_info_takes(recognizer, o.option);
        (own ? d.options : recognition).push_back(o);
    }
    d.options = region_options(d.options);
    options = recognition;
    return d;
}

/**
 * The diarization model of `path`, on the recognizer's device, and the options that are its own: those it takes and
 * neither the recognizer nor the detection model takes, which leave `options`.
 */
std::optional<Diarizer> diarizer_of(const CommandLine & line, const std::optional<std::string> & path, const speech_model_info * recognizer,
                                    const speech_model_info * detector, std::vector<RequestOption> & options) {
    if (!path) return std::nullopt;
    Diarizer d;
    d.model = load_model(*path, line.loading(false));
    const ModelInfo info = model_info(d.model.get());
    std::vector<RequestOption> rest;
    for (const RequestOption & o : options) {
        const bool own = speech_model_info_takes(info.get(), o.option) && !speech_model_info_takes(recognizer, o.option) &&
                         !speech_model_info_takes(detector, o.option);
        (own ? d.options : rest).push_back(o);
    }
    options = rest;
    return d;
}

/**
 * Writes a file transcribed by `regions` with --diarize, each region with the speakers within it: in text a line each,
 * whose speaker is the one who speaks the most in it, and in JSON the file's recognition followed by "regions".
 */
void write_diarized(FILE * out, bool json, const std::string & path, const Transcript & transcript, bool timestamps, const std::vector<Region> & regions,
                    const std::vector<Transcript> & parts, const std::vector<SpeakerSegment> & speakers) {
    std::string listed;
    for (size_t k = 0; k < regions.size(); k++) {
        const std::vector<SpeakerSegment> within = speakers_within(speakers, regions[k].start, regions[k].end);
        if (!json) {
            std::fprintf(out, "%s\t%.3f\t%.3f\t%s\t%s\n", path.c_str(), regions[k].start, regions[k].end, speaker_column(main_speaker(within)).c_str(),
                         parts[k].text.c_str());
            continue;
        }
        listed += std::string(k ? "," : "") + "{\"start\":" + json_number(regions[k].start) + ",\"end\":" + json_number(regions[k].end) +
                  ",\"text\":" + json_string(parts[k].text) + ",\"speakers\":" + speakers_json(within) + "}";
    }
    if (json) {
        std::fprintf(out, "{\"file\":%s%s,\"regions\":[%s]}\n", json_string(path).c_str(), recognition_members(transcript, timestamps).c_str(),
                     listed.c_str());
    }
}

int run_asr(const CommandLine & line, FILE * out) {
    const std::string format = line.value("--format").value_or("text");
    if (format != "text" && format != "json") throw UsageError("--format takes text or json, not \"" + format + "\"");
    const bool live = line.has("--live");
    const bool piped = std::find(line.args.begin() + 1, line.args.end(), "-") != line.args.end();
    if (piped && line.args.size() > 2) throw UsageError("- reads 16-bit PCM from stdin, alone; give it without files");
    if (live && line.args.size() > 1) throw UsageError("--live listens to the microphone and takes no AUDIO; give one or the other");
    if (!live && line.args.size() < 2) {
        // Without --live, AUDIO follows MODEL, and a command line short of it is answered as the parser answers one short
        // of its arguments: a lone argument that names no model, such as a WAVE file, is told what MODEL is and which
        // model to start with.
        if (!names_a_model(line.args[0])) throw UsageError(no_model_message(line.command().usage, *line.command().model));
        throw UsageError("give the WAVE files to transcribe, - for 16-bit PCM on stdin, or --live for the microphone");
    }
    if (line.has("--rate") && !piped) throw UsageError("--rate is the rate of the PCM that - reads from stdin");
    if (line.integer("--rate").value_or(1) <= 0) throw UsageError("--rate takes the PCM's rate in Hz");
    if ((live || piped) && !line.has("--vad")) {
        throw UsageError(std::string(live ? "--live" : "-") + " writes each utterance as it ends, cut where a detection model finds that "
                         "someone speaks; give one with --vad, as in --vad silero-vad");
    }
    if ((live || piped) && std::any_of(line.options.begin(), line.options.end(), [](const RequestOption & o) { return o.option == SPEECH_OPT_TIMESTAMPS; })) {
        throw UsageError("--timestamps gives the segments of files; the utterances of " + std::string(live ? "--live" : "-") +
                         " carry their times in the events of --format json");
    }
    if (line.has("--vad") && std::any_of(line.options.begin(), line.options.end(), [](const RequestOption & o) {
            return o.option == SPEECH_OPT_SPEAKERS && std::get<bool>(o.value);
        })) {
        throw UsageError("--speakers numbers the speakers of one recognition, and --vad recognizes each region alone; leave out --vad, "
                         "or give --diarize for the speakers of a whole file or a stream");
    }

    // The models named for the first time are fetched here, before the load is timed.
    const std::string path = model_file(line.args[0]);
    require_options(path, line);
    const std::optional<std::string> detector = detection_file(line);
    const std::optional<std::string> diarizer_path = diarization_file(line);
    const std::optional<std::string> enhancer_path = enhancement_file(line);
    const auto t0 = Clock::now();
    terminal::Status loading("Loading " + argument_label(line.args[0]) + (detector ? " and " + argument_label(*line.value("--vad")) : "") +
                             (diarizer_path ? " and " + argument_label(*line.value("--diarize")) : "") +
                             (enhancer_path ? " and " + argument_label(*line.value("--enhance")) : ""));
    const Model model = load_model(path, line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    std::vector<RequestOption> options = line.options;
    // Each option goes to the one model that takes it alone: the diarization model's, which neither of the others takes,
    // are taken from the rest before the detection model's.
    const ModelInfo diarizing_file = diarizer_path ? file_info(*diarizer_path) : ModelInfo(nullptr, speech_model_info_free);
    const ModelInfo detecting_file = detector ? file_info(*detector) : ModelInfo(nullptr, speech_model_info_free);
    // The enhancement model's options are those none of the others takes.
    const std::optional<Enhancer> enhancer = enhancer_of(line, enhancer_path, {m, detecting_file.get(), diarizing_file.get()}, options);
    const std::optional<Diarizer> diarizer = diarizer_of(line, diarizer_path, m, detecting_file.get(), options);
    const std::optional<Detection> detection = detection_of(line, detector, m, options);
    const ModelInfo detecting = detection ? model_info(detection->model.get()) : ModelInfo(nullptr, speech_model_info_free);
    const ModelInfo diarizing = diarizer ? model_info(diarizer->model.get()) : ModelInfo(nullptr, speech_model_info_free);
    const ModelInfo enhancing = enhancer ? model_info(enhancer->model.get()) : ModelInfo(nullptr, speech_model_info_free);
    std::vector<const speech_model_info *> loaded{m};
    if (detecting) loaded.push_back(detecting.get());
    if (diarizing) loaded.push_back(diarizing.get());
    if (enhancing) loaded.push_back(enhancing.get());
    loading.done(loaded_line(loaded, seconds_since(t0)));
    if (line.has("-v")) {
        std::string languages;
        for (size_t i = 0; i < speech_model_info_language_count(m); i++) languages += (i ? ", " : "") + std::string(speech_model_info_language(m, i));
        std::fprintf(stderr, "speech.cpp %s, %d Hz, languages: %s\n", speech_version(), speech_model_info_sample_rate(m), languages.c_str());
    }
    if (live || piped) {
        return transcribe_stream(line, out, model.get(), options, detection->model.get(), detection->options, diarizer ? diarizer->model.get() : nullptr,
                                 diarizer ? diarizer->options : std::vector<RequestOption>(), enhancer ? enhancer->model.get() : nullptr,
                                 enhancer ? enhancer->options : std::vector<RequestOption>(), format == "json");
    }
    const bool timestamps = timestamps_in_effect(m, options), with_speakers = in_effect(m, options, SPEECH_OPT_SPEAKERS);

    double audio_total = 0, busy = 0;
    bool model_limit = false;
    for (size_t i = 1; i < line.args.size(); i++) {
        const std::string & path = line.args[i];
        Transcript transcript;
        size_t regions = 0;
        std::vector<Region> found;
        std::vector<Transcript> parts;
        std::vector<SpeakerSegment> speakers;
        double audio = 0;
        const auto start = Clock::now();
        {
            // The line is erased before the text reaches stdout, which may be the same terminal.
            const terminal::Status status("Transcribing " + path);
            try {
                const Wav wav = read_audio(path);
                std::vector<float> samples = wav.mono();
                int rate = wav.sample_rate;
                audio = (double) samples.size() / rate;
                Cancellation never;
                // The file is enhanced whole, and its regions are found in the enhanced audio.
                if (enhancer) {
                    Enhanced enhanced = *enhance_whole(enhancer->model.get(), samples, rate, enhancer->options, never);
                    samples = std::move(enhanced.samples);
                    rate = enhanced.rate;
                }
                if (detection) {
                    const Request request = detection_request(detection->model.get(), samples, rate, detection->options);
                    found = *detect_regions(request.get(), never);
                    regions = found.size();
                    transcript = *transcribe_regions(model.get(), samples, rate, found, options, timestamps, never, &parts);
                    if (diarizer) {
                        const Request diarization = diarization_request(diarizer->model.get(), samples, rate, diarizer->options);
                        speakers = *diarize_segments(diarization.get(), never);
                    }
                } else {
                    const Request request = new_request(model.get());
                    check(speech_request_set_audio(request.get(), samples.data(), samples.size(), rate));
                    apply_options(request.get(), options);
                    check(speech_transcribe(request.get()));
                    transcript = transcript_of(speech_request_result(request.get()), timestamps, with_speakers);
                }
            } catch (const Failure & e) {
                throw Failure(e.code(), e.option(), path + ": " + e.what());
            }
        }
        const double took = seconds_since(start);
        if (diarizer) {
            write_diarized(out, format == "json", path, transcript, timestamps, found, parts, speakers);
        } else if (format == "json") {
            std::fprintf(out, "{\"file\":%s%s}\n", json_string(path).c_str(), recognition_members(transcript, timestamps, with_speakers).c_str());
        } else if (!timestamps && !with_speakers) {
            std::fprintf(out, "%s\n", transcript.text.c_str());
        } else if (transcript.untimed) {
            // The text with no times in the columns of the times, so that every line keeps the same columns.
            std::fprintf(out, "%s\t\t\t%s%s\n", path.c_str(), with_speakers ? "-\t" : "", transcript.text.c_str());
        } else {
            for (const Timed & s : transcript.segments) {
                std::fprintf(out, "%s\t%.3f\t%.3f\t%s%s\n", path.c_str(), s.start, s.end, with_speakers ? (speaker_column(s.speaker) + "\t").c_str() : "",
                             s.text.c_str());
            }
        }
        if (transcript.untimed) {
            std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Warning, path + ": the text has no times: " + *transcript.untimed).c_str());
        }
        std::fflush(out);
        char said[128];
        std::snprintf(said, sizeof said, "%.2f s of audio in %.3f s, RTFx %.1f", audio, took, audio / took);
        std::string details = said;
        if (detection) details += ", " + std::to_string(regions) + (regions == 1 ? " region" : " regions");
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, path + ": " + terminal::paint(terminal::Color::Gray, details)).c_str());
        if (transcript.stop == SPEECH_STOP_MODEL_LIMIT) {
            std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Warning, path + ": the recognition reached the most the model writes and was "
                                                                                      "stopped there; its text is what was written").c_str());
            model_limit = true;
        }
        audio_total += audio;
        busy += took;
    }
    if (line.args.size() > 2) {
        char said[128];
        std::snprintf(said, sizeof said, "%zu files, %.2f s of audio in %.3f s, RTFx %.1f", line.args.size() - 1, audio_total, busy, audio_total / busy);
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, said).c_str());
    }
    return model_limit ? 3 : 0;
}

}  // namespace

Command asr_command() {
    Command c;
    c.name = "asr";
    c.usage = "asr MODEL [options] AUDIO.wav...|-|--live";
    c.summary = "write the text of WAVE files, or of speech as it is said";
    c.description =
        "Recognizes each WAVE file (16-, 24- or 32-bit PCM or 32-bit float, any rate, channels averaged) and writes\n"
        "its text to stdout in the order given: with --format text one line per file, or with --timestamps one line\n"
        "per segment, FILE<TAB>START<TAB>END<TAB>TEXT, times in seconds, and with --speakers, for a model that tells\n"
        "who speaks, FILE<TAB>START<TAB>END<TAB>SPEAKER<TAB>TEXT, SPEAKER the model's number from 0 or - for none;\n"
        "with --format json one object per file, {\"file\", \"text\", \"stop\"}, with \"languages\", the tags of the\n"
        "languages the model heard, where it names any, \"segments\" with --timestamps or --speakers, each with its\n"
        "\"speaker\" with --speakers, and \"tokens\" with --timestamps. Where the model cannot time its text, the text\n"
        "comes with empty times, or \"untimed\" with the reason in JSON, and a warning says why. It exits with 3 when a\n"
        "recognition stopped at the most the model writes, after writing every file's text.\n"
        "With --vad, a detection model, on the CPU, finds where someone speaks in each file, each region is recognized\n"
        "alone and the texts are joined, the times of the whole file. The detection options are those of speech vad,\n"
        "with OpenAI's server_vad defaults and a longest region:\n" + region_defaults() + ".\n"
        "A file in which no one speaks gives an empty text.\n"
        "With --diarize, a diarization model finds who speaks where, each file whole in its offline mode, and each region\n"
        "has its speakers: with --format text a line FILE<TAB>START<TAB>END<TAB>SPEAKER<TAB>TEXT, SPEAKER the number of\n"
        "the one who speaks the most in it, or - for none, and with --format json \"regions\", each {\"start\", \"end\",\n"
        "\"text\", \"speakers\": [{\"speaker\", \"start\", \"end\"}]}. It runs on the device of MODEL.\n"
        "With --vad, - reads 16-bit little-endian mono PCM at --rate from stdin until it ends, and --live the default\n"
        "microphone until Ctrl-C, and each region is written as it ends: with --format text its text as a line, which\n"
        "on a terminal shows the text agreed so far while it is said; with --format json OpenAI's Realtime server\n"
        "events, one per line: input_audio_buffer.speech_started, .speech_stopped and .committed, and\n"
        "conversation.item.input_audio_transcription.delta, .completed and .failed. With --diarize, in its low_latency\n"
        "mode unless --streaming-mode says otherwise, each line of text begins with its main speaker and a tab, and each\n"
        ".completed carries \"speakers\" once the diarization has settled its utterance.\n"
        "With --enhance, an enhancement model removes noise before the recognition: each file whole, before --vad finds\n"
        "its regions, and with - or --live each utterance whole once it has ended, while the detection and the text shown\n"
        "as it is said hear the audio as it arrives. It runs on the device of MODEL; its options are those only it takes,\n"
        "such as --chunk-size-in-seconds. Enhancement can help on noisy audio and hurt on clean audio: try both.";
    c.flags = {
        {"--format", "text|json", false, "text (the default) or one JSON object per file, or per event, and line"},
        {"--vad", "MODEL", false, "transcribe by the regions where this detection model finds speech"},
        {"--live", "", false, "transcribe the default microphone until Ctrl-C, with --vad"},
        {"--diarize", "MODEL", false, "give each region the speakers this diarization model finds, with --vad"},
        {"--enhance", "MODEL", false, "remove noise with this enhancement model before the recognition"},
        {"--rate", "HZ", false, "the rate of the PCM that - reads from stdin, 16000 unless given"},
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists; for MODEL"),
        threads_flag(),
        verbose_flag("also report the release, the sample rate and the model's languages, and each utterance as it ends"),
    };
    c.request_options = true;
    c.model = ModelKind::Recognition;
    c.min_args = 1;
    c.max_args = SIZE_MAX;
    c.run = run_asr;
    return c;
}
