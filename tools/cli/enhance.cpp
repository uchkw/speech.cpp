#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "commands.h"
#include "enhancement.h"
#include "fetch.h"
#include "terminal.h"
#include "wave-input.h"
#include "wave-output.h"

// speech enhance: enhances WAVE files with an enhancement model and writes each as a 16-bit mono WAVE at the rate it was
// enhanced at (wave-output.h), written as the audio comes: one file to the file -o names or to stdout, several to the
// folder -o names, each under its own name; or 16-bit PCM from stdin (-) as it arrives, through an enhancement of audio
// given a piece at a time, whose samples are the whole audio's. It reports on stderr the load time and, for each file,
// its seconds of audio, the time they took and the RTFx, the seconds of audio over the seconds they took.

namespace {

using Clock = std::chrono::steady_clock;
namespace fs = std::filesystem;

/**
 * What has arrived on stdin, up to `size` bytes, as soon as any has: fread() would wait for the whole buffer, and so
 * hold back audio that a short chunk would already enhance. 0 at its end.
 */
size_t read_stdin(unsigned char * data, size_t size) {
    for (;;) {
#ifdef _WIN32
        const int got = _read(0, data, (unsigned) size);
#else
        const ssize_t got = read(0, data, size);
#endif
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) throw Failure(speech_status_name(SPEECH_ERROR_IO), "audio", std::string("cannot read stdin: ") + std::strerror(errno));
        return (size_t) got;
    }
}

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

/** What the audio callback needs of one file, and the failure to write that stopped it. */
struct Enhancing {
    WavWriter & wav;
    terminal::Status & status;
    std::string doing;
    double rate;
    uint64_t samples = 0;
    std::optional<Failure> failed;
};

/** Writes the audio; a failure to write stops the request, since no exception may cross the C API. */
int on_audio(const float * s, size_t n, void * user_data) {
    Enhancing & e = *static_cast<Enhancing *>(user_data);
    try {
        e.wav.write(s, n);
    } catch (const Failure & f) {
        e.failed = f;
        return 1;
    }
    e.samples += n;
    char made[48];
    std::snprintf(made, sizeof made, ", %.1f s of audio", e.samples / e.rate);
    e.status.set(e.doing + made);
    return 0;
}

/**
 * Where each input goes: the file or stdout -o names for one input that is not a folder, and otherwise the folder -o
 * names, made if it is not there, with each input under its own name as a .wav file. Two inputs of one name, and an
 * output that is one of the inputs, are refused before anything is written.
 */
std::vector<std::string> destinations(const std::vector<std::string> & inputs, const std::string & output) {
    std::error_code error;
    const bool folder = inputs.size() > 1 || fs::is_directory(fs::u8path(output), error);
    std::vector<std::string> out;
    if (!folder) {
        out.push_back(output);
    } else {
        if (output == "-") throw UsageError("-o - writes one WAVE to stdout, and " + std::to_string(inputs.size()) + " files were given; give -o a folder");
        std::set<std::string> names;
        for (const std::string & input : inputs) {
            const std::string name = fs::u8path(input).stem().u8string() + ".wav";
            if (!names.insert(name).second) {
                throw UsageError("-o names each enhanced file by its input's name, and two inputs are named " + fs::u8path(input).stem().u8string() +
                                 "; enhance them into folders of their own");
            }
            out.push_back((fs::u8path(output) / fs::u8path(name)).u8string());
        }
    }
    // An output that is an input given would replace it before it is read; equivalent() also sees the same file under
    // another case or through a link.
    for (const std::string & to : out) {
        for (const std::string & input : inputs) {
            std::error_code same;
            if (to != "-" && fs::equivalent(fs::u8path(to), fs::u8path(input), same)) {
                throw UsageError("-o would write " + to + " over " + input + ", a file given; give -o another file or a folder that holds none of them");
            }
        }
    }
    if (folder) {
        fs::create_directories(fs::u8path(output), error);
        if (error) throw io_failure("cannot make the folder " + output + " for -o: " + error.message());
    }
    return out;
}

/**
 * Enhances the 16-bit little-endian mono PCM on stdin at `rate` as it arrives, and writes it to `to` as it settles; the
 * PCM's seconds and the time they took go to `audio` and `took`.
 */
void enhance_stdin(speech_model * model, const speech_model_info * info, const std::vector<RequestOption> & options, int rate, const std::string & to,
                   FILE * out, double & audio) {
    const int enhanced = enhanced_rate(info, rate, options);
    Output file(to, out);
    WavWriter writer(file.file(), enhanced);
    terminal::Status status("Enhancing stdin");
    Enhancing enhancing{writer, status, "Enhancing stdin", (double) enhanced};
    const Request request = new_request(model);
    apply_options(request.get(), options);
    speech_enhancement * raw = nullptr;
    check(speech_enhancement_start(request.get(), rate, on_audio, &enhancing, &raw));
    const std::unique_ptr<speech_enhancement, decltype(&speech_enhancement_free)> enhancement(raw, speech_enhancement_free);
    std::vector<unsigned char> bytes(1 << 16);
    std::vector<float> samples;
    size_t carried = 0;
    uint64_t total = 0;
    speech_status pushed = SPEECH_OK;
    for (;;) {
        const size_t got = read_stdin(bytes.data() + carried, bytes.size() - carried);
        const size_t have = carried + got;
        samples.resize(have / 2);
        for (size_t i = 0; i < samples.size(); i++) samples[i] = (float) (int16_t) ((uint16_t) bytes[2 * i] | (uint16_t) bytes[2 * i + 1] << 8) / 32768.0f;
        carried = have % 2;
        if (carried) bytes[0] = bytes[have - 1];
        total += samples.size();
        if (!samples.empty()) pushed = check(speech_enhancement_push(enhancement.get(), samples.data(), samples.size()));
        if (pushed == SPEECH_CANCELLED || got == 0) break;
    }
    if (enhancing.failed) throw *enhancing.failed;
    if (carried) {
        throw Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "audio",
                      "stdin ended within a sample: 16-bit PCM has two bytes for each, and an odd number arrived");
    }
    check(speech_enhancement_end(enhancement.get()));
    if (enhancing.failed) throw *enhancing.failed;
    writer.finish();
    file.keep();
    audio = (double) total / rate;
}

int run_enhance(const CommandLine & line, FILE * out) {
    const std::optional<std::string> output = line.value("-o");
    if (!output) throw UsageError("name the WAVE file or folder to write with -o, or give -o - for stdout");
    const std::vector<std::string> inputs(line.args.begin() + 1, line.args.end());
    const bool piped = std::find(inputs.begin(), inputs.end(), "-") != inputs.end();
    if (piped && inputs.size() > 1) throw UsageError("- reads 16-bit PCM from stdin, alone; give it without files");
    if (line.has("--rate") && !piped) throw UsageError("--rate is the rate of the PCM that - reads from stdin");
    if (line.integer("--rate").value_or(1) <= 0) throw UsageError("--rate takes the PCM's rate in Hz");
    const std::vector<std::string> to = piped ? std::vector<std::string>{*output} : destinations(inputs, *output);

    // A model named for the first time is fetched here, before the load is timed.
    const std::string path = model_file(line.args[0]);
    require_options(path, line);
    const auto t0 = Clock::now();
    terminal::Status loading("Loading " + argument_label(line.args[0]));
    const Model model = load_model(path, line.loading(false));
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    loading.done(loaded_line({m}, seconds_since(t0)));
    if (line.has("-v")) std::fprintf(stderr, "speech.cpp %s\n", speech_version());

    if (piped) {
        const int rate = (int) line.integer("--rate").value_or(16000);
        std::fprintf(stderr, "reading 16-bit PCM at %d Hz from stdin\n", rate);
        const auto start = Clock::now();
        double audio = 0;
        enhance_stdin(model.get(), m, line.options, rate, *output, out, audio);
        const double took = seconds_since(start);
        char said[160];
        std::snprintf(said, sizeof said, "%.2f s of audio in %.3f s, RTFx %.2f", audio, took, audio / took);
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Audio, "stdin: " + terminal::paint(terminal::Color::Gray, said)).c_str());
        return 0;
    }
    double audio_total = 0, busy = 0;
    // Nothing cancels a run of the command line; a write that fails stops it through the audio callback.
    Cancellation cancellation;
    for (size_t i = 0; i < inputs.size(); i++) {
        const std::string & input = inputs[i];
        const std::string where = to[i] == "-" ? "" : " to " + to[i];
        const std::string doing = "Enhancing " + input + where;
        terminal::Status status(doing);
        const auto start = Clock::now();
        double audio = 0;
        int rate = 0;
        try {
            const Wav wav = read_audio(input);
            const std::vector<float> samples = wav.mono();
            audio = (double) samples.size() / wav.sample_rate;
            const Request request = enhancement_request(model.get(), samples, wav.sample_rate, line.options);
            rate = enhanced_rate(m, wav.sample_rate, line.options);
            Output file(to[i], out);
            WavWriter writer(file.file(), rate);
            Enhancing enhancing{writer, status, doing, (double) rate};
            enhance_audio(request.get(), on_audio, &enhancing, cancellation);
            if (enhancing.failed) throw *enhancing.failed;
            writer.finish();
            file.keep();
        } catch (const Failure & e) {
            throw Failure(e.code(), e.option(), input + ": " + e.what());
        }
        const double took = seconds_since(start);
        char said[160];
        std::snprintf(said, sizeof said, "%.2f s of audio at %d Hz in %.3f s, RTFx %.2f", audio, rate, took, audio / took);
        status.done(input + where + ": " + terminal::paint(terminal::Color::Gray, said), terminal::Mark::Audio);
        audio_total += audio;
        busy += took;
    }
    if (inputs.size() > 1) {
        char said[128];
        std::snprintf(said, sizeof said, "%zu files, %.2f s of audio in %.3f s, RTFx %.2f", inputs.size(), audio_total, busy, audio_total / busy);
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, said).c_str());
    }
    return 0;
}

}  // namespace

Command enhance_command() {
    Command c;
    c.name = "enhance";
    c.usage = "enhance MODEL -o FILE|DIR|- [options] AUDIO.wav...|-";
    c.summary = "remove noise and other degradations from WAVE files";
    c.description =
        "Enhances each WAVE file (16-, 24- or 32-bit PCM or 32-bit float, channels averaged) with an enhancement model,\n"
        "removing noise, reverberation and other degradations, and writes it as a 16-bit mono WAVE at the rate it was\n"
        "enhanced at, its own or --bwe's, as the audio comes: one file to the file -o names, or to stdout with -o -, and\n"
        "several to the folder -o names, made if it is not there, each as its own name with .wav. The model takes the\n"
        "audio in chunks of --chunk-size-in-seconds that start every --hop-length-portion of a chunk, RE-USE's\n"
        "inference_chunk.py's 5 s and 0.5 unless given; a chunk at least as long as a file enhances it whole. - reads\n"
        "16-bit little-endian mono PCM at --rate from stdin until it ends and writes the WAVE as each part of it settles,\n"
        "a chunk and a hop after it arrives; its samples are those of the whole audio. Enhancement can help a recognizer\n"
        "on noisy audio and hurt it on clean audio: try both.";
    c.flags = {
        {"-o", "FILE|DIR", false, "the WAVE file to write, a folder for several, or - for stdout"},
        {"--rate", "HZ", false, "the rate of the PCM that - reads from stdin, 16000 unless given"},
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release"),
    };
    c.request_options = true;
    c.model = ModelKind::Enhancement;
    c.min_args = 2;
    c.max_args = SIZE_MAX;
    c.run = run_enhance;
    return c;
}
