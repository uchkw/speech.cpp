#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "commands.h"
#include "fetch.h"
#include "json.h"
#include "regions.h"
#include "terminal.h"
#include "wave-input.h"

// speech vad: finds where someone speaks in WAVE files and writes each file's regions to stdout in the order the files
// are given, as text or as one JSON object per file, and with --split each region as a WAVE file of its own, and
// reports on stderr the load time and, for each file, its seconds of audio, the time to its regions and the RTFx, the
// seconds of audio over the seconds they took.

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

namespace fs = std::filesystem;

/**
 * Writes samples as a mono 16-bit WAVE file, each sample scaled by 32768 and clamped, the inverse of the reader's
 * scaling, so that a region of a 16-bit file keeps its samples as they were.
 */
void write_region(const fs::path & path, const float * samples, size_t n, int rate) {
    std::string bytes;
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; i++) bytes += (char) ((v >> (8 * i)) & 0xFF); };
    auto u16 = [&](uint16_t v) { bytes += (char) (v & 0xFF); bytes += (char) (v >> 8); };
    bytes += "RIFF";
    u32((uint32_t) (36 + 2 * n));
    bytes += "WAVEfmt ";
    u32(16);
    u16(1);
    u16(1);
    u32((uint32_t) rate);
    u32((uint32_t) rate * 2);
    u16(2);
    u16(16);
    bytes += "data";
    u32((uint32_t) (2 * n));
    for (size_t i = 0; i < n; i++) u16((uint16_t) (int16_t) std::max(-32768.0f, std::min(32767.0f, std::round(samples[i] * 32768.0f))));
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), (std::streamsize) bytes.size());
    if (!f.flush()) throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot write " + path.u8string() + "; check the folder of --split");
}

/** The name of a file's region in --split's folder: its stem and its number from 1, as many digits as the last one has. */
std::string region_name(const std::string & stem, size_t number, size_t count) {
    std::string digits = std::to_string(number);
    digits.insert(0, std::to_string(count).size() - digits.size(), '0');
    return stem + "-" + digits + ".wav";
}

int run_vad(const CommandLine & line, FILE * out) {
    const std::string format = line.value("--format").value_or("text");
    if (format != "text" && format != "json") throw UsageError("--format takes text or json, not \"" + format + "\"");

    // A model named for the first time is fetched here, before the load is timed.
    const std::string path = model_file(line.args[0]);
    require_options(path, line);
    const auto t0 = Clock::now();
    // The device and threads of every other detection, unless the command line names its own.
    Loading loading = line.loading(false);
    const Loading usual = detection_loading(loading);
    if (!loading.device) loading.device = usual.device;
    if (!loading.threads) loading.threads = usual.threads;
    terminal::Status status("Loading " + argument_label(line.args[0]));
    const Model model = load_model(path, loading);
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    status.done(loaded_line({m}, seconds_since(t0)));
    if (line.has("-v")) std::fprintf(stderr, "speech.cpp %s, %d Hz\n", speech_version(), speech_model_info_sample_rate(m));
    const std::optional<std::string> split = line.value("--split");
    if (split) {
        // Two files of one stem would write their regions under the same names.
        std::set<std::string> stems;
        for (size_t i = 1; i < line.args.size(); i++) {
            if (!stems.insert(fs::u8path(line.args[i]).stem().u8string()).second) {
                throw UsageError("--split names each region by its file's name, and two files are named " + fs::u8path(line.args[i]).stem().u8string() +
                                 "; split them into folders of their own");
            }
        }
        std::error_code error;
        fs::create_directories(fs::u8path(*split), error);
        if (error) throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot make the folder " + *split + " for --split: " + error.message());
    }

    double audio_total = 0, busy = 0;
    for (size_t i = 1; i < line.args.size(); i++) {
        const std::string & path = line.args[i];
        std::vector<Region> regions;
        std::vector<float> samples;
        int rate = 0;
        double audio = 0;
        const auto start = Clock::now();
        {
            // The line is erased before the regions reach stdout, which may be the same terminal.
            const terminal::Status status("Detecting speech in " + path);
            try {
                const Wav wav = read_audio(path);
                samples = wav.mono();
                rate = wav.sample_rate;
                audio = (double) samples.size() / rate;
                const Request request = detection_request(model.get(), samples, rate, line.options);
                Cancellation never;
                regions = *detect_regions(request.get(), never);
            } catch (const Failure & e) {
                throw Failure(e.code(), e.option(), path + ": " + e.what());
            }
        }
        const double took = seconds_since(start);
        const size_t n = regions.size();
        if (split) {
            const std::string stem = fs::u8path(path).stem().u8string();
            for (size_t k = 0; k < n; k++) {
                const fs::path to = fs::u8path(*split) / fs::u8path(region_name(stem, k + 1, n));
                // A region's name can be another file's given here, as foo.wav's first region is foo-1.wav, and writing
                // it would replace that file, read already or not; equivalent() also sees the same file under another
                // case or through a link.
                for (size_t j = 1; j < line.args.size(); j++) {
                    std::error_code same;
                    if (fs::equivalent(to, fs::u8path(line.args[j]), same)) {
                        throw UsageError("--split would write " + to.u8string() + " over " + line.args[j] +
                                         ", a file given; give --split a folder that holds none of them");
                    }
                }
                const auto [first, last] = region_samples(regions[k], rate, samples.size());
                write_region(to, samples.data() + first, last - first, rate);
            }
        }
        std::string json = "{\"file\":" + json_string(path) + ",\"regions\":[";
        for (size_t k = 0; k < n; k++) {
            const double from = regions[k].start, to = regions[k].end;
            if (format == "json") json += std::string(k ? "," : "") + "{\"start\":" + json_number(from) + ",\"end\":" + json_number(to) + "}";
            else std::fprintf(out, "%s\t%.3f\t%.3f\n", path.c_str(), from, to);
        }
        if (format == "json") std::fprintf(out, "%s]}\n", json.c_str());
        std::fflush(out);
        char said[128];
        std::snprintf(said, sizeof said, "%.2f s of audio in %.3f s, RTFx %.0f, %zu region%s", audio, took, audio / took, n, n == 1 ? "" : "s");
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, path + ": " + terminal::paint(terminal::Color::Gray, said)).c_str());
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

Command vad_command() {
    Command c;
    c.name = "vad";
    c.usage = "vad MODEL [options] AUDIO.wav...";
    c.summary = "write where someone speaks in WAVE files";
    c.description =
        "Finds the regions where someone speaks in each WAVE file (16-, 24- or 32-bit PCM or 32-bit float, any rate,\n"
        "channels averaged) with a detection model and writes them to stdout in the order given: with --format text\n"
        "one line per region, FILE<TAB>START<TAB>END, times in seconds; with --format json one object per file,\n"
        "{\"file\", \"regions\": [{\"start\", \"end\"}]}. A file without speech has no line and no region. The options\n"
        "--threshold, --min-speech-duration-ms, --min-silence-duration-ms, --speech-pad-ms and\n"
        "--max-speech-duration-s are Silero VAD's get_speech_timestamps(), with its defaults. --split DIR also writes each\n"
        "region as a mono 16-bit WAVE file at the file's rate in DIR, FILE-1.wav, FILE-2.wav and so on, numbered with as\n"
        "many digits as the last number has so that they sort in order.";
    c.flags = {
        {"--format", "text|json", false, "text (the default) or one JSON object per file and line"},
        {"--split", "DIR", false, "also write each region as a WAVE file in DIR, made if it is not there"},
        device_flag("cpu (the default, with one thread unless --threads says otherwise), auto, gpu or a name `speech devices` lists"),
        threads_flag(),
        verbose_flag("also report the release and the sample rate"),
    };
    c.request_options = true;
    c.model = ModelKind::Detection;
    c.min_args = 2;
    c.max_args = SIZE_MAX;
    c.run = run_vad;
    return c;
}
