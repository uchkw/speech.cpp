// speech: speech synthesis, speech recognition and the detection of speech with every model speech.cpp runs, through
// its C API, as one executable with a subcommand per program: tts, asr and vad for the command line, voice, info and
// devices about models and devices, models, pull and rm about the models it fetches by name, quantize to write a model file in another
// weight type, update to explain source updates, serve for OpenAI's audio API over HTTP and worker for JSON Lines on
// stdin and stdout.
//
// usage: speech <subcommand> [options]     speech <subcommand> --help     speech --version
//
// It exits with 0 when done, 1 on a failure, which it reports as "speech: <code> (<option>): <message>" with the
// library's category, 2 on a command line it cannot run, and 3 when `speech tts` or `speech asr` stopped a request at
// the most the model makes: the longest speech, or the most tokens of a text. Its stdout carries its output alone:
// whatever ggml or a driver prints there goes to stderr.

#include <cstdio>
#include <exception>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "args.h"
#include "commands.h"
#include "spelling.h"
#include "take-stdout.h"
#include "terminal.h"

namespace {

/** The subcommands in the groups the overview lists them in. */
std::vector<std::pair<const char *, std::vector<Command>>> groups() {
    return {
        {"Speak and transcribe", {tts_command(), asr_command(), vad_command(), diarize_command(), enhance_command()}},
        {"Serve", {serve_command(), worker_command()}},
        {"Models", {models_command(), pull_command(), rm_command(), info_command(), voice_command(), quantize_command()}},
        {"This installation", {devices_command(), update_command()}},
    };
}

std::vector<Command> commands() {
    std::vector<Command> all;
    for (const auto & [title, members] : groups()) all.insert(all.end(), members.begin(), members.end());
    return all;
}

/** `speech --help`, colored where `stream` has colors. */
std::string overview(FILE * stream) {
    using terminal::Color;
    using terminal::paint;
    const std::string logo = terminal::logo(stream);
    std::string out = (logo.empty() ? "" : logo + " ") + paint(Color::Bold, "speech.cpp", stream) + " " + speech_version() +
                      ", speech synthesis, recognition and detection on ggml\n\nusage: speech <subcommand> [options]\n";
    for (const auto & [title, members] : groups()) {
        out += "\n" + paint(Color::Bold, title, stream) + "\n";
        for (const Command & c : members) {
            out += "  " + paint(Color::Lavender, c.name + std::string(10 - c.name.size(), ' '), stream) + c.summary + "\n";
        }
    }
    out += "\n" + paint(Color::Bold, "Try", stream) + "\n" +
           "  speech tts qwen3-tts-0.6b --voice ryan -o hello.wav \"Hello there.\"\n"
           "  speech asr qwen3-asr-0.6b hello.wav\n"
           "  speech serve --open\n"
           "\n" + paint(Color::Gray, "  speech <subcommand> --help   the subcommand's usage and options\n"
                                     "  speech --version             the release and the version of the C API", stream) + "\n";
    return out;
}

void report(const Failure & e) {
    const std::string head = "speech: " + e.code() + (e.option().empty() ? "" : " (" + e.option() + ")") + ":";
    std::fprintf(stderr, "%s %s\n", terminal::paint(terminal::Color::Red, head).c_str(), e.what());
}

int run(const std::vector<std::string> & args, FILE * out) {
    if (args.size() < 2) {
        std::fputs(overview(stderr).c_str(), stderr);
        return 2;
    }
    const std::string & name = args[1];
    if (name == "--version") {
        std::fprintf(out, "speech.cpp %s, C API %d.%d\n", speech_version(), speech_api_version_major(), speech_api_version_minor());
        return 0;
    }
    if (name == "--help" || name == "-h") {
        std::fputs(overview(out).c_str(), out);
        return 0;
    }
    const std::vector<Command> all = commands();
    const Command * command = nullptr;
    std::vector<std::string> names;
    for (const Command & c : all) {
        if (c.name == name) command = &c;
        names.push_back(c.name);
    }
    if (!command) {
        const std::string near = nearest(name, names);
        std::fprintf(stderr, "%s there is no subcommand %s%s\n\n%s", terminal::paint(terminal::Color::Red, "speech:").c_str(), name.c_str(),
                     near.empty() ? "" : ("; did you mean " + near + "?").c_str(), overview(stderr).c_str());
        return 2;
    }
    const std::vector<std::string> rest(args.begin() + 2, args.end());
    if (asks_for_help(rest)) {
        std::fputs(command_help(*command, out).c_str(), out);
        return 0;
    }
    int status = 0;
    try {
        const CommandLine line = parse_command_line(*command, rest);
        status = command->run(line, out);
    } catch (const UsageError & e) {
        std::fprintf(stderr, "%s %s\n%s\n", terminal::paint(terminal::Color::Red, "speech:").c_str(), e.what(),
                     terminal::paint(terminal::Color::Gray, "Run speech " + command->name + " --help for its usage and options.").c_str());
        if (command->report) command->report(Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "", e.what()), out);
        return 2;
    } catch (const Failure & e) {
        report(e);
        if (command->report) command->report(e, out);
        status = 1;
    } catch (const std::exception & e) {
        const Failure internal(speech_status_name(SPEECH_ERROR_INTERNAL), "", e.what());
        report(internal);
        if (command->report) command->report(internal, out);
        status = 1;
    }
    std::fflush(out);
    return status;
}

}  // namespace

int main(int argc, char ** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stdin), _O_BINARY);
#endif
    FILE * out = nullptr;
    try {
        out = take_stdout();
    } catch (const std::exception & e) {
        std::fprintf(stderr, "speech: io: %s\n", e.what());
        return 1;
    }
    terminal::setup(out);
    const int status = run(utf8_args(argc, argv), out);
    std::fflush(out);
    return status;
}
