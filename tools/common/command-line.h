#pragma once

#include <cstddef>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "catalog.h"
#include "library.h"
#include "request-options.h"

// The one parser of every subcommand's command line. A subcommand declares its flags and how many arguments it
// takes; one that runs requests also takes every option of the C API's vocabulary as a flag of its name in kebab-case,
// read by the option's type, so that an option the library adds is a flag without the parser changing. A subcommand
// that holds a model of each task also takes each option after the name of a task's subcommand and a dot, as
// --asr.language or --diarize.streaming-mode, for the model of that task alone. A flag's value
// follows it or an "=" ("--seed 7", "--seed=7"), but for a boolean option, which is true alone and takes "=true" or
// "=false" ("--do-sample=false"); "--" ends the flags, so that an argument may begin with "-".

/** A request option given for the model of one task, as --asr.language gives it for the recognition model. */
struct TaskOption {
    speech_task task;
    RequestOption option;
};

/**
 * The prefix of a task's options on the command line, the name of its subcommand: "tts", "asr", "vad", "diarize" or
 * "enhance".
 */
const char * task_prefix(speech_task task);

/** A flag of a subcommand other than a request option. */
struct Flag {
    /** "--device" or "-o". */
    std::string name;
    /** What the value is, for the help ("NAME"), or "" for a flag that takes none. */
    std::string value;
    bool repeatable = false;
    std::string help;
};

class CommandLine;

struct Command {
    std::string name;
    /** The usage after "speech ", such as "tts MODEL -o FILE|- [options] [TEXT]". */
    std::string usage;
    /** One line for `speech --help`. */
    std::string summary;
    /** What the subcommand does, for its --help. */
    std::string description;
    std::vector<Flag> flags;
    /** Whether every request option of the vocabulary is a flag of the subcommand. */
    bool request_options = false;
    /**
     * Whether a request option's flag also takes a task's prefix, --tts., --asr., --vad., --diarize. or --enhance., for that
     * task's model.
     */
    bool task_options = false;
    /**
     * For a subcommand whose first argument is MODEL, the models that fit it: a command line without one is told what
     * MODEL is and which model to start with.
     */
    std::optional<ModelKind> model;
    size_t min_args = 0;
    size_t max_args = 0;
    int (*run)(const CommandLine & line, FILE * out) = nullptr;
    /**
     * Reports a failure that ends the subcommand on `out` as well as on stderr, or nullptr: the worker's caller reads
     * the protocol alone, so the worker answers it with a `fatal` line.
     */
    void (*report)(const Failure & failure, FILE * out) = nullptr;
};

/** A subcommand's command line as read, each flag's value checked against the flag's type. */
class CommandLine {
public:
    explicit CommandLine(const Command & command) : command_(&command) {}

    const Command & command() const { return *command_; }
    /** The arguments that are not flags, in their order. */
    std::vector<std::string> args;
    /** The request options given, in their order. */
    std::vector<RequestOption> options;
    /** The request options given with a task's prefix, in their order. */
    std::vector<TaskOption> task_options;

    bool has(const std::string & flag) const;
    /** The value of a flag given at most once, or `absent`. */
    std::optional<std::string> value(const std::string & flag) const;
    /** The values of a repeatable flag in their order. */
    std::vector<std::string> values(const std::string & flag) const;
    /** The value of a flag read as a whole number, or `absent`; any other text is a UsageError. */
    std::optional<int> integer(const std::string & flag) const;

    /**
     * The model loading the command line asks for: --device, or `device` when it is not given, --threads and the
     * voices of --add-voice NAME=FILE, with a warm-up unless --no-warmup is a flag of the subcommand and given or
     * `warmup` is false.
     */
    Loading loading(bool warmup, const std::optional<std::string> & device = std::nullopt) const;

private:
    friend CommandLine parse_command_line(const Command & command, const std::vector<std::string> & args);

    struct Given {
        std::string flag, value;
    };
    const Command * command_;
    std::vector<Given> given_;
};

/** Reads `args`, the command line after the subcommand's name; a command line that cannot be run throws a UsageError. */
CommandLine parse_command_line(const Command & command, const std::vector<std::string> & args);

/** Whether `args` asks for the subcommand's help with --help or -h before any "--". */
bool asks_for_help(const std::vector<std::string> & args);

/**
 * The subcommand's --help: its usage, what it does, its flags and, where it takes them, the request options, colored
 * where `stream`, on which it is written, has colors.
 */
std::string command_help(const Command & command, FILE * stream);

/** Flags that several subcommands share. */
Flag device_flag(const std::string & help);
Flag threads_flag();
Flag add_voice_flag();
Flag no_warmup_flag();
Flag verbose_flag(const std::string & help);
