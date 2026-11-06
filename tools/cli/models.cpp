#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "cache.h"
#include "commands.h"
#include "fetch.h"
#include "terminal.h"

// The subcommands about the catalog of models a release fetches by name: speech models lists it with what is in the
// model folder and which model to start with for each language, speech pull fetches models ahead of their use, and
// speech rm removes them, or the files of earlier releases that this one names no more.

namespace fs = std::filesystem;

namespace {

/** A size in gigabytes, or in megabytes below 10 MB, which two decimals of a gigabyte would show as 0.00 GB. */
std::string gigabytes(uint64_t bytes) {
    char s[32];
    if (bytes < 10000000) std::snprintf(s, sizeof s, "%.1f MB", bytes / 1e6);
    else std::snprintf(s, sizeof s, "%.2f GB", bytes / 1e9);
    return s;
}

std::string padded(const std::string & s, size_t width) {
    return s.size() >= width ? s + "  " : s + std::string(width - s.size(), ' ');
}

/** The width of the column of types: the longest type and two spaces. */
size_t type_width() {
    size_t width = 4;
    for (const CatalogModel & m : catalog()) {
        for (const CatalogFile & f : m.files) width = std::max(width, f.type.size());
    }
    return width + 2;
}

std::string joined(const std::vector<std::string> & items) {
    std::string out;
    for (size_t i = 0; i < items.size(); i++) out += (i ? " " : "") + items[i];
    return out;
}

/** A path in the model folder as `speech models` shows it, from the folder down. */
std::string in_folder(const fs::path & path) {
    return path.lexically_relative(model_dir()).u8string();
}

/** `words` joined with spaces in at most `room` columns, the ones that do not fit counted after an ellipsis. */
std::string fitted(const std::vector<std::string> & words, size_t room) {
    const std::string whole = joined(words);
    if (whole.size() <= room) return whole;
    std::string out;
    for (size_t i = 0; i < words.size(); i++) {
        const std::string more = " \xE2\x80\xA6 " + std::to_string(words.size() - i) + " more";
        if (out.size() + (out.empty() ? 0 : 1) + words[i].size() + more.size() - 2 > room) return out + more;
        out += (out.empty() ? "" : " ") + words[i];
    }
    return out;
}

/**
 * The listing as a terminal shows it: the models by task, each file a row with a check where it is fetched, a model's
 * languages on the row of its own type alone and cut to the terminal's width.
 */
std::string drawn_listing(FILE * out, size_t width) {
    using terminal::Color;
    using terminal::paint;
    const size_t columns = (size_t) terminal::columns(out);
    std::string text = paint(Color::Bold, "Models of speech.cpp " + std::string(speech_version()), out) +
                       paint(Color::Gray, ", fetched into " + model_dir().u8string(), out) + "\n";
    for (const char * task : {"synthesis", "recognition", "detection", "diarization", "enhancement"}) {
        std::string title = task;
        title[0] = (char) std::toupper((unsigned char) title[0]);
        text += "\n" + paint(Color::Bold, title, out) + "\n";
        for (const CatalogModel & m : catalog()) {
            if (m.task != task) continue;
            for (const CatalogFile & f : m.files) {
                const Presence p = presence({&m, &f});
                const bool own = f.type == m.type;
                const std::string share = p.partial ? std::to_string(p.partial * 100 / f.size) + "%" : "";
                const std::string fetched = p.whole ? paint(Color::Mint, "\xE2\x9C\x93", out) + "   "
                                            : p.partial ? paint(Color::Peach, share, out) + std::string(4 - std::min<size_t>(share.size(), 3), ' ')
                                                        : paint(Color::Gray, "-", out) + "   ";
                const std::string row = "  " + paint(own ? Color::Bold : Color::Gray, padded(choice_name({&m, &f}), width), out) +
                                        padded(f.type, type_width()) + padded(gigabytes(f.size), 10) + fetched;
                const size_t used = 2 + width + type_width() + 10 + 4;
                std::string line = row + (own && columns > used + 8 ? fitted(m.languages, columns - used - 1) : "");
                while (!line.empty() && line.back() == ' ') line.pop_back();
                text += line + "\n";
            }
        }
    }
    text += "\n" + paint(Color::Bold, "Start with", out) + "\n";
    for (const char * task : {"synthesis", "recognition"}) {
        bool first = true;
        for (const CatalogModel & m : catalog()) {
            if (m.task != task || m.start.empty()) continue;
            const size_t used = 2 + 13 + width;
            text += "  " + paint(Color::Gray, padded(first ? task : "", 13), out) + padded(m.name, width) +
                    (columns > used + 8 ? fitted(m.start, columns - used - 1) : "") + "\n";
            first = false;
        }
    }
    text += "\n" + paint(Color::Gray, "A subcommand given a NAME[:TYPE] fetches its file the first time; speech pull fetches it ahead, and speech rm removes it.",
                         out) + "\n";
    return text;
}

int run_models(const CommandLine & line, FILE * out) {
    if (line.has("--json")) {
        std::fprintf(out, "%s\n", models_json().c_str());
        return 0;
    }
    size_t width = 4;
    for (const CatalogModel & m : catalog()) {
        for (const CatalogFile & f : m.files) width = std::max(width, choice_name({&m, &f}).size());
    }
    width += 2;
    if (terminal::draws_on(out)) {
        std::string text = drawn_listing(out, width);
        const std::vector<OldFile> old = old_files();
        if (!old.empty()) {
            text += "\nOld files, which no model of this release names; speech rm --old removes them:\n";
            for (const OldFile & f : old) text += "  " + padded(gigabytes(f.size), 9) + in_folder(f.path) + "\n";
        }
        std::fputs(text.c_str(), out);
        std::fflush(out);
        return 0;
    }
    std::string text = "The models of speech.cpp " + std::string(speech_version()) + ", fetched into " + model_dir().u8string() + "\n\n" +
                       padded("NAME", width) + padded("TYPE", type_width()) + "SIZE     FETCHED  TASK         LANGUAGES\n";
    for (const CatalogModel & m : catalog()) {
        for (const CatalogFile & f : m.files) {
            const Presence p = presence({&m, &f});
            const std::string fetched = p.whole ? "yes" : p.partial ? std::to_string(p.partial * 100 / f.size) + "%" : "no";
            text += padded(choice_name({&m, &f}), width) + padded(f.type, type_width()) + padded(gigabytes(f.size), 9) + padded(fetched, 9) + padded(m.task, 13) +
                    joined(m.languages) + "\n";
        }
    }
    text += "\nThe model to start with for each language:\n";
    for (const char * task : {"synthesis", "recognition"}) {
        bool first = true;
        for (const CatalogModel & m : catalog()) {
            if (m.task != task || m.start.empty()) continue;
            text += "  " + padded(first ? task : "", 13) + padded(m.name, width) + joined(m.start) + "\n";
            first = false;
        }
    }
    text += "\nA subcommand given a NAME[:TYPE] fetches its file the first time; speech pull fetches it ahead, and speech rm removes it.\n";
    const std::vector<OldFile> old = old_files();
    if (!old.empty()) {
        text += "\nOld files, which no model of this release names; speech rm --old removes them:\n";
        for (const OldFile & f : old) text += "  " + padded(gigabytes(f.size), 9) + in_folder(f.path) + "\n";
    }
    std::fputs(text.c_str(), out);
    std::fflush(out);
    return 0;
}

/** The catalog's files the arguments name, every one checked before any is fetched or removed. */
std::vector<CatalogChoice> choices(const std::vector<std::string> & args) {
    std::vector<CatalogChoice> out;
    for (const std::string & a : args) {
        if (is_model_path(a)) throw UsageError("\"" + a + "\" is a model file; give the name of a model `speech models` lists");
        out.push_back(find_model(a));
    }
    return out;
}

int run_pull(const CommandLine & line, FILE * out) {
    for (const CatalogChoice & c : choices(line.args)) {
        const Presence p = presence(c);
        if (p.whole) std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, choice_name(c) + " is fetched already").c_str());
        const fs::path path = p.whole ? model_path(c) : fetch_on_stderr(c);
        std::fprintf(out, "%s\n", path.u8string().c_str());
        std::fflush(out);
    }
    return 0;
}

int run_rm(const CommandLine & line, FILE *) {
    const bool old = line.has("--old");
    if (line.args.empty() && !old) throw UsageError("name the models to remove, or give --old for the files of earlier releases");
    const std::vector<CatalogChoice> named = choices(line.args);
    for (const CatalogChoice & c : named) {
        const Presence p = presence(c);
        if (!p.whole && !p.partial) {
            throw Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "", choice_name(c) + " is not in " + model_dir().u8string() +
                                                                                     "; speech models says what is");
        }
    }
    for (const CatalogChoice & c : named) {
        const uint64_t removed = remove_from_folder(model_path(c));
        std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, "Removed " + choice_name(c) + ", " + gigabytes(removed)).c_str());
    }
    if (old) {
        const std::vector<OldFile> files = old_files();
        if (files.empty()) std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, "No old files in " + model_dir().u8string()).c_str());
        for (const OldFile & f : files) {
            const uint64_t removed = remove_from_folder(f.path);
            std::fprintf(stderr, "%s\n", terminal::marked(terminal::Mark::Done, "Removed " + in_folder(f.path) + ", " + gigabytes(removed)).c_str());
        }
    }
    return 0;
}

}  // namespace

Command models_command() {
    Command c;
    c.name = "models";
    c.usage = "models [--json]";
    c.summary = "list the models a NAME fetches, what is fetched, and which to start with";
    c.description =
        "Lists the catalog of this release: each model's NAME[:TYPE], which a subcommand takes in place of a model file and\n"
        "fetches from Hugging Face the first time, with its type, size, whether it is fetched, its task and languages; for\n"
        "each language the model to start with; and the files in the model folder that no model of this release names.\n"
        "The folder is SPEECH_MODEL_DIR, or speech.cpp/models in the system's cache folder.";
    c.flags = {
        {"--json", "", false, "print one JSON object with every pin of the catalog, each file's path and whether it is fetched"},
    };
    c.run = run_models;
    return c;
}

Command pull_command() {
    Command c;
    c.name = "pull";
    c.usage = "pull NAME[:TYPE]...";
    c.summary = "fetch models ahead of their use";
    c.description =
        "Fetches each model's file from Hugging Face into the model folder unless it is there, checks its size and SHA-256\n"
        "against the catalog, and prints its path on stdout. A fetch that stops resumes where it stopped the next time, and\n"
        "a process that fetches the same file waits for this one. Progress goes to stderr.";
    c.min_args = 1;
    c.max_args = SIZE_MAX;
    c.run = run_pull;
    return c;
}

Command rm_command() {
    Command c;
    c.name = "rm";
    c.usage = "rm NAME[:TYPE]... | speech rm --old";
    c.summary = "remove fetched models, or the files of earlier releases";
    c.description =
        "Removes each model's file from the model folder, with what was fetched of it, or with --old every file there that\n"
        "no model of this release names. Nothing in the folder is removed otherwise.";
    c.flags = {
        {"--old", "", false, "remove the files no model of this release names, which speech models lists as old"},
    };
    c.max_args = SIZE_MAX;
    c.run = run_rm;
    return c;
}
