#pragma once

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine.h"
#include "speech.h"

// What the files of the C API share: the table of families, the objects behind the opaque handles, and the way a
// failure becomes a status.

/**
 * A family behind the C API, one line of the table in speech.cpp: its task, the layout its reader takes, which names
 * the general.architecture of its files, its table of options and what else the information shows (describe), its
 * engine (load), given the backend of the model's device and the threads the CPU computes with, and, for a family that
 * takes voice files, how it makes one of a VoiceRecipe.
 */
struct Family {
    speech_task task;
    const Layout & layout;
    FamilyInfo (*describe)(const std::shared_ptr<const ModelFile> & file);
    std::unique_ptr<Engine> (*load)(const std::string & path, ggml_backend_t backend, int threads);
    void (*make_voice)(const std::string & model_path, const VoiceRecipe & recipe, const std::string & voice_path, ggml_backend_t backend);
    /** Selects an unset CPU thread preference from the machine default before any backend or engine is created. */
    int (*auto_threads)(const ModelFile & file, ggml_backend_dev_t device, int machine_default) = nullptr;
};

/** The family of the model file at `path`, by its general.architecture; an architecture no family has throws. */
const Family & family_of(const std::string & path);

/**
 * What a model file says of its model, read once from its metadata and shared by every information made of the file or
 * of a model loaded from it. It keeps the file's metadata, from which it writes the metadata's values and makes the
 * tokenizer of the token counts when they are first asked for.
 */
struct FileInfo {
    const Family * family = nullptr;
    std::shared_ptr<const ModelFile> file;
    ModelIdentity identity;
    int sample_rate = 0;
    std::vector<std::string> languages;
    FamilyInfo described;
    uint64_t file_bytes = 0, weight_bytes = 0;

    /** The family's declaration of `option`, or null when the model does not take it. */
    const OptionSpec * spec(speech_option option) const;
    /** What the file lacks for `option`, or for the voice `voice` of SPEECH_OPT_VOICE, or null. */
    const Lack * lack(speech_option option, const std::string & voice = "") const;
    /** The key and the JSON value of a metadata entry, which live as long as this. */
    const char * meta_key(size_t index) const;
    const char * meta_value(size_t index) const;

private:
    mutable std::mutex meta_mutex_;
    mutable std::vector<std::unique_ptr<std::string>> meta_keys_, meta_values_;
};

/** Reads the information of the model file at `path` from its metadata. */
std::shared_ptr<const FileInfo> read_file_info(const std::string & path);

struct speech_load_params {
    std::string device = "auto";
    /** 0 for the library's default. */
    int threads = 0;
    bool warmup = false;
};

struct speech_model_info {
    std::shared_ptr<const FileInfo> file;
    /** The model's own voices, then those added to a loaded model. */
    std::vector<VoiceInfo> voices;
    /** The device of a loaded model, which information read from a file has not. */
    std::optional<std::string> device;
    int threads = 0;
    std::string json;
};

/** The information of a file, or of a loaded model with its device, threads and added voices. */
speech_model_info * make_info(std::shared_ptr<const FileInfo> file, std::vector<VoiceInfo> added, std::optional<std::string> device,
                              int threads);

struct speech_model {
    std::shared_ptr<const FileInfo> file;
    /** Freed after the engine, which computes on it. */
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend{nullptr, ggml_backend_free};
    std::unique_ptr<Engine> engine;
    std::string device;
    int threads = 0;
    /** Held while a request runs or a voice is added, so that the model does one thing at a time. */
    std::mutex busy;

    /** The voices added since loading. */
    std::vector<VoiceInfo> added() const {
        std::lock_guard<std::mutex> lock(voices_mutex_);
        return added_;
    }
    void add(VoiceInfo voice) {
        std::lock_guard<std::mutex> lock(voices_mutex_);
        added_.push_back(std::move(voice));
    }
    /** Whether the model has a voice of `name`, compared with case. */
    bool has_voice(const std::string & name) const;

private:
    mutable std::mutex voices_mutex_;
    std::vector<VoiceInfo> added_;
};

struct speech_result {
    speech_task task = SPEECH_TASK_SYNTHESIS;
    speech_stop stop = SPEECH_STOP_COMPLETE;
    int64_t seed = -1;
    uint64_t samples = 0;
    std::string text;
    std::vector<TimedText> segments, tokens;
    /** The speaker of each segment of a diarization, and of a recognition that set speakers. */
    std::vector<int> speakers;
    std::vector<std::string> languages;
    /** Why a recognition that set timestamps or speakers has no times. */
    std::optional<std::string> untimed;
};

struct speech_request {
    speech_model * model = nullptr;
    std::optional<std::string> text;
    std::vector<float> audio;
    int sample_rate = 0;
    /** The options set, each checked as it was set. */
    std::map<speech_option, OptionValue> values;
    speech_progress_callback on_progress = nullptr;
    void * progress_data = nullptr;
    std::atomic<bool> cancelled{false};
    /**
     * Whether a run has started its work, which a request allows once. A run that the request's checks refuse before
     * any work leaves the request to be fixed and run again.
     */
    bool ran = false;
    std::unique_ptr<speech_result> result;
};

/**
 * The request's options as its run takes them: those it set, the defaults, and a seed drawn where it took one. A
 * required option left out throws, naming it.
 */
RequestValues run_values(const speech_request & request);

/** A refusal of the C API itself: its status, its message, and the input it concerns, or none. */
class ApiError : public std::runtime_error {
public:
    ApiError(speech_status status, const std::string & message, const char * input = nullptr)
        : std::runtime_error(message), status_(status), input_(input ? input : "") {}

    speech_status status() const { return status_; }
    const std::string & input() const { return input_; }

private:
    speech_status status_;
    std::string input_;
};

/**
 * The status of the exception being handled, which it records as the calling thread's last error with the input it
 * concerns: an ApiError's own, a family's Error's by its kind, and any other exception's as a defect of the library.
 */
speech_status record_failure();

/** Runs `body`, which returns a status, turning what it throws into one. */
template <typename Body>
speech_status guarded(Body && body) {
    try {
        return body();
    } catch (...) {
        return record_failure();
    }
}

/**
 * Runs `body` for a function that has no status to return, recording what it throws as the last error and returning
 * `failed` instead.
 */
template <typename T, typename Body>
T quietly(T failed, Body && body) {
    try {
        return body();
    } catch (...) {
        record_failure();
        return failed;
    }
}

/** Throws SPEECH_ERROR_INVALID_ARGUMENT naming `input` when `pointer` is NULL. */
inline void require(const void * pointer, const char * what, const char * input = nullptr) {
    if (!pointer) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string(what) + " is NULL", input);
}

/** The name of a task as messages and the information write it. */
inline const char * task_name(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "synthesis";
        case SPEECH_TASK_RECOGNITION: return "recognition";
        case SPEECH_TASK_DETECTION: return "detection";
        case SPEECH_TASK_DIARIZATION: return "diarization";
        case SPEECH_TASK_ENHANCEMENT: return "enhancement";
    }
    throw std::logic_error("a task the library does not know");
}

/** The call that runs a request of a model of `task`, for a message that names it. */
inline const char * call_of(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "speech_synthesize() to speak with it";
        case SPEECH_TASK_RECOGNITION: return "speech_transcribe() to recognize speech with it";
        case SPEECH_TASK_DETECTION: return "speech_detect() to find where someone speaks";
        case SPEECH_TASK_DIARIZATION: return "speech_diarize() to find who speaks where";
        case SPEECH_TASK_ENHANCEMENT: return "speech_enhance() to enhance speech with it";
    }
    throw std::logic_error("a task the library does not know");
}

/**
 * Whether a model of `task` has languages, which its file names in general.languages: a detection, diarization or
 * enhancement model takes none.
 */
inline bool task_has_languages(speech_task task) {
    return task == SPEECH_TASK_SYNTHESIS || task == SPEECH_TASK_RECOGNITION;
}
