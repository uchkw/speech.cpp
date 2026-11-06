#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "api.h"
#include "error.h"
#include "language.h"
#include "resample.h"
#include "speech.h"
#include "utf8.h"
#include "worker-trace.h"

// Requests: each value checked against the model's table as it is set, the whole request when it runs, its
// cancellation from any thread, and its result.

namespace {

std::string number(double v) {
    char s[32];
    std::snprintf(s, sizeof s, "%g", v);
    return s;
}

const char * type_name(speech_type type) {
    switch (type) {
        case SPEECH_TYPE_STRING: return "a string";
        case SPEECH_TYPE_INT: return "an integer";
        case SPEECH_TYPE_FLOAT: return "a number";
        case SPEECH_TYPE_BOOL: return "a boolean";
    }
    return "";
}

/** Whether `value` is the neutral value of `option`, which every model accepts whether or not it takes the option. */
bool neutral(speech_option option, const OptionValue & value) {
    switch (option) {
        case SPEECH_OPT_SPEED:
        case SPEECH_OPT_DURATION_SCALE: return std::get<double>(value) == 1;
        case SPEECH_OPT_LANGUAGE: return language_is_auto(std::get<std::string>(value));
        case SPEECH_OPT_TIMESTAMPS:
        case SPEECH_OPT_SPEAKERS: return !std::get<bool>(value);
        case SPEECH_OPT_PROMPT:
        case SPEECH_OPT_INSTRUCTIONS:
        case SPEECH_OPT_KEYWORDS: return std::get<std::string>(value).empty();
        default: return false;
    }
}

std::string joined(const std::vector<std::string> & items) {
    std::string out;
    for (const std::string & s : items) out += (out.empty() ? "" : ", ") + s;
    return out;
}

/** Throws unless `value` of `option`, given through the setter of `type`, is one the request's model takes. */
void check(const speech_request & request, speech_option option, speech_type type, const OptionValue & value) {
    const char * name = speech_option_name(option);
    if (!name) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the option " + std::to_string((int) option) + " is not one speech.cpp knows; it knows " +
                                                          std::to_string(speech_option_count()) + " options, from 0");
    }
    if (speech_option_type(option) != type) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + name + " takes " + type_name(speech_option_type(option)) +
                                                          ", not " + type_name(type) + "; set it with its own setter",
                       name);
    }
    if (type == SPEECH_TYPE_FLOAT && !std::isfinite(std::get<double>(value))) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + name + " is not a finite number", name);
    }
    if (type == SPEECH_TYPE_STRING && !decode_utf8(std::get<std::string>(value))) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the option ") + name + " is not UTF-8 text; give it in UTF-8", name);
    }
    const FileInfo & file = *request.model->file;
    const OptionSpec * spec = file.spec(option);
    if (!spec) {
        if (neutral(option, value)) return;
        if (const Lack * lack = file.lack(option)) throw ApiError(SPEECH_ERROR_UNSUPPORTED, lack->message, name);
        throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " does not take the option " + name +
                                                     (option == SPEECH_OPT_SPEED || option == SPEECH_OPT_DURATION_SCALE ? " but at 1" : "") +
                                                     "; leave it out",
                       name);
    }
    if (type == SPEECH_TYPE_INT || type == SPEECH_TYPE_FLOAT) {
        const double v = type == SPEECH_TYPE_INT ? (double) std::get<int64_t>(value) : std::get<double>(value);
        const bool low = spec->minimum_exclusive ? !(v > spec->minimum) : !(v >= spec->minimum);
        if (low || !(v <= spec->maximum)) {
            const auto bound = [&](double b) { return type == SPEECH_TYPE_INT ? std::to_string((int64_t) b) : number(b); };
            std::string range = std::isfinite(spec->minimum) ? (spec->minimum_exclusive ? "above " : "from ") + bound(spec->minimum) : "";
            if (std::isfinite(spec->maximum)) range += (range.empty() ? "up " : " ") + std::string("to ") + bound(spec->maximum);
            const std::string given = type == SPEECH_TYPE_INT ? std::to_string(std::get<int64_t>(value)) : number(std::get<double>(value));
            throw ApiError(SPEECH_ERROR_OUT_OF_RANGE, std::string("the option ") + name + " is " + given + "; " + file.identity.name + " takes it " + range,
                           name);
        }
    }
    if (option == SPEECH_OPT_VOICE) {
        const std::string & voice = std::get<std::string>(value);
        if (!request.model->has_voice(voice)) {
            if (const Lack * lack = file.lack(option, voice)) throw ApiError(SPEECH_ERROR_OUT_OF_RANGE, lack->message, name);
            std::vector<std::string> names;
            for (const VoiceInfo & v : file.described.voices) names.push_back(v.name);
            for (const VoiceInfo & v : request.model->added()) names.push_back(v.name);
            throw ApiError(SPEECH_ERROR_OUT_OF_RANGE, file.identity.name + " has no voice named \"" + voice + "\"; " +
                                                          (names.empty() ? "add one with speech_voice_add()" : "its voices are " + joined(names)),
                           name);
        }
    }
    if (!spec->choices.empty()) {
        const std::string & v = std::get<std::string>(value);
        if (std::find(spec->choices.begin(), spec->choices.end(), v) == spec->choices.end()) {
            throw ApiError(SPEECH_ERROR_OUT_OF_RANGE, std::string("the option ") + name + " is \"" + v + "\"; " + file.identity.name + " takes one of " +
                                                          joined(spec->choices),
                           name);
        }
    }
    if (option == SPEECH_OPT_LANGUAGE) {
        const std::string & tag = std::get<std::string>(value);
        if (language_is_auto(tag)) return;
        for (const std::string & language : file.languages) {
            if (bcp47_matches(tag, language)) return;
        }
        throw ApiError(SPEECH_ERROR_OUT_OF_RANGE, file.identity.name + (file.family->task == SPEECH_TASK_SYNTHESIS ? " speaks " : " recognizes ") +
                                                      joined(file.languages) + ", not \"" + tag + "\"; give one of them, a region or script of one, or auto",
                       name);
    }
}

speech_status set_value(speech_request * request, speech_option option, speech_type type, OptionValue value) {
    return guarded([&] {
        require(request, "request");
        if (request->ran) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the request has run; make a new one for the next");
        check(*request, option, type, value);
        if (request->model->file->spec(option)) request->values[option] = std::move(value);
        return SPEECH_OK;
    });
}

/** A seed from 0 to 2^53 - 1. */
int64_t draw_seed() {
    std::random_device device;
    return (int64_t) (((uint64_t) device() << 32 | device()) & (uint64_t) kMaxSeed);
}

/** Joins a request that runs to its caller's callbacks and cancellation. */
class RequestRun : public Run {
public:
    RequestRun(speech_request & request, speech_audio_callback on_audio, void * user_data)
        : request_(request), on_audio_(on_audio), user_data_(user_data) {}

    bool audio(const float * samples, size_t n) override {
        if (stopped()) return false;
        // Every family's audio passes here, so no synthesis returns audio that is not finite as speech.
        for (size_t i = 0; i < n; i++) {
            if (!std::isfinite(samples[i])) {
                throw Error(Fault::OutOfRange, "the speech came out as numbers that are not finite, which the model's float32 cannot hold: the "
                                               "request's scales or settings are larger than the model computes with; give smaller ones");
            }
        }
        passed_ = true;
        samples_ += n;
        if (on_audio_(samples, n, user_data_) != 0) stop_ = true;
        return !stopped();
    }

    bool progress(double done) override {
        if (stopped()) return false;
        passed_ = true;
        if (request_.on_progress && request_.on_progress(done, request_.progress_data) != 0) stop_ = true;
        return !stopped();
    }

    bool stopped() override { return stop_ || request_.cancelled.load(); }

    uint64_t samples() const { return samples_; }
    /** Whether the run has passed audio or progress to the caller, which only work does. */
    bool passed() const { return passed_; }

private:
    speech_request & request_;
    speech_audio_callback on_audio_;
    void * user_data_;
    bool stop_ = false, passed_ = false;
    uint64_t samples_ = 0;
};

/**
 * Runs the work of a request and spends the request, unless the family's checks of the whole request refused it
 * before the work passed anything to the caller: a length that the options give together, or a text too long. Such a
 * request is left as it was, to be fixed and run again.
 */
template <typename Body>
void spend(speech_request & request, const RequestRun & run, Body && body) {
    try {
        body();
    } catch (const Error & e) {
        const bool refused = e.fault() == Fault::InvalidArgument || e.fault() == Fault::OutOfRange;
        request.ran = !refused || run.passed();
        throw;
    } catch (...) {
        request.ran = true;
        throw;
    }
    request.ran = true;
}

/** Checks that a request can run its one run of `task`, and returns its options. */
RequestValues take(speech_request * request, speech_task task) {
    require(request, "request");
    const FileInfo & file = *request->model->file;
    if (file.family->task != task) {
        throw ApiError(SPEECH_ERROR_UNSUPPORTED,
                       file.identity.name + " is a model of speech " + task_name(file.family->task) + "; use " + call_of(file.family->task));
    }
    if (request->ran) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the request has run; make a new one for the next");
    if (task == SPEECH_TASK_SYNTHESIS && !request->text) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the request has no text; set it with speech_request_set_text()", "text");
    }
    if (task != SPEECH_TASK_SYNTHESIS && request->audio.empty()) {
        throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the request has no audio; set it with speech_request_set_audio()", "audio");
    }
    return run_values(*request);
}

/**
 * Runs a request of `task` on its audio, resampled to the model's rate, with `work`, which fills the result from the
 * samples unless the run stops; a request that stops has no more in its result than its stop reason.
 */
template <typename Work>
speech_status hear(speech_request * request, speech_task task, Work && work) {
    worker_trace::OnExit exit{"hear_exit"};
    worker_trace::mark("hear_enter");
    const RequestValues values = take(request, task);
    auto result = std::make_unique<speech_result>();
    result->task = task;
    RequestRun run(*request, [](const float *, size_t, void *) { return 0; }, nullptr);
    std::lock_guard<std::mutex> lock(request->model->busy);
    spend(*request, run, [&] {
        worker_trace::OnExit cleanup{"resample_scope_exit"};
        if (run.stopped()) return;
        worker_trace::mark("resample_before");
        const std::vector<float> samples = Resampler(request->sample_rate, request->model->file->sample_rate)(request->audio);
        worker_trace::mark("resample_after");
        work(samples, values, run, *result);
        worker_trace::mark("hear_work_end");
    });
    if (run.stopped()) result->stop = SPEECH_STOP_CANCELLED;
    const speech_status status = run.stopped() ? SPEECH_CANCELLED : SPEECH_OK;
    request->result = std::move(result);
    return status;
}

/** A token or a segment of a result, whose list is `list`; `what` names it in a message. */
speech_status timed_text(const speech_result * result, const std::vector<TimedText> * list, size_t index, double * start, double * end,
                                const char ** text, const char * what) {
    return guarded([&] {
        require(result, "result");
        if (index >= list->size()) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "there is no " + std::string(what) + " " + std::to_string(index) + "; the result has " +
                                                              std::to_string(list->size()));
        }
        const TimedText & t = (*list)[index];
        if (start) *start = t.start;
        if (end) *end = t.end;
        if (text) *text = t.text.c_str();
        return SPEECH_OK;
    });
}

}  // namespace

RequestValues run_values(const speech_request & request) {
    const FileInfo & file = *request.model->file;
    RequestValues values;
    for (const OptionSpec & spec : file.described.options) {
        const auto set = request.values.find(spec.option);
        if (set != request.values.end()) {
            values.set(spec.option, set->second, true);
        } else if (spec.required) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, file.identity.name + " needs the option " + speech_option_name(spec.option) + "; set it",
                           speech_option_name(spec.option));
        } else if (spec.default_value) {
            values.set(spec.option, *spec.default_value, false);
        } else if (spec.option == SPEECH_OPT_SEED) {
            values.set(spec.option, draw_seed(), false);
        }
    }
    return values;
}

extern "C" {

speech_status speech_request_new(speech_model * model, speech_request ** request) {
    if (request) *request = nullptr;
    return guarded([&] {
        require(model, "model");
        require(request, "request");
        auto r = std::make_unique<speech_request>();
        r->model = model;
        *request = r.release();
        return SPEECH_OK;
    });
}

void speech_request_free(speech_request * request) {
    delete request;
}

speech_status speech_request_set_text(speech_request * request, const char * text) {
    return guarded([&] {
        require(request, "request");
        const FileInfo & file = *request->model->file;
        if (file.family->task != SPEECH_TASK_SYNTHESIS) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED,
                           file.identity.name + " is a model of speech " + task_name(file.family->task) + " and takes no text; give it audio", "text");
        }
        require(text, "text", "text");
        if (!*text) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the text is empty; give a text to speak", "text");
        if (request->ran) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the request has run; make a new one for the next");
        request->text = text;
        return SPEECH_OK;
    });
}

speech_status speech_request_set_audio(speech_request * request, const float * samples, size_t n_samples, int sample_rate) {
    return guarded([&] {
        require(request, "request");
        const FileInfo & file = *request->model->file;
        if (file.family->task == SPEECH_TASK_SYNTHESIS) {
            throw ApiError(SPEECH_ERROR_UNSUPPORTED, file.identity.name + " speaks text and takes no audio; give it a text", "audio");
        }
        if (n_samples == 0) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the audio has no samples; give at least one", "audio");
        require(samples, "samples", "audio");
        if (request->ran) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "the request has run; make a new one for the next");
        try {
            Resampler(sample_rate, file.sample_rate ? file.sample_rate : sample_rate);
        } catch (const std::invalid_argument & e) {
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, e.what(), "audio");
        }
        request->audio.assign(samples, samples + n_samples);
        request->sample_rate = sample_rate;
        return SPEECH_OK;
    });
}

speech_status speech_request_set_string(speech_request * request, speech_option option, const char * value) {
    if (!value) {
        return guarded([&]() -> speech_status {
            const char * name = speech_option_name(option);
            throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, std::string("the value of ") + (name ? name : "the option") + " is NULL", name);
        });
    }
    return set_value(request, option, SPEECH_TYPE_STRING, std::string(value));
}

speech_status speech_request_set_int(speech_request * request, speech_option option, int64_t value) {
    return set_value(request, option, SPEECH_TYPE_INT, value);
}

speech_status speech_request_set_float(speech_request * request, speech_option option, double value) {
    return set_value(request, option, SPEECH_TYPE_FLOAT, value);
}

speech_status speech_request_set_bool(speech_request * request, speech_option option, int value) {
    return set_value(request, option, SPEECH_TYPE_BOOL, value != 0);
}

speech_status speech_request_set_progress(speech_request * request, speech_progress_callback on_progress, void * user_data) {
    return guarded([&] {
        require(request, "request");
        request->on_progress = on_progress;
        request->progress_data = user_data;
        return SPEECH_OK;
    });
}

void speech_request_cancel(speech_request * request) {
    if (request) request->cancelled = true;
}

speech_status speech_synthesize(speech_request * request, speech_audio_callback on_audio, void * user_data) {
    return guarded([&] {
        if (!on_audio) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "on_audio is NULL; give a callback that takes the audio");
        const RequestValues values = take(request, SPEECH_TASK_SYNTHESIS);
        auto result = std::make_unique<speech_result>();
        result->seed = values.has(SPEECH_OPT_SEED) ? values.integer(SPEECH_OPT_SEED) : -1;
        RequestRun run(*request, on_audio, user_data);
        std::lock_guard<std::mutex> lock(request->model->busy);
        spend(*request, run, [&] {
            if (!run.stopped()) result->stop = request->model->engine->speak(*request->text, values, run);
        });
        if (run.stopped()) result->stop = SPEECH_STOP_CANCELLED;
        result->samples = run.samples();
        const speech_status status = result->stop == SPEECH_STOP_CANCELLED ? SPEECH_CANCELLED : SPEECH_OK;
        request->result = std::move(result);
        return status;
    });
}

speech_status speech_transcribe(speech_request * request) {
    return guarded([&] {
        return hear(request, SPEECH_TASK_RECOGNITION, [&](const std::vector<float> & samples, const RequestValues & values, Run & run, speech_result & result) {
            Recognized found = request->model->engine->transcribe(samples, values, run);
            if (run.stopped()) return;
            result.text = std::move(found.text);
            result.segments = std::move(found.segments);
            result.tokens = std::move(found.tokens);
            result.stop = found.stop;
            result.languages = std::move(found.languages);
            result.speakers = std::move(found.speakers);
            result.untimed = std::move(found.untimed);
        });
    });
}

speech_status speech_detect(speech_request * request) {
    return guarded([&] {
        return hear(request, SPEECH_TASK_DETECTION, [&](const std::vector<float> & samples, const RequestValues & values, Run & run, speech_result & result) {
            std::vector<TimedText> regions = request->model->engine->detect(samples, values, run);
            if (!run.stopped()) result.segments = std::move(regions);
        });
    });
}

speech_status speech_diarize(speech_request * request) {
    return guarded([&] {
        return hear(request, SPEECH_TASK_DIARIZATION, [&](const std::vector<float> & samples, const RequestValues & values, Run & run, speech_result & result) {
            const std::vector<SpeakerSegment> segments = request->model->engine->diarize(samples, values, run);
            if (run.stopped()) return;
            for (const SpeakerSegment & s : segments) {
                result.segments.push_back({s.start, s.end, ""});
                result.speakers.push_back(s.speaker);
            }
        });
    });
}

speech_status speech_enhance(speech_request * request, speech_audio_callback on_audio, void * user_data) {
    return guarded([&] {
        if (!on_audio) throw ApiError(SPEECH_ERROR_INVALID_ARGUMENT, "on_audio is NULL; give a callback that takes the audio");
        const RequestValues values = take(request, SPEECH_TASK_ENHANCEMENT);
        auto result = std::make_unique<speech_result>();
        result->task = SPEECH_TASK_ENHANCEMENT;
        RequestRun run(*request, on_audio, user_data);
        std::lock_guard<std::mutex> lock(request->model->busy);
        spend(*request, run, [&] {
            if (!run.stopped()) request->model->engine->enhance(request->audio, request->sample_rate, values, run);
        });
        if (run.stopped()) result->stop = SPEECH_STOP_CANCELLED;
        result->samples = run.samples();
        const speech_status status = result->stop == SPEECH_STOP_CANCELLED ? SPEECH_CANCELLED : SPEECH_OK;
        request->result = std::move(result);
        return status;
    });
}

const speech_result * speech_request_result(const speech_request * request) {
    return request ? request->result.get() : nullptr;
}

speech_stop speech_result_stop(const speech_result * result) {
    return result ? result->stop : SPEECH_STOP_COMPLETE;
}

int64_t speech_result_seed(const speech_result * result) {
    return result ? result->seed : -1;
}

uint64_t speech_result_samples(const speech_result * result) {
    return result ? result->samples : 0;
}

const char * speech_result_text(const speech_result * result) {
    return result && result->task == SPEECH_TASK_RECOGNITION ? result->text.c_str() : nullptr;
}

size_t speech_result_segment_count(const speech_result * result) {
    return result ? result->segments.size() : 0;
}

speech_status speech_result_segment(const speech_result * result, size_t index, double * start, double * end, const char ** text) {
    return timed_text(result, result ? &result->segments : nullptr, index, start, end, text, "segment");
}

int speech_result_segment_speaker(const speech_result * result, size_t index) {
    return result && index < result->speakers.size() ? result->speakers[index] : -1;
}

size_t speech_result_token_count(const speech_result * result) {
    return result ? result->tokens.size() : 0;
}

speech_status speech_result_token(const speech_result * result, size_t index, double * start, double * end, const char ** text) {
    return timed_text(result, result ? &result->tokens : nullptr, index, start, end, text, "token");
}

const char * speech_result_untimed(const speech_result * result) {
    return result && result->task == SPEECH_TASK_RECOGNITION && result->untimed ? result->untimed->c_str() : nullptr;
}

size_t speech_result_language_count(const speech_result * result) {
    return result ? result->languages.size() : 0;
}

const char * speech_result_language(const speech_result * result, size_t index) {
    return result && index < result->languages.size() ? result->languages[index].c_str() : nullptr;
}

}  // extern "C"
