#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "base64.h"
#include "cancellation.h"
#include "commands.h"
#include "enhancement.h"
#include "fetch.h"
#include "inbox.h"
#include "json.h"
#include "sentences.h"

// speech worker: serves one model over the worker protocol, JSON Lines on stdin and stdout, for another program
// that starts it, ASIST among them. Its stdout carries the protocol and nothing else. It loads the model, warmed up
// unless --no-warmup, adds the voices of --add-voice and says `ready` with the model's information, or `fatal` when it
// cannot; then it runs the requests one at a time in the order they become complete, but for info and count_tokens,
// which it answers as they arrive, each answered by exactly one terminal message, `end`, `error` or `cancelled`, and
// exits with 0 once stdin closes and every request is answered.

namespace {

/**
 * The number of the worker protocol, in `ready`. It rises when a caller must change to keep working: a message or member
 * removed, renamed or given another meaning.
 */
constexpr int kProtocol = 4;

using Clock = std::chrono::steady_clock;

/** The protocol of the running worker, which a failure that ends it reports `fatal` through. */
Protocol * current = nullptr;

/** Reads an opt-in worker experiment before the input thread starts. */
bool worker_experiment(const char * name) {
    const char * value = std::getenv(name);
    if (!value || std::strcmp(value, "0") == 0) return false;
    if (std::strcmp(value, "1") == 0) return true;
    throw Failure(speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), name, std::string(name) + " must be 0 or 1; unset it to disable");
}

/** The failure an exception is: the library's, or a defect, which the worker reports as internal. */
Failure failure_of(const std::exception & e) {
    if (const auto * f = dynamic_cast<const Failure *>(&e)) return *f;
    return Failure(speech_status_name(SPEECH_ERROR_INTERNAL), "", e.what());
}

std::string error_line(const std::string & id, const std::exception & e) {
    const Failure f = failure_of(e);
    return "{\"type\":\"error\",\"id\":" + json_string(id) + ",\"error\":" + error_object(f.code(), f.option(), f.what()) + "}";
}

std::string cancelled_line(const std::string & id) {
    return "{\"type\":\"cancelled\",\"id\":" + json_string(id) + "}";
}

/**
 * What the callbacks of a running request need: the id, the next chunk's number and the time of the request's last
 * message, which progress waits a second after.
 */
struct Running {
    Protocol & protocol;
    const std::string & id;
    Clock::time_point last;
    int seq = 0;
    std::vector<int16_t> pcm;
};

int on_audio(const float * s, size_t n, void * user_data) {
    Running & r = *static_cast<Running *>(user_data);
    r.pcm.resize(n);
    for (size_t i = 0; i < n; i++) r.pcm[i] = (int16_t) std::lround(std::max(-1.0f, std::min(1.0f, s[i])) * 32767.0f);
    r.protocol.line("{\"type\":\"chunk\",\"id\":" + json_string(r.id) + ",\"seq\":" + std::to_string(r.seq++) + ",\"pcm\":\"" +
                    base64((const uint8_t *) r.pcm.data(), n * sizeof(int16_t)) + "\"}");
    r.last = Clock::now();
    return 0;
}

int on_progress(double done, void * user_data) {
    Running & r = *static_cast<Running *>(user_data);
    const Clock::time_point now = Clock::now();
    if (now - r.last >= std::chrono::seconds(1)) {
        r.protocol.line("{\"type\":\"progress\",\"id\":" + json_string(r.id) + ",\"done\":" + json_number(done) + "}");
        r.last = now;
    }
    return 0;
}

/** The audio of transcribe and enhance as the library takes it: 16-bit samples scaled as a 16-bit WAVE file is read. */
std::vector<float> samples_of(const Job & job) {
    std::vector<float> out(job.pcm.size());
    for (size_t i = 0; i < out.size(); i++) out[i] = (float) job.pcm[i] / 32768.0f;
    return out;
}

/**
 * Speaks the job's text as one request of the library, as its caller chose it; on a model that speaks by sentence, a
 * text the library refuses as too long is cut and spoken in pieces, as speech tts speaks one.
 */
void synthesize(Inbox & inbox, Protocol & protocol, speech_model * model, const Job & job) {
    Cancellation cancellation;
    if (!inbox.start(job, &cancellation)) return;
    Running running{protocol, job.id, Clock::now()};
    std::string terminal;
    try {
        const std::optional<Spoken> spoken =
            speak_text(model, job.text, job.options, Split::Refused, on_audio, on_progress, &running, cancellation);
        if (!spoken) {
            terminal = cancelled_line(job.id);
        } else {
            terminal = "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"seed\":" + std::to_string(spoken->seed) +
                       ",\"samples\":" + std::to_string(spoken->samples) + ",\"stop\":\"" + speech_stop_name(spoken->stop) + "\"}";
        }
    } catch (const std::exception & e) {
        terminal = error_line(job.id, e);
    }
    inbox.finish(job, terminal);
}

void recognize(Inbox & inbox, Protocol & protocol, speech_model * model, const speech_model_info * info, const Job & job) {
    const Request request = new_request(model);
    speech_request * r = request.get();
    Cancellation cancellation;
    if (!inbox.start(job, &cancellation)) return;
    Running running{protocol, job.id, Clock::now()};
    std::string terminal;
    try {
        const std::vector<float> samples = samples_of(job);
        check(speech_request_set_audio(r, samples.data(), samples.size(), job.sample_rate));
        apply_options(r, job.options);
        check(speech_request_set_progress(r, on_progress, &running));
        if (check(cancellation.run(r, [&] { return speech_transcribe(r); })) == SPEECH_CANCELLED) {
            terminal = cancelled_line(job.id);
        } else {
            const bool timestamps = timestamps_in_effect(info, job.options), speakers = in_effect(info, job.options, SPEECH_OPT_SPEAKERS);
            terminal = "{\"type\":\"end\",\"id\":" + json_string(job.id) + recognition_members(speech_request_result(r), timestamps, speakers) + "}";
        }
    } catch (const std::exception & e) {
        terminal = error_line(job.id, e);
    }
    inbox.finish(job, terminal);
}

/** Enhances the job's audio as one request of the library, passing the enhanced audio in chunks as it comes. */
void enhance(Inbox & inbox, Protocol & protocol, speech_model * model, const Job & job) {
    Cancellation cancellation;
    if (!inbox.start(job, &cancellation)) return;
    Running running{protocol, job.id, Clock::now()};
    std::string terminal;
    try {
        const std::vector<float> samples = samples_of(job);
        const Request request = enhancement_request(model, samples, job.sample_rate, job.options);
        check(speech_request_set_progress(request.get(), on_progress, &running));
        const std::optional<uint64_t> passed = enhance_audio(request.get(), on_audio, &running, cancellation);
        if (!passed) {
            terminal = cancelled_line(job.id);
        } else {
            terminal = "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"samples\":" + std::to_string(*passed) + ",\"stop\":\"" +
                       speech_stop_name(speech_result_stop(speech_request_result(request.get()))) + "\"}";
        }
    } catch (const std::exception & e) {
        terminal = error_line(job.id, e);
    }
    inbox.finish(job, terminal);
}

/** Runs add_voice, which a cancel no longer stops once it runs. */
void add_voice(Inbox & inbox, speech_model * model, const Job & job) {
    if (!inbox.start(job, nullptr)) return;
    std::string terminal = "{\"type\":\"end\",\"id\":" + json_string(job.id) + "}";
    try {
        check(speech_voice_add(model, job.name.c_str(), job.path.c_str()));
    } catch (const std::exception & e) {
        terminal = error_line(job.id, e);
    }
    inbox.finish(job, terminal);
}

/**
 * The terminal message of an info, the defaults of the command line and the model's information with the voices added
 * so far, or of a count_tokens. Both read information, which any thread may read while a request runs, so the reader
 * answers them as they arrive.
 */
std::string answer_at_once(const speech_model * model, const speech_model_info * info, const std::string & defaults, const Job & job) {
    try {
        if (job.kind == Job::Kind::Info) {
            return "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"defaults\":" + defaults + ",\"model\":" +
                   speech_model_info_json(model_info(model).get()) + "}";
        }
        size_t tokens = 0;
        check(speech_model_info_text_tokens(info, job.text.c_str(), &tokens));
        return "{\"type\":\"end\",\"id\":" + json_string(job.id) + ",\"tokens\":" + std::to_string(tokens) + "}";
    } catch (const std::exception & e) {
        return error_line(job.id, e);
    }
}

int run_worker(const CommandLine & line, FILE * out) {
    // The reader thread reads stdin until it closes, which can outlive this function when a failure ends the worker,
    // so the protocol and the inbox it uses are never freed.
    Protocol & protocol = *new Protocol(out);
    current = &protocol;
    const bool unsync = worker_experiment("SPEECH_HAAS_WORKER_UNSYNC");
    const bool untie = worker_experiment("SPEECH_HAAS_WORKER_UNTIE");
    // The protocol uses its own FILE descriptor, and this command reads stdin only through std::cin.
    if (unsync) std::ios::sync_with_stdio(false);
    if (untie) std::cin.tie(nullptr);
    const std::string path = model_file(line.args[0]);
    const Loading loading = line.loading(true);
    // The protocol has the messages of synthesis, recognition and enhancement alone.
    const speech_task task = speech_model_info_task(file_info(path).get());
    if (task == SPEECH_TASK_DETECTION) {
        throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                      path + " is a model of speech detection, which the worker does not serve; use `speech vad` or the C API's speech_detect()");
    }
    if (task == SPEECH_TASK_DIARIZATION) {
        throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), "",
                      path + " is a model of speech diarization, which the worker does not serve; use `speech diarize` or the C API's speech_diarize()");
    }
    const Model model = load_model(path, loading);
    const ModelInfo info = model_info(model.get());
    const speech_model_info * m = info.get();
    for (const RequestOption & o : line.options) {
        if (!speech_model_info_takes(m, o.option)) {
            throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), speech_option_name(o.option),
                          std::string(speech_model_info_name(m)) + " does not take " + speech_option_name(o.option) + ", which " +
                              option_flag(o.option) + " sets; `speech info` lists the options a model takes");
        }
    }
    const std::vector<RequestOption> defaults = defaults_for(model.get(), line.options);
    const std::string defaults_json = options_json(defaults);
    protocol.line("{\"type\":\"ready\",\"protocol\":" + std::to_string(kProtocol) + ",\"version\":" + json_string(speech_version()) +
                  ",\"defaults\":" + defaults_json + ",\"model\":" + speech_model_info_json(m) + "}");
    std::fprintf(stderr, "speech worker: %s on %s, ready\n", speech_model_info_name(m), speech_model_info_device(m));
    if (!defaults.empty()) std::fprintf(stderr, "speech worker: by default %s\n", options_text(defaults).c_str());

    const speech_model * loaded = model.get();
    Inbox & inbox = *new Inbox(protocol, speech_model_info_task(m), speech_model_info_name(m),
                               [loaded, m, defaults_json](const Job & job) { return answer_at_once(loaded, m, defaults_json, job); });
    std::thread reader([&inbox] { inbox.read(std::cin); });
    reader.detach();
    for (Job job; inbox.take(job);) {
        job.options = with_defaults(defaults, job.options);
        switch (job.kind) {
            case Job::Kind::Synthesize: synthesize(inbox, protocol, model.get(), job); break;
            case Job::Kind::Transcribe: recognize(inbox, protocol, model.get(), m, job); break;
            case Job::Kind::Enhance: enhance(inbox, protocol, model.get(), job); break;
            default: add_voice(inbox, model.get(), job); break;
        }
    }
    inbox.close_collecting();
    return 0;
}

void report_fatal(const Failure & e, FILE * out) {
    const std::string line = "{\"type\":\"fatal\",\"error\":" + error_object(e.code(), e.option(), e.what()) + "}";
    if (current) {
        current->line(line);
    } else {
        Protocol(out).line(line);
    }
}

}  // namespace

Command worker_command() {
    Command c;
    c.name = "worker";
    c.usage = "worker MODEL [options]";
    c.summary = "serve a model over JSON Lines on stdin and stdout (protocol " + std::to_string(kProtocol) + ")";
    c.description =
        "Loads MODEL, warmed up unless --no-warmup, adds the voices of --add-voice, and serves it over the worker protocol\n" +
        std::to_string(kProtocol) + ": one JSON object per line on stdin and on stdout, and nothing else on stdout. speech.cpp's\n"
        "docs/worker.md gives the messages. A request option given as a flag is the default of every request that does not\n"
        "set it, in place of the model's own.";
    c.flags = {
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists"),
        threads_flag(),
        no_warmup_flag(),
    };
    c.request_options = true;
    c.model = ModelKind::Any;
    c.min_args = 1;
    c.max_args = 1;
    c.run = run_worker;
    c.report = report_fatal;
    return c;
}
