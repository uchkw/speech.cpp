#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "httplib.h"

#include "commands.h"
#include "fetch.h"
#include "jobs.h"
#include "json-reader.h"
#include "json.h"
#include "log.h"
#include "openai-api.h"
#include "terminal.h"

#include "access.h"
#include "audio-enhancements.h"
#include "audio-speech.h"
#include "page.h"
#include "realtime.h"
#include "served-models.h"

// speech serve: serves a model of each task over HTTP with OpenAI's audio API, so that a program that speaks HTTP (a web
// app, Python with requests, curl) can use speech.cpp without starting the worker: the synthesis model answers
// POST /v1/audio/speech and the recognition model POST /v1/audio/transcriptions, which with chunking_strategy recognizes
// the regions where the detection model finds that someone speaks, and the enhancement model speech.cpp's own
// POST /v1/audio/enhancements (audio-enhancements.h); GET /v1/models gives the models with their
// information, and GET /health says the server is up, which it is once the models given are loaded. A WebSocket at
// /v1/realtime carries OpenAI's Realtime transcription session (realtime.h). A failure of
// the library becomes OpenAI's error object by its category alone (openai::library_error()). A model serves one request
// at a time in the order they arrive, and a client that goes away while it waits or while its request runs cancels it.
// On a loopback address the server also serves its page (page.h), which replaces the model of a task by one of the
// catalog.

using namespace server;
using openai::ApiError;
using openai::send_error;

namespace {

/** What a task's model serves. */
const char * use_of(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "/v1/audio/speech";
        case SPEECH_TASK_RECOGNITION: return "/v1/audio/transcriptions";
        case SPEECH_TASK_DETECTION: return "chunking_strategy of /v1/audio/transcriptions and server_vad of /v1/realtime";
        case SPEECH_TASK_DIARIZATION: return "diarized_json of /v1/audio/transcriptions and diarization of /v1/realtime";
        case SPEECH_TASK_ENHANCEMENT: return "/v1/audio/enhancements";
    }
    throw std::logic_error("a task speech.h does not have");
}

/** The tasks of the endpoints that take a model, the synthesis and recognition models first, which a 404 names besides the one it lacks. */
constexpr speech_task kEndpointTasks[] = {SPEECH_TASK_SYNTHESIS, SPEECH_TASK_RECOGNITION, SPEECH_TASK_ENHANCEMENT};

/** OpenAI's model object, with the release of speech.cpp and the model's information. */
std::string model_json(const Served & served) {
    const auto info = served.info();
    return "{\"id\":" + json_string(speech_model_info_name(info.get())) + ",\"object\":\"model\",\"created\":" +
           std::to_string(served.created()) + ",\"owned_by\":\"speech.cpp\",\"version\":" + json_string(speech_version()) +
           ",\"defaults\":" + options_json(served.defaults()) + ",\"speech\":" + speech_model_info_json(info.get()) + "}";
}

/** The message of an error the server answered with, for the request's log line, or "". */
std::string error_message(const httplib::Response & res) {
    if (res.status < 400 || res.get_header_value("Content-Type") != "application/json") return "";
    const JsonValue body = parse_json(res.body);
    const JsonValue * error = body.member("error");
    const JsonValue * message = error ? error->member("message") : nullptr;
    return message ? message->text : "";
}

class Server {
public:
    Server(ServedModels & models, const Access & access, const Realtime & realtime) : models_(models), access_(access), realtime_(realtime) {}

    void route(httplib::Server & http) {
        http.set_pre_routing_handler([this](const httplib::Request & req, httplib::Response & res) { return admit(req, res); });
        http.set_error_handler([](const httplib::Request & req, httplib::Response & res) {
            if (res.body.empty()) {
                send_error(res, {res.status, res.status == 404 ? "Invalid URL (" + req.method + " " + req.path + ")."
                                                               : std::string(httplib::status_message(res.status)) + ".", "", ""});
            }
        });
        http.set_logger([](const httplib::Request & req, const httplib::Response & res) {
            const auto [what, skip] = take_request_note();
            if (!skip) log_request(req.method, req.path, std::to_string(res.status), what.empty() ? error_message(res) : what, req.remote_addr);
        });
        http.Get("/health", [](const httplib::Request &, httplib::Response & res) { res.set_content("{\"status\":\"ok\"}", "application/json"); });
        http.Get("/v1/models", [this](const httplib::Request &, httplib::Response & res) {
            std::string data;
            for (const auto & served : models_.all()) data += (data.empty() ? "" : ",") + model_json(*served);
            res.set_content("{\"object\":\"list\",\"data\":[" + data + "]}", "application/json");
        });
        http.Get("/v1/models/(.+)", [this](const httplib::Request & req, httplib::Response & res) {
            for (const auto & served : models_.all()) {
                if (req.matches[1] == speech_model_info_name(served->info().get())) {
                    res.set_content(model_json(*served), "application/json");
                    return;
                }
            }
            send_error(res, {404, "The model " + json_string(req.matches[1]) + " does not exist.", "model", "model_not_found"});
        });
        http.Post("/v1/audio/speech", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_SYNTHESIS, req, res)) answer_speech(std::move(served), req, res);
        });
        http.Post("/v1/audio/transcriptions", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_RECOGNITION, req, res)) transcription(std::move(served), req, res);
        });
        http.Post("/v1/audio/enhancements", [this](const httplib::Request & req, httplib::Response & res) {
            if (auto served = held(SPEECH_TASK_ENHANCEMENT, req, res)) answer_enhancement(std::move(served), req, res);
        });
    }

private:
    ServedModels & models_;
    const Access & access_;
    const Realtime & realtime_;

    /**
     * The model of the task the endpoint asks for; otherwise answers a 404, as for a path the server does not have,
     * or a 503 while the page loads one in its place.
     */
    std::shared_ptr<Served> held(speech_task wanted, const httplib::Request & req, httplib::Response & res) const {
        if (auto served = models_.of(wanted)) return served;
        const std::string task = std::string("speech ") + task_name(wanted);
        if (const std::string coming = models_.replacing(wanted); !coming.empty()) {
            res.set_header("Retry-After", "5");
            send_error(res, {503, "The " + task + " model is being loaded (" + coming + "); send the request again once it is in place.", "",
                             "model_loading"});
            return nullptr;
        }
        std::string message = "This server holds no " + task + " model, which " + req.path + " is for";
        std::string holds;
        for (const speech_task other : kEndpointTasks) {
            const auto served = other == wanted ? nullptr : models_.of(other);
            if (!served) continue;
            holds += std::string(holds.empty() ? "; it holds " : " and ") + speech_model_info_name(served->info().get()) + ", a speech " +
                     task_name(other) + " model, which " + use_of(other) + " is for";
        }
        message += holds;
        send_error(res, {404, message + ". Give speech serve a " + task + " model" + (access_.loopback() ? ", or pick one on its page." : "."), "", ""});
        return nullptr;
    }

    /**
     * Refuses a request from an origin that may not call the server, adds the CORS headers for an origin --cors-origin
     * allows, and answers a preflight.
     */
    httplib::Server::HandlerResponse admit(const httplib::Request & req, httplib::Response & res) const {
        // A note left by a request whose logger never ran, as when its connection broke, must not reach the next one.
        take_request_note();
        if (const auto refusal = access_.origin_refusal(req)) {
            send_error(res, *refusal);
            return httplib::Server::HandlerResponse::Handled;
        }
        // A WebSocket is answered by its handler only once it is upgraded, so a refusal of one is an HTTP answer here.
        if (req.path == "/v1/realtime") {
            if (const auto refusal = realtime_.refusal(req)) {
                send_error(res, *refusal);
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        const std::string origin = req.get_header_value("Origin");
        const bool allowed = access_.cross_origin_allowed(origin);
        if (allowed) {
            res.set_header("Access-Control-Allow-Origin", access_.any_origin() ? "*" : origin);
            res.set_header("Access-Control-Expose-Headers", "X-Sample-Rate, X-Speech-Seed, X-Speech-Stop");
            if (!access_.any_origin()) res.set_header("Vary", "Origin");
        }
        if (req.method != "OPTIONS") return httplib::Server::HandlerResponse::Unhandled;
        if (allowed) {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            const std::string headers = req.get_header_value("Access-Control-Request-Headers");
            res.set_header("Access-Control-Allow-Headers", headers.empty() ? "Content-Type, Authorization" : headers);
            res.set_header("Access-Control-Max-Age", "86400");
        }
        res.status = 204;
        return httplib::Server::HandlerResponse::Handled;
    }

    /**
     * Recognizes the form's WAV file, or with chunking_strategy each region of it where the detection model finds that
     * someone speaks, and answers with its text once it is done, as json, text, verbose_json, which also carries the
     * language the model heard, or diarized_json, whose regions carry the speakers the diarization model finds in the
     * whole file, and with why the recognition ended in X-Speech-Stop.
     */
    void transcription(std::shared_ptr<Served> served, const httplib::Request & req, httplib::Response & res) {
        const auto info = served->info();
        openai::TranscriptionRequest asked;
        try {
            std::vector<openai::FormPart> parts;
            for (const auto & [field_name, field] : req.form.fields) parts.push_back({field_name, field.content, "", false});
            for (const auto & [file_name, file] : req.form.files) parts.push_back({file_name, file.content, file.filename, true});
            asked = openai::read_transcription_request(req.is_multipart_form_data(), parts, speech_model_info_name(info.get()),
                                                       speech_model_info_takes(info.get(), SPEECH_OPT_TIMESTAMPS),
                                                       speech_model_info_takes(info.get(), SPEECH_OPT_SPEAKERS), served->defaults());
        } catch (const ApiError & e) {
            send_error(res, e);
            return;
        }
        asked.options = served->with_defaults(asked.options);
        auto job = std::make_shared<TranscriptionJob>();
        job->sample_rate = asked.sample_rate;
        job->duration = (double) asked.samples.size() / asked.sample_rate;
        job->served = served;
        job->timestamps = asked.timestamps;
        job->own_speakers = asked.speakers;
        job->method = req.method;
        job->path = req.path;
        job->client = req.remote_addr;
        const std::string recognizer = speech_model_info_name(info.get());
        if (asked.regions) {
            job->detector = needed(SPEECH_TASK_DETECTION, "chunking_strategy",
                                   {400, "chunking_strategy recognizes the regions where a speech detection model finds that someone speaks, and this "
                                         "server holds none. Give speech serve one beside the recognition model, as in speech serve " + recognizer +
                                         " silero-vad, or leave chunking_strategy out.",
                                    "chunking_strategy", "unsupported_parameter"},
                                   res);
            if (!job->detector) return;
            try {
                job->detection = detection_request(job->detector->get(), asked.samples, asked.sample_rate,
                                                   region_options(*asked.regions, job->detector->defaults()));
            } catch (const Failure & e) {
                ApiError error = openai::library_error(e);
                const std::string member = server_vad_member(e.option());
                if (!member.empty()) error.param = "chunking_strategy[" + member + "]";
                send_error(res, error);
                return;
            }
        }
        if (asked.enhance) {
            job->enhancer = needed(SPEECH_TASK_ENHANCEMENT, "noise_reduction",
                                   {400, "noise_reduction enhances the file with a speech enhancement model, and this server holds none. Give speech "
                                         "serve one beside the recognition model, as in speech serve " + recognizer + " re-use, or leave noise_reduction "
                                         "out.",
                                    "noise_reduction", "unsupported_value"},
                                   res);
            if (!job->enhancer) return;
            try {
                job->enhancement = enhancement_request(job->enhancer->get(), asked.samples, asked.sample_rate, job->enhancer->defaults());
                job->enhanced_rate = enhanced_rate(job->enhancer->info().get(), asked.sample_rate, job->enhancer->defaults());
            } catch (const Failure & e) {
                send_error(res, openai::library_error(e));
                return;
            }
        }
        if (asked.format == "diarized_json" && !asked.speakers) {
            job->diarizer = needed(SPEECH_TASK_DIARIZATION, "diarized_json",
                                   {400, "\"response_format\": \"diarized_json\" gives each region the speakers that a speech diarization model "
                                         "finds, and this server holds none. Give speech serve one beside the other models, as in speech serve " +
                                         recognizer + " silero-vad nemotron-3-diarization, or ask for another response_format.",
                                    "response_format", "unsupported_value"},
                                   res);
            if (!job->diarizer) return;
            try {
                job->diarization = diarization_request(job->diarizer->get(), asked.samples, asked.sample_rate, job->diarizer->defaults());
            } catch (const Failure & e) {
                send_error(res, openai::library_error(e));
                return;
            }
        }
        try {
            job->request = new_request(served->get());
            if (!asked.regions) check(speech_request_set_audio(job->request.get(), asked.samples.data(), asked.samples.size(), asked.sample_rate));
            apply_options(job->request.get(), asked.options);
        } catch (const Failure & e) {
            send_error(res, openai::transcription_error(e));
            return;
        }
        if (asked.regions) {
            job->request.reset();
            job->recognizer = served->get();
            job->samples = std::move(asked.samples);
            job->options = std::move(asked.options);
        }
        const uint64_t ticket = served->turns().take();
        std::thread(transcribe, job, std::ref(served->turns()), ticket).detach();
        const LogLine log_line(*job);
        std::unique_lock<std::mutex> lock(job->mutex);
        if (!wait_for(*job, lock, [] { return false; }, [&] { return req.is_connection_closed(); })) return;
        if (job->failure) {
            send_error(res, openai::transcription_error(*job->failure));
            return;
        }
        if (job->status != SPEECH_OK) return;
        res.set_header("X-Speech-Stop", speech_stop_name(job->transcript.stop));
        if (asked.format == "text") {
            res.set_content(job->transcript.text, "text/plain; charset=utf-8");
        } else if (asked.format == "verbose_json") {
            res.set_content(openai::transcription_verbose_json(job->transcript, job->duration, asked.timestamps, asked.segments, asked.words),
                            "application/json");
        } else if (asked.format == "diarized_json" && asked.speakers) {
            res.set_content(openai::transcription_diarized_json(job->transcript, job->duration), "application/json");
        } else if (asked.format == "diarized_json") {
            res.set_content(openai::transcription_diarized_json(job->transcript, job->duration, job->regions, job->parts, job->speakers),
                            "application/json");
        } else {
            res.set_content(openai::transcription_json(job->transcript.text), "application/json");
        }
    }

    /**
     * The model of `task` that a request's `need` needs; otherwise answers `none`, a 400 that says to give speech serve
     * one, or a 503 while the page loads one.
     */
    std::shared_ptr<Served> needed(speech_task task, const std::string & need, const ApiError & none, httplib::Response & res) const {
        if (auto served = models_.of(task)) return served;
        if (const std::string coming = models_.replacing(task); !coming.empty()) {
            res.set_header("Retry-After", "5");
            send_error(res, {503, "The speech " + std::string(task_name(task)) + " model, which " + need + " needs, is being loaded (" + coming +
                                      "); send the request again once it is in place.", "", "model_loading"});
            return nullptr;
        }
        send_error(res, none);
        return nullptr;
    }
};

/** A model's row in the list the server starts with: its name, then its task, its device and its load time in gray. */
std::string model_row(const Served & served) {
    const auto info = served.info();
    char rest[64];
    std::snprintf(rest, sizeof rest, "%-11s  %-5s  %.2f s", task_name(speech_model_info_task(info.get())), speech_model_info_device(info.get()),
                  served.load_seconds());
    std::string name = speech_model_info_name(info.get());
    if (name.size() < 32) name += std::string(32 - name.size(), ' ');
    return name + "  " + terminal::paint(terminal::Color::Gray, rest);
}

/** The flag of an option given for one task's model: "--asr.language". */
std::string task_flag(speech_task task, speech_option option) {
    return std::string("--") + task_prefix(task) + "." + option_flag(option).substr(2);
}

/**
 * Gives each model the command line named its defaults: the request options given without a task's prefix that it
 * takes, and those given with its task's prefix, which win over them. An option that no model given takes, or one given
 * for a task without a model, stops the server. The models are held only while this runs: a model the server keeps a
 * reference to past it could not be let go when the page replaces it, and the replacement would wait for it forever.
 */
void give_defaults(ServedModels & models, const CommandLine & line) {
    const auto given = models.all();
    for (const RequestOption & o : line.options) {
        const bool taken = std::any_of(given.begin(), given.end(), [&](const std::shared_ptr<Served> & s) {
            return speech_model_info_takes(s->info().get(), o.option) != 0;
        });
        if (!taken) {
            throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), speech_option_name(o.option),
                          option_flag(o.option) + " sets " + speech_option_name(o.option) +
                              " by default for the models given, and none of them takes it; `speech info` lists the options a model takes");
        }
    }
    for (const TaskOption & t : line.task_options) {
        const auto served = models.of(t.task);
        const std::string flag = task_flag(t.task, t.option.option);
        if (!served) throw UsageError(flag + " sets a default for the model that " + use_of(t.task) + " uses, and none is given");
        if (!speech_model_info_takes(served->info().get(), t.option.option)) {
            throw Failure(speech_status_name(SPEECH_ERROR_UNSUPPORTED), speech_option_name(t.option.option),
                          std::string(speech_model_info_name(served->info().get())) + " does not take " + speech_option_name(t.option.option) +
                              ", which " + flag + " sets; `speech info` lists the options a model takes");
        }
    }
    for (const auto & served : given) {
        const speech_task task = speech_model_info_task(served->info().get());
        std::vector<RequestOption> own;
        for (const TaskOption & t : line.task_options) {
            if (t.task == task) own.push_back(t.option);
        }
        try {
            served->set_defaults(defaults_for(served->get(), with_defaults(line.options, own)));
        } catch (const Failure & e) {
            // A value given for every model that one of them refuses is given to the others with their task's prefix.
            const auto refused = std::find_if(line.options.begin(), line.options.end(),
                                              [&](const RequestOption & o) { return e.option() == speech_option_name(o.option); });
            if (refused == line.options.end()) throw;
            std::string prefixed;
            for (const auto & s : given) {
                if (!speech_model_info_takes(s->info().get(), refused->option)) continue;
                prefixed += (prefixed.empty() ? "" : " or ") + task_flag(speech_model_info_task(s->info().get()), refused->option);
            }
            throw Failure(e.code(), e.option(), std::string(e.what()) + "; " + option_flag(refused->option) +
                                                    " sets it for every model given that takes it, and " + prefixed + " for one alone");
        }
    }
}

int run_serve(const CommandLine & line, FILE *) {
    const std::string host = line.value("--host").value_or("127.0.0.1");
    const int port = line.integer("--port").value_or(8080);
    if (port < 0 || port > 65535) throw UsageError("--port takes a port from 0 to 65535, not " + std::to_string(port));
    const bool open = line.has("--open");
    if (open && host != "127.0.0.1" && host != "::1" && host != "localhost") {
        throw UsageError("--open opens the page, which the server has only on 127.0.0.1, ::1 or localhost, not on " + host +
                         "; leave --host out");
    }
    if (line.args.empty() && !open) {
        throw UsageError(no_model_message("serve MODEL [options]", ModelKind::Any) + "\nOr give --open, and pick the models on the page.");
    }
    const Loading loading = line.loading(true);
    ServedModels models(loading);
    const std::string logo = terminal::logo();
    std::fprintf(stderr, "%s%s %s\n\n", logo.empty() ? "" : (logo + " ").c_str(), terminal::paint(terminal::Color::Bold, "speech.cpp").c_str(),
                 terminal::paint(terminal::Color::Gray, std::string(speech_version()) + " \xC2\xB7 serve").c_str());
    for (const std::string & argument : line.args) {
        // A model named for the first time is fetched here, with its own progress, before its load is shown.
        const std::string path = model_file(argument);
        terminal::Status status("Loading " + argument_label(argument), 2);
        const auto served = models.load_given(path, is_model_path(argument) ? "" : choice_name(find_model(argument)));
        status.done(model_row(*served));
    }
    if (line.args.empty()) std::fprintf(stderr, "  %s\n", terminal::paint(terminal::Color::Gray, "No model yet: pick one of each task on the page").c_str());
    if (!loading.voices.empty()) {
        const auto synthesis = models.of(SPEECH_TASK_SYNTHESIS);
        if (!synthesis) throw UsageError("--add-voice adds voices to the synthesis model, and none is given");
        for (const auto & [name, file] : loading.voices) synthesis->add_voice(name, file);
    }
    give_defaults(models, line);
    for (const auto & served : models.all()) {
        if (served->defaults().empty()) continue;
        std::fprintf(stderr, "      %s\n", terminal::paint(terminal::Color::Gray, std::string(speech_model_info_name(served->info().get())) + " by default " +
                                                                              options_text(served->defaults())).c_str());
    }

    httplib::Server http;
    // Nagle's algorithm would hold a small chunk of a stream until the client acknowledges the previous one.
    http.set_tcp_nodelay(true);
    // A file to recognize, or a recording to make a voice of, may take OpenAI's limit for an upload, 25 MB, and a
    // Realtime session may hold as much audio not yet transcribed, 546 s of its 16-bit PCM at 24000 Hz.
    constexpr size_t kUpload = 25 << 20;
    http.set_payload_max_length(kUpload);
    const int bound = port == 0 ? http.bind_to_any_port(host) : (http.bind_to_port(host, port) ? port : -1);
    if (bound < 0) {
        throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "cannot listen on " + host + ":" + std::to_string(port) + "; choose another --port or --host");
    }
    const Access access(host, bound, line.values("--cors-origin"));
    log_clients(!access.loopback());
    Realtime realtime(models, access, kUpload);
    Server server(models, access, realtime);
    server.route(http);
    realtime.route(http);
    Page page(models, access);
    page.route(http);
    const std::string address = host.find(':') == std::string::npos ? host : "[" + host + "]";
    const std::string api = "http://" + address + ":" + std::to_string(bound) + "/v1";
    const std::string stop = "Ctrl-C stops the server";
    std::fprintf(stderr, "\n  %s\n",
                 terminal::marked(terminal::Mark::Audio, "Ready at " + terminal::paint(terminal::Color::Sky, access.loopback() ? access.page_url() : api)).c_str());
    std::fprintf(stderr, "    %s\n\n", terminal::paint(terminal::Color::Gray, access.loopback() ? "API at " + api + "; " + stop : stop).c_str());
    if (open) open_in_browser(access.page_url());
    if (!http.listen_after_bind()) throw Failure(speech_status_name(SPEECH_ERROR_IO), "", "the server stopped listening");
    return 0;
}

}  // namespace

Command serve_command() {
    Command c;
    c.name = "serve";
    c.usage = "serve [MODEL...] [options]";
    c.summary = "serve models over HTTP with OpenAI's audio API, and a page to try them";
    c.description =
        "Loads each MODEL, one model of each task at most (synthesis, recognition, detection, diarization and\n"
        "enhancement), warmed up unless --no-warmup, adds the voices of --add-voice to the synthesis model, and serves\n"
        "them over HTTP: POST /v1/audio/speech, POST /v1/audio/transcriptions, which with chunking_strategy recognizes the\n"
        "regions where the detection model finds speech and with response_format diarized_json gives each its speakers,\n"
        "POST /v1/audio/enhancements, speech.cpp's own, which enhances a WAV file, GET /v1/models,\n"
        "GET /health, and OpenAI's Realtime transcription over a WebSocket at /v1/realtime, which transcribes each buffer\n"
        "the client commits, or with server_vad each region of the detection model, and the text agreed so far while it\n"
        "is said, and with session.audio.input.diarization gives each utterance its speakers. It has no authentication\n"
        "and no TLS, and refuses a request that a web page at another origin than --cors-origin's sends. On 127.0.0.1,\n"
        "::1 or localhost it also serves a page on which to pick models of the catalog, fetch them and try them, at the\n"
        "address it prints; --open opens it, and starts the server without a model if none is given. A request option\n"
        "given as a flag is the default, in place of the model's own, of every request that does not set it on each\n"
        "MODEL that takes it; a model the page loads keeps its own defaults.";
    c.flags = {
        {"--host", "ADDRESS", false, "the address to listen on, 127.0.0.1 unless given"},
        {"--port", "N", false, "the port, 8080 unless given; 0 for any free one"},
        {"--open", "", false, "open the page in the browser once the server listens"},
        {"--cors-origin", "ORIGIN|*", true, "an origin a web page may call the server from, or * for any"},
        add_voice_flag(),
        device_flag("auto (the first GPU, or the CPU without one), gpu, cpu or a name `speech devices` lists; a detection model "
                    "runs on the CPU"),
        threads_flag(),
        no_warmup_flag(),
    };
    c.request_options = true;
    c.task_options = true;
    c.model = ModelKind::Any;
    c.max_args = 5;
    c.run = run_serve;
    return c;
}
