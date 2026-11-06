#pragma once

#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <istream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "cancellation.h"
#include "failure.h"
#include "request-options.h"
#include "speech.h"
#include "transport-trace.h"

// The worker's side of reading its protocol: each line on stdin checked as a message of its type, the requests in
// flight until their terminal message, the audio of a recognition or enhancement request gathered from its chunks,
// cancels, and the queue of what runs in turn. Every answer the reading gives, and every terminal message, goes out with the inbox
// locked, so that a request's state and what the caller has been told never differ.

/** Writes the worker's lines on its stdout, whole, one at a time. */
class Protocol {
public:
    explicit Protocol(FILE * out) : out_(out) {}
    void line(const std::string & json);

private:
    std::mutex mutex_;
    FILE * out_;
};

/** An error object of the protocol: {"code", "option", "message"}, the option null when `option` is empty. */
std::string error_object(const std::string & code, const std::string & option, const std::string & message);

/** What the worker runs in turn, or for info and count_tokens answers at once. */
struct Job {
    enum class Kind { Synthesize, Transcribe, Enhance, AddVoice, Info, CountTokens };
    Kind kind = Kind::Info;
    std::string id;
    /** The serial of the request the job belongs to, which tells it from a later request under the same id. */
    uint64_t serial = 0;
    /** The text of synthesize and count_tokens. */
    std::string text;
    /** The name and path of add_voice. */
    std::string name, path;
    std::vector<RequestOption> options;
    /** The audio of transcribe and enhance: 16-bit samples at `sample_rate`. */
    std::vector<int16_t> pcm;
    int sample_rate = 0;
};

class Inbox {
public:
    /**
     * `at_once` gives the terminal message of an info or a count_tokens, which only read the model's information and
     * so are answered as they arrive rather than in turn behind a running request.
     */
    Inbox(Protocol & protocol, speech_task task, std::string model_name, std::function<std::string(const Job &)> at_once, TransportTrace & trace)
        : protocol_(protocol), task_(task), model_name_(std::move(model_name)), at_once_(std::move(at_once)), trace_(trace) {}

    /** Reads stdin's lines until it closes, answering at once those it refuses, and queues the rest. */
    void read(std::istream & in);

    /** The next job, waiting for one; false once stdin has closed and every job has been taken. */
    bool take(Job & job);

    /**
     * Marks the job's request as running, `cancellation` being what a cancel stops, the library's requests the job runs
     * through it (nullptr for a job without one), or returns false for a job whose request was cancelled while it
     * waited, which has had its `cancelled`.
     */
    bool start(const Job & job, Cancellation * cancellation);

    /** Sends the request's terminal message and forgets the request. */
    void finish(const Job & job, const std::string & line);

    /** Answers each request still collecting chunks once stdin has closed, which no transcribe or enhance can follow. */
    void close_collecting();

private:
    /** A request that has not had its terminal message. */
    struct Pending {
        enum class State { Collecting, Waiting, Running };
        uint64_t serial = 0;
        State state = State::Waiting;
        uint64_t next_seq = 0;
        std::vector<int16_t> pcm;
        /** What a cancel stops while the request runs. */
        Cancellation * running = nullptr;
    };

    Protocol & protocol_;
    speech_task task_;
    std::string model_name_;
    std::function<std::string(const Job &)> at_once_;
    TransportTrace & trace_;
    /** The reader's current line, used only while that reader holds mutex_. */
    TransportTrace::Line * tracing_line_ = nullptr;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::deque<Job> queue_;
    std::map<std::string, Pending> pending_;
    /**
     * The ids of requests answered while they collected chunks, by an error or a cancel: their later chunk lines are
     * dropped up to and including their transcribe or enhance line, which frees the id, or a chunk 0, which starts a
     * new request under it.
     */
    std::set<std::string> dropping_;
    uint64_t serial_ = 0;
    bool closed_ = false;

    void take_line(const std::string & line);
    void cancel(const std::string & id);
    void chunk(const std::string & id, const JsonValue & message);
    /** A transcribe or an enhance, which ends the chunks of request `id` and queues it as a job of `kind`. */
    void end_chunks(Job::Kind kind, const std::string & id, const JsonValue & message);
    void request(Job::Kind kind, const std::string & type, const std::string & id, const JsonValue & message);

    /** An error for the line; with `id` when it names a request, and without one for a line that names none. */
    void reply_error(const std::string * id, const std::string & code, const std::string & option, const std::string & message);
    /** An error for a line that starts a request: with its id, or without one when a request under that id is in flight. */
    void refuse(const std::string & id, const std::string & code, const std::string & option, const std::string & message);
    /** Ends a request that is collecting chunks with an error, and drops its later lines. */
    void fail_collecting(const std::string & id, const std::string & option, const std::string & message);
    /** The members of `message` read for a job: refuses a member it does not have and returns false. */
    bool read_options(const JsonValue & message, const std::vector<std::string> & members, bool options, std::vector<RequestOption> & out,
                      std::string & option, std::string & problem) const;
    /** The members of a transcribe or an enhance: the options, and the rate of the chunks' audio. */
    bool read_rate(const JsonValue & message, std::vector<RequestOption> & options, int64_t & rate, std::string & option,
                   std::string & problem) const;
    bool in_flight(const std::string & id) const { return pending_.count(id) > 0; }
    Pending & open(const std::string & id, Pending::State state);
    void queue(Job job);
};
