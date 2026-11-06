#include "inbox.h"

#include <algorithm>
#include <climits>
#include <stdexcept>
#include <utility>

#include "base64.h"
#include "json.h"
#include "library.h"

namespace {

const char * const kTypes = "synthesize, chunk, transcribe, enhance, add_voice, info, count_tokens or cancel";

/** The requests a model of `task` takes, for a message that refuses another. */
std::string requests_of(speech_task task) {
    switch (task) {
        case SPEECH_TASK_SYNTHESIS: return "synthesize";
        case SPEECH_TASK_RECOGNITION: return "chunk and transcribe";
        case SPEECH_TASK_ENHANCEMENT: return "chunk and enhance";
        default: return "none";
    }
}

std::string joined(const std::vector<std::string> & names) {
    std::string out;
    for (const std::string & n : names) out += (out.empty() ? "" : ", ") + n;
    return out;
}

/** What a JSON value is, for a message that refuses it. */
std::string kind_name(const JsonValue & v) {
    switch (v.kind) {
        case JsonValue::Kind::Null: return "null";
        case JsonValue::Kind::Bool: return "a boolean";
        case JsonValue::Kind::Number: return "a number";
        case JsonValue::Kind::String: return "a string";
        case JsonValue::Kind::Array: return "an array";
        case JsonValue::Kind::Object: return "an object";
    }
    return "";
}

/** A string member, or nullptr when it is absent; a member of another type sets `problem`. */
const std::string * string_member(const JsonValue & message, const char * name, std::string & problem) {
    const JsonValue * v = message.member(name);
    if (!v) return nullptr;
    if (v->kind != JsonValue::Kind::String) {
        problem = std::string("\"") + name + "\" is " + json_excerpt(*v) + "; give it as a string";
        return nullptr;
    }
    return &v->text;
}

/** A whole-number member from `minimum` to INT_MAX, or false with `problem` set; `what` says what it is. */
bool count_member(const JsonValue & message, const char * name, int64_t minimum, const std::string & what, int64_t & out, std::string & problem) {
    const JsonValue * v = message.member(name);
    if (!v) {
        problem = std::string("the message has no \"") + name + "\", " + what;
        return false;
    }
    if (v->is_integer() && v->text.size() <= 12) {
        out = std::stoll(v->text);
        if (out >= minimum && out <= INT_MAX) return true;
    }
    problem = std::string("\"") + name + "\" is " + json_excerpt(*v) + "; give " + what + " as a whole number from " + std::to_string(minimum);
    return false;
}

}  // namespace

void Protocol::line(const std::string & json) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::fwrite(json.data(), 1, json.size(), out_);
    std::fputc('\n', out_);
    std::fflush(out_);
}

std::string error_object(const std::string & code, const std::string & option, const std::string & message) {
    return "{\"code\":" + json_string(code) + ",\"option\":" + (option.empty() ? "null" : json_string(option)) + ",\"message\":" +
           json_string(message) + "}";
}

void Inbox::read(std::istream & in) {
    std::string line;
    while (true) {
        TransportTrace::Line timing;
        if (trace_.enabled) timing.read_begin = trace_.stamp();
        if (!std::getline(in, line)) break;
        if (trace_.enabled) {
            timing.read_end = trace_.stamp();
            timing.ordinal = trace_.lines.size() + 1;
            timing.bytes = line.size();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (trace_.enabled) {
                timing.locked = trace_.stamp();
                tracing_line_ = &timing;
            }
            take_line(line);
            if (trace_.enabled) timing.dispatch_end = trace_.stamp();
            tracing_line_ = nullptr;
        }
        if (trace_.enabled) trace_.lines.push_back(timing);
    }
    if (trace_.enabled) trace_.eof = trace_.stamp();
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
    ready_.notify_all();
}

bool Inbox::take(Job & job) {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [&] { return !queue_.empty() || closed_; });
    if (queue_.empty()) return false;
    job = std::move(queue_.front());
    queue_.pop_front();
    if (trace_.enabled) {
        TransportTrace::Job timing;
        timing.serial = job.serial;
        timing.recognition = job.kind == Job::Kind::Transcribe;
        timing.taken = trace_.stamp();
        trace_.jobs.push_back(timing);
    }
    return true;
}

bool Inbox::start(const Job & job, Cancellation * cancellation) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = pending_.find(job.id);
    if (it == pending_.end() || it->second.serial != job.serial) return false;
    it->second.state = Pending::State::Running;
    it->second.running = cancellation;
    return true;
}

void Inbox::finish(const Job & job, const std::string & line) {
    std::lock_guard<std::mutex> lock(mutex_);
    pending_.erase(job.id);
    protocol_.line(line);
}

void Inbox::close_collecting() {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool enhancement = task_ == SPEECH_TASK_ENHANCEMENT;
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (it->second.state == Pending::State::Collecting) {
            reply_error(&it->first, speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "",
                        enhancement ? "stdin closed before the request's enhance line, so its audio was not enhanced"
                                    : "stdin closed before the request's transcribe line, so its audio was not recognized");
            it = pending_.erase(it);
        } else {
            ++it;
        }
    }
}

void Inbox::reply_error(const std::string * id, const std::string & code, const std::string & option, const std::string & message) {
    protocol_.line("{\"type\":\"error\"," + (id ? "\"id\":" + json_string(*id) + "," : std::string()) + "\"error\":" +
                   error_object(code, option, message) + "}");
}

void Inbox::refuse(const std::string & id, const std::string & code, const std::string & option, const std::string & message) {
    reply_error(in_flight(id) ? nullptr : &id, code, option, message);
}

void Inbox::fail_collecting(const std::string & id, const std::string & option, const std::string & message) {
    pending_.erase(id);
    reply_error(&id, speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), option, message);
    dropping_.insert(id);
}

Inbox::Pending & Inbox::open(const std::string & id, Pending::State state) {
    Pending & p = pending_[id];
    p.serial = ++serial_;
    p.state = state;
    if (tracing_line_) tracing_line_->serial = p.serial;
    return p;
}

void Inbox::queue(Job job) {
    queue_.push_back(std::move(job));
    if (tracing_line_) tracing_line_->queued = trace_.stamp();
    ready_.notify_one();
}

bool Inbox::read_options(const JsonValue & message, const std::vector<std::string> & members, bool options, std::vector<RequestOption> & out,
                         std::string & option, std::string & problem) const {
    const std::string type = message.member("type")->text;
    for (const auto & [name, value] : message.members) {
        if (std::find(members.begin(), members.end(), name) != members.end()) continue;
        speech_option o;
        if (!options || !option_named(name, o)) {
            option = name;
            problem = type + " has no member \"" + name + "\"; its members are " + joined(members) + (options ? " and the options " + option_names() : "");
            return false;
        }
        try {
            out.push_back({o, option_from_json(o, value)});
        } catch (const std::invalid_argument & e) {
            option = name;
            problem = e.what();
            return false;
        }
    }
    return true;
}

bool Inbox::read_rate(const JsonValue & message, std::vector<RequestOption> & options, int64_t & rate, std::string & option,
                      std::string & problem) const {
    if (!read_options(message, {"type", "id", "sample_rate"}, true, options, option, problem)) return false;
    option = "sample_rate";
    return count_member(message, "sample_rate", 1, "the rate of the chunks' audio in Hz", rate, problem);
}

void Inbox::take_line(const std::string & line) {
    const std::string invalid = speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT);
    JsonValue message;
    if (tracing_line_) tracing_line_->parse_begin = trace_.stamp();
    try {
        message = parse_json(line);
    } catch (const std::invalid_argument & e) {
        if (tracing_line_) tracing_line_->parse_end = trace_.stamp();
        reply_error(nullptr, invalid, "", std::string("the line is not JSON: ") + e.what() + "; send one JSON object per line");
        return;
    }
    if (tracing_line_) tracing_line_->parse_end = trace_.stamp();
    if (message.kind != JsonValue::Kind::Object) {
        reply_error(nullptr, invalid, "", "the line is " + kind_name(message) + ", not a JSON object; send one JSON object per line");
        return;
    }
    // A member set to null counts as left out.
    for (auto it = message.members.begin(); it != message.members.end();) {
        it = it->second.kind == JsonValue::Kind::Null ? message.members.erase(it) : it + 1;
    }
    const JsonValue * id_value = message.member("id");
    if (!id_value || id_value->kind != JsonValue::Kind::String || id_value->text.empty()) {
        reply_error(nullptr, invalid, "id", "the message has no \"id\" that is a non-empty string; every message names its request by one");
        return;
    }
    const std::string id = id_value->text;
    if (tracing_line_) {
        const auto pending = pending_.find(id);
        if (pending != pending_.end()) tracing_line_->serial = pending->second.serial;
    }
    const JsonValue * type = message.member("type");
    if (!type || type->kind != JsonValue::Kind::String) {
        refuse(id, invalid, "type", std::string("the message has no \"type\" that is a string; give one of ") + kTypes);
        return;
    }
    const std::string & t = type->text;
    if (tracing_line_) {
        tracing_line_->kind = t == "chunk" ? "chunk" : t == "transcribe" ? "transcribe" : t == "cancel" ? "cancel" : "other";
    }
    if (t == "cancel") {
        for (const auto & member : message.members) {
            if (member.first == "type" || member.first == "id") continue;
            // A cancel has no answer of its own, so an error with its id would read as the answer of its request.
            reply_error(nullptr, invalid, member.first, "cancel has no member \"" + member.first + "\"; its members are id and type");
            return;
        }
        cancel(id);
    } else if (t == "chunk") {
        chunk(id, message);
    } else if (t == "transcribe") {
        end_chunks(Job::Kind::Transcribe, id, message);
    } else if (t == "enhance") {
        end_chunks(Job::Kind::Enhance, id, message);
    } else if (t == "synthesize") {
        request(Job::Kind::Synthesize, t, id, message);
    } else if (t == "add_voice") {
        request(Job::Kind::AddVoice, t, id, message);
    } else if (t == "info") {
        request(Job::Kind::Info, t, id, message);
    } else if (t == "count_tokens") {
        request(Job::Kind::CountTokens, t, id, message);
    } else {
        refuse(id, invalid, "type", "the message's \"type\" is " + json_string(t) + "; give one of " + kTypes);
    }
}

void Inbox::cancel(const std::string & id) {
    const auto it = pending_.find(id);
    if (it == pending_.end()) return;
    Pending & p = it->second;
    if (p.state == Pending::State::Running) {
        // The run ends with `cancelled`, or with its answer when it was past the point where it could stop.
        if (p.running) p.running->cancel();
        return;
    }
    if (p.state == Pending::State::Collecting) dropping_.insert(id);
    pending_.erase(it);
    protocol_.line("{\"type\":\"cancelled\",\"id\":" + json_string(id) + "}");
}

void Inbox::request(Job::Kind kind, const std::string & type, const std::string & id, const JsonValue & message) {
    const std::string invalid = speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT);
    if (in_flight(id)) {
        reply_error(nullptr, invalid, "id", "a request under the id " + json_string(id) + " has not had its answer; give the " + type +
                                                " an id that no request in flight has");
        return;
    }
    if (kind == Job::Kind::Synthesize && task_ != SPEECH_TASK_SYNTHESIS) {
        reply_error(&id, speech_status_name(SPEECH_ERROR_UNSUPPORTED), "type",
                    model_name_ + " is a model of speech " + task_name(task_) + ", which takes " + requests_of(task_) +
                        "; synthesize needs a synthesis model");
        return;
    }
    Job job;
    job.kind = kind;
    job.id = id;
    std::vector<std::string> members = {"type", "id"};
    if (kind == Job::Kind::Synthesize || kind == Job::Kind::CountTokens) members.push_back("text");
    if (kind == Job::Kind::AddVoice) members.insert(members.end(), {"name", "path"});
    std::string option, problem;
    if (!read_options(message, members, kind == Job::Kind::Synthesize, job.options, option, problem)) {
        reply_error(&id, invalid, option, problem);
        return;
    }
    for (const std::string & name : members) {
        if (name == "type" || name == "id") continue;
        const std::string * value = string_member(message, name.c_str(), problem);
        if (!value && problem.empty()) {
            const char * what = name == "text" ? "the text" : name == "name" ? "the voice's name" : "the voice file or WAVE file";
            problem = type + " needs \"" + name + "\", " + what;
        }
        if (!value) {
            reply_error(&id, invalid, name, problem);
            return;
        }
        (name == "text" ? job.text : name == "name" ? job.name : job.path) = *value;
    }
    if (kind == Job::Kind::Info || kind == Job::Kind::CountTokens) {
        protocol_.line(at_once_(job));
        return;
    }
    job.serial = open(id, Pending::State::Waiting).serial;
    queue(std::move(job));
}

void Inbox::chunk(const std::string & id, const JsonValue & message) {
    const JsonValue * seq_value = message.member("seq");
    if (dropping_.count(id)) {
        // A chunk 0 under the id of a request that was answered starts a new request; any other is the old one's.
        if (!seq_value || !seq_value->is_integer() || seq_value->text != "0") return;
        dropping_.erase(id);
    }
    if (in_flight(id) && pending_[id].state != Pending::State::Collecting) {
        reply_error(nullptr, speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT), "id",
                    "the request " + json_string(id) + " has had its transcribe or enhance, or takes no chunks; send a request's chunks before "
                    "its transcribe or enhance, under an id that no other request in flight has");
        return;
    }
    if (!in_flight(id) && task_ != SPEECH_TASK_RECOGNITION && task_ != SPEECH_TASK_ENHANCEMENT) {
        reply_error(&id, speech_status_name(SPEECH_ERROR_UNSUPPORTED), "type",
                    model_name_ + " is a model of speech " + task_name(task_) + ", which takes " + requests_of(task_) +
                        "; chunk needs a recognition or enhancement model");
        dropping_.insert(id);
        return;
    }
    Pending & p = in_flight(id) ? pending_[id] : open(id, Pending::State::Collecting);
    if (tracing_line_) tracing_line_->serial = p.serial;
    std::vector<RequestOption> none;
    std::string option, problem;
    if (!read_options(message, {"type", "id", "seq", "pcm"}, false, none, option, problem)) {
        fail_collecting(id, option, problem);
        return;
    }
    int64_t seq = 0;
    if (!count_member(message, "seq", 0, "the chunk's number in its request", seq, problem)) {
        fail_collecting(id, "seq", problem);
        return;
    }
    if (tracing_line_) tracing_line_->seq = seq;
    if ((uint64_t) seq != p.next_seq) {
        fail_collecting(id, "seq", "the chunk's \"seq\" is " + std::to_string(seq) + " where " + std::to_string(p.next_seq) +
                                       " comes next; send a request's chunks in order from 0");
        return;
    }
    const std::string * pcm = string_member(message, "pcm", problem);
    if (!pcm) {
        fail_collecting(id, "pcm", problem.empty() ? "the chunk has no \"pcm\", its audio as base64 of 16-bit little-endian samples" : problem);
        return;
    }
    std::string bytes;
    if (tracing_line_) tracing_line_->decode_begin = trace_.stamp();
    try {
        bytes = base64_decode(*pcm);
    } catch (const std::invalid_argument & e) {
        if (tracing_line_) tracing_line_->decode_end = trace_.stamp();
        fail_collecting(id, "pcm", std::string("the chunk's \"pcm\" is not base64: ") + e.what());
        return;
    }
    if (tracing_line_) tracing_line_->decode_end = trace_.stamp();
    if (bytes.size() % 2 != 0) {
        fail_collecting(id, "pcm", "the chunk's \"pcm\" holds an odd number of bytes; it is 16-bit samples");
        return;
    }
    if (tracing_line_) tracing_line_->append_begin = trace_.stamp();
    const size_t at = p.pcm.size();
    p.pcm.resize(at + bytes.size() / 2);
    for (size_t i = 0; i < bytes.size() / 2; i++) p.pcm[at + i] = (int16_t) (uint16_t) ((uint8_t) bytes[2 * i] | (uint8_t) bytes[2 * i + 1] << 8);
    p.next_seq++;
    if (tracing_line_) tracing_line_->append_end = trace_.stamp();
}

void Inbox::end_chunks(Job::Kind kind, const std::string & id, const JsonValue & message) {
    const std::string invalid = speech_status_name(SPEECH_ERROR_INVALID_ARGUMENT);
    const bool enhance = kind == Job::Kind::Enhance;
    const std::string type = enhance ? "enhance" : "transcribe";
    if (dropping_.erase(id)) return;
    if (in_flight(id) && pending_[id].state != Pending::State::Collecting) {
        reply_error(nullptr, invalid, "id", "a request under the id " + json_string(id) + " has not had its answer; give the " + type +
                                                " the id of the chunks it ends, or an id that no request in flight has");
        return;
    }
    if (task_ != (enhance ? SPEECH_TASK_ENHANCEMENT : SPEECH_TASK_RECOGNITION)) {
        // The chunks before it, which a model of the other task that takes chunks has gathered, end with the request.
        pending_.erase(id);
        reply_error(&id, speech_status_name(SPEECH_ERROR_UNSUPPORTED), "type",
                    model_name_ + " is a model of speech " + task_name(task_) + ", which takes " + requests_of(task_) + "; " + type + " needs " +
                        (enhance ? "an enhancement model" : "a recognition model"));
        return;
    }
    Pending & p = in_flight(id) ? pending_[id] : open(id, Pending::State::Collecting);
    if (tracing_line_) tracing_line_->serial = p.serial;
    Job job;
    job.kind = kind;
    job.id = id;
    std::string option, problem;
    int64_t rate = 0;
    if (!read_rate(message, job.options, rate, option, problem)) {
        pending_.erase(id);
        reply_error(&id, invalid, option, problem);
        return;
    }
    job.sample_rate = (int) rate;
    job.pcm = std::move(p.pcm);
    job.serial = p.serial;
    p.state = Pending::State::Waiting;
    queue(std::move(job));
}
