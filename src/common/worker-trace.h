#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.h"

/** Diagnostic timestamps for one worker request, emitted together after its normal terminal message. */
namespace worker_trace {

using Clock = std::chrono::steady_clock;
struct Record;
inline thread_local Record * current = nullptr;

inline bool enabled() {
    static const bool value = [] {
        const char * setting = std::getenv("SPEECH_HAAS_WORKER_TRACE");
        if (!setting || std::strcmp(setting, "0") == 0) return false;
        if (std::strcmp(setting, "1") != 0) throw std::runtime_error("SPEECH_HAAS_WORKER_TRACE must be 0 or 1");
        return true;
    }();
    return value;
}

struct Record {
    struct Event {
        const char * name;
        Clock::time_point at;
    };
    const std::string & id;
    std::vector<Event> events;
    bool terminal = false;

    explicit Record(const std::string & id) : id(id) {
        if (!enabled()) return;
        events.reserve(48);
        current = this;
        events.push_back({"worker_enter", Clock::now()});
    }

    ~Record() {
        if (current != this) return;
        events.push_back({"worker_exit", Clock::now()});
        current = nullptr;
        std::string line = "{\"haas_worker_trace\":1,\"id\":" + json_string(id) + ",\"points_ms\":{";
        for (size_t i = 0; i < events.size(); ++i) {
            const double ms = std::chrono::duration<double, std::milli>(events[i].at - events[0].at).count();
            char value[48];
            std::snprintf(value, sizeof value, "%.9f", ms);
            line += (i ? "," : "") + json_string(events[i].name) + ":" + value;
        }
        line += "}}\n";
        std::fwrite(line.data(), 1, line.size(), stderr);
        std::fflush(stderr);
    }
};

inline void mark(const char * name) {
    if (current) current->events.push_back({name, Clock::now()});
}

/** Marks a function's exit after the locals constructed after this guard have been destroyed. */
struct OnExit {
    const char * name;
    ~OnExit() { mark(name); }
};

}  // namespace worker_trace
