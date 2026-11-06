#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <vector>

/** Measures only the worker transport experiment; it never stores a message's payload or id. */
struct TransportTrace {
    struct Stamp {
        int64_t wall = -1;
        int64_t cpu = -1;
    };

    struct Line {
        uint64_t ordinal = 0;
        size_t bytes = 0;
        uint64_t serial = 0;
        int64_t seq = -1;
        const char * kind = "other";
        Stamp read_begin, read_end, locked, parse_begin, parse_end;
        Stamp decode_begin, decode_end, append_begin, append_end, queued, dispatch_end;
    };

    struct Job {
        uint64_t serial = 0;
        bool recognition = false;
        Stamp taken, entered, api_begin, api_end, terminal_ready, finish_returned;
    };

    const bool enabled;
    const std::chrono::steady_clock::time_point origin = std::chrono::steady_clock::now();
    // The reader alone appends lines, and the worker alone appends jobs. The inbox's closed flag publishes the
    // reader's final writes before the worker drains and prints either vector.
    std::vector<Line> lines;
    std::vector<Job> jobs;
    Stamp eof;

    explicit TransportTrace(bool on) : enabled(on) {
        if (enabled) {
            lines.reserve(256);
            jobs.reserve(64);
        }
    }

    /** Wall time shares one process origin; CPU time belongs to the calling thread alone. */
    Stamp stamp() const {
        Stamp result;
        result.wall = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - origin).count();
#if defined(CLOCK_THREAD_CPUTIME_ID)
        timespec value{};
        if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0) result.cpu = int64_t(value.tv_sec) * 1000000000 + value.tv_nsec;
#endif
        return result;
    }

    /** Prints after stdin closes and all queued work drains, never inside a timed request. */
    void dump() const {
        if (!enabled) return;
        std::fprintf(stderr, "haas-transport {\"schema\":1,\"event\":\"drained\",\"lines\":%zu,\"jobs\":%zu", lines.size(), jobs.size());
        point("eof", eof);
        std::fputs("}\n", stderr);
        for (const Line & row : lines) {
            std::fprintf(stderr, "haas-transport {\"schema\":1,\"event\":\"line\",\"line\":%llu,\"bytes\":%zu,\"serial\":%llu,\"seq\":%lld,\"kind\":\"%s\"",
                         (unsigned long long) row.ordinal, row.bytes, (unsigned long long) row.serial, (long long) row.seq, row.kind);
            point("read_begin", row.read_begin);
            point("read_end", row.read_end);
            point("locked", row.locked);
            point("parse_begin", row.parse_begin);
            point("parse_end", row.parse_end);
            point("decode_begin", row.decode_begin);
            point("decode_end", row.decode_end);
            point("append_begin", row.append_begin);
            point("append_end", row.append_end);
            point("queued", row.queued);
            point("dispatch_end", row.dispatch_end);
            std::fputs("}\n", stderr);
        }
        for (const Job & row : jobs) {
            std::fprintf(stderr, "haas-transport {\"schema\":1,\"event\":\"job\",\"serial\":%llu,\"recognition\":%s",
                         (unsigned long long) row.serial, row.recognition ? "true" : "false");
            point("taken", row.taken);
            point("entered", row.entered);
            point("api_begin", row.api_begin);
            point("api_end", row.api_end);
            point("terminal_ready", row.terminal_ready);
            point("finish_returned", row.finish_returned);
            std::fputs("}\n", stderr);
        }
        std::fflush(stderr);
    }

private:
    static void point(const char * name, Stamp value) {
        std::fprintf(stderr, ",\"%s\":[%lld,%lld]", name, (long long) value.wall, (long long) value.cpu);
    }
};
