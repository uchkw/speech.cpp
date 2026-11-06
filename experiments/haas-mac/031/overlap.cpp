/*
 * Measures prepared C API requests on one or two independent model handles.
 * Usage: haas-overlap-031 MODEL DEVICE THREADS A.wav B.wav MODE TIMESTAMPS CANCEL
 * MODE is isolated, serial, two-serial or overlap. TIMESTAMPS is 0 or 1.
 * CANCEL is none, a-before, b-before, a-encoder, b-encoder, a-decode or b-decode.
 * Model loading and shape warm-ups precede the timed requests. No worker or server queue changes.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <future>
#include <mach/mach.h>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

#include "json.h"
#include "speech.h"
#include "wav.h"

namespace {

using Clock = std::chrono::steady_clock;
using Time = Clock::time_point;
using Model = std::unique_ptr<speech_model, decltype(&speech_model_free)>;
using Request = std::unique_ptr<speech_request, decltype(&speech_request_free)>;

double ms(Time from, Time to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

void ok(speech_status status, const char * what) {
    if (status != SPEECH_OK) throw std::runtime_error(std::string(what) + ": " + speech_last_error());
}

std::string memory() {
    task_vm_info_data_t vm{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    require(task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&vm), &count) == KERN_SUCCESS,
            "task_info cannot read the process footprint");
    rusage usage{};
    require(getrusage(RUSAGE_SELF, &usage) == 0, "getrusage cannot read the process peak RSS");
    return "{\"physical_footprint_bytes\":" + std::to_string(vm.phys_footprint) +
           ",\"resident_bytes\":" + std::to_string(vm.resident_size) +
           ",\"process_peak_rss_bytes\":" + std::to_string(usage.ru_maxrss) + "}";
}

/** A follower waits for the encoder boundary or for a leader that ends before reaching it. */
class Gate {
public:
    void encoder() {
        std::lock_guard<std::mutex> guard(mutex_);
        encoder_ = true;
        cv_.notify_one();
    }
    void finish(bool abort = false) {
        std::lock_guard<std::mutex> guard(mutex_);
        ended_ = true;
        abort_ = abort;
        cv_.notify_one();
    }
    bool wait() {
        std::unique_lock<std::mutex> guard(mutex_);
        cv_.wait(guard, [&] { return encoder_ || ended_; });
        return !abort_;
    }
private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool encoder_ = false, ended_ = false, abort_ = false;
};

struct Job {
    Request request{nullptr, speech_request_free};
    Gate * gate = nullptr;
    Time origin{}, start{}, finish{};
    double frontend = -1, encoder = -1, first_decode = -1, last_progress = -1, cancelled_at = -1;
    int callbacks = 0;
    bool bad_callback = false;
    std::string cancel, error;
    speech_status status = SPEECH_ERROR_INTERNAL;

    Job(speech_model * model, const Wav & audio, bool timestamps, std::string stop = "none") : cancel(std::move(stop)) {
        speech_request * raw = nullptr;
        ok(speech_request_new(model, &raw), "create request");
        request.reset(raw);
        ok(speech_request_set_audio(raw, audio.samples.data(), audio.samples.size(), audio.sample_rate), "set audio");
        ok(speech_request_set_bool(raw, SPEECH_OPT_DECODE_ON_CPU, 1), "set CPU decoding");
        ok(speech_request_set_bool(raw, SPEECH_OPT_TIMESTAMPS, timestamps), "set timestamps");
        ok(speech_request_set_progress(raw, progress, this), "set progress callback");
        if (cancel == "before") speech_request_cancel(raw);
    }

    static int progress(double done, void * opaque) noexcept {
        Job & self = *static_cast<Job *>(opaque);
        try {
            self.callbacks++;
            if (!std::isfinite(done) || done < self.last_progress || done < 0 || done > 1) self.bad_callback = true;
            self.last_progress = done;
            const double elapsed = ms(self.origin, Clock::now());
            if (done == 1.0 / 3 && self.frontend < 0) self.frontend = elapsed;
            if (done == 2.0 / 3 && self.encoder < 0) {
                self.encoder = elapsed;
                if (self.gate) self.gate->encoder();
            }
            if (done > 2.0 / 3 && self.first_decode < 0) self.first_decode = elapsed;
            if ((self.cancel == "encoder" && done == 2.0 / 3) || (self.cancel == "decode" && done > 2.0 / 3)) {
                self.cancelled_at = done;
                return 1;
            }
            return self.bad_callback ? 1 : 0;
        } catch (...) {
            self.bad_callback = true;
            return 1;
        }
    }

    void run() {
        start = Clock::now();
        status = speech_transcribe(request.get());
        finish = Clock::now();
        if (status < 0) error = speech_last_error();
    }

    std::string result() const {
        require(status == SPEECH_OK || status == SPEECH_CANCELLED, "transcribe failed: " + error);
        require(!bad_callback, "the progress callback failed or was not monotonic");
        const speech_result * r = speech_request_result(request.get());
        require(r != nullptr && speech_result_text(r) != nullptr, "recognition has no result");
        std::string out = "{\"status\":" + json_string(speech_status_name(status)) +
                          ",\"stop\":" + json_string(speech_stop_name(speech_result_stop(r))) +
                          ",\"text\":" + json_string(speech_result_text(r));
        const char * untimed = speech_result_untimed(r);
        out += ",\"untimed\":" + (untimed ? json_string(untimed) : "null");
        for (bool tokens : {true, false}) {
            out += tokens ? ",\"tokens\":[" : ",\"segments\":[";
            const size_t count = tokens ? speech_result_token_count(r) : speech_result_segment_count(r);
            for (size_t i = 0; i < count; i++) {
                double from, to;
                const char * text;
                ok(tokens ? speech_result_token(r, i, &from, &to, &text) : speech_result_segment(r, i, &from, &to, &text),
                   "read timed result");
                require(std::isfinite(from) && std::isfinite(to), "a result timestamp is not finite");
                out += (i ? "," : "") + std::string("{\"start\":") + json_number(from) + ",\"end\":" + json_number(to) +
                       ",\"text\":" + json_string(text) + "}";
            }
            out += "]";
        }
        out += ",\"languages\":[";
        for (size_t i = 0; i < speech_result_language_count(r); i++) {
            out += (i ? "," : "") + json_string(speech_result_language(r, i));
        }
        return out + "]}";
    }

    bool matches(const std::string & expected) const {
        if (cancel == "none") return status == SPEECH_OK && result() == expected;
        const speech_result * r = speech_request_result(request.get());
        return !bad_callback && status == SPEECH_CANCELLED && r && speech_result_stop(r) == SPEECH_STOP_CANCELLED &&
               std::string(speech_result_text(r)).empty() && speech_result_token_count(r) == 0 &&
               speech_result_segment_count(r) == 0 && speech_result_language_count(r) == 0 &&
               (cancel != "before" || callbacks == 0) && (cancel != "encoder" || cancelled_at == 2.0 / 3) &&
               (cancel != "decode" || (cancelled_at > 2.0 / 3 && cancelled_at < 1));
    }

    std::string report(const std::string & expected) const {
        return "{\"start_ms\":" + json_number(ms(origin, start)) + ",\"completion_ms\":" + json_number(ms(origin, finish)) +
               ",\"service_ms\":" + json_number(ms(start, finish)) + ",\"frontend_done_ms\":" + json_number(frontend) +
               ",\"encoder_done_ms\":" + json_number(encoder) + ",\"first_decode_progress_ms\":" + json_number(first_decode) +
               ",\"decode_tail_ms\":" + (encoder < 0 ? "null" : json_number(ms(origin, finish) - encoder)) +
               ",\"progress_callbacks\":" + std::to_string(callbacks) + ",\"cancel_progress\":" + json_number(cancelled_at) +
               ",\"output_matches_reference_or_cancel\":" + (matches(expected) ? "true" : "false") +
               ",\"result\":" + result() + "}";
    }
};

std::string recognize(speech_model * model, const Wav & audio, bool timestamps, double * elapsed = nullptr) {
    Job job(model, audio, timestamps);
    job.origin = Clock::now();
    job.run();
    if (elapsed) *elapsed = ms(job.start, job.finish);
    require(job.status == SPEECH_OK, "a warm-up or restart request did not complete");
    return job.result();
}

Model load(const char * path, const char * device, int threads, std::string & info, double & elapsed) {
    speech_load_params * raw_params = nullptr;
    ok(speech_load_params_new(&raw_params), "create load parameters");
    std::unique_ptr<speech_load_params, decltype(&speech_load_params_free)> params(raw_params, speech_load_params_free);
    ok(speech_load_params_set_device(params.get(), device), "set device");
    ok(speech_load_params_set_threads(params.get(), threads), "set threads");
    ok(speech_load_params_set_warmup(params.get(), 0), "disable load warm-up");
    speech_model * raw = nullptr;
    const Time start = Clock::now();
    ok(speech_model_load(path, params.get(), &raw), "load model");
    elapsed = ms(start, Clock::now());
    Model model(raw, speech_model_free);
    speech_model_info * raw_info = nullptr;
    ok(speech_model_get_info(raw, &raw_info), "read model information");
    std::unique_ptr<speech_model_info, decltype(&speech_model_info_free)> details(raw_info, speech_model_info_free);
    require(std::string(speech_model_info_architecture(raw_info)) == "fastconformer" &&
            std::string(speech_model_info_name(raw_info)) == "Haas-ja-v1.0", "031 requires Haas-ja-v1.0");
    require(speech_model_info_threads(raw_info) == threads, "the effective CPU thread count differs");
    info = speech_model_info_json(raw_info);
    return model;
}

int trial(int argc, char ** argv) {
    require(argc == 9, "usage: haas-overlap-031 MODEL DEVICE THREADS A.wav B.wav MODE TIMESTAMPS CANCEL");
    const std::string mode = argv[6], cancel = argv[8];
    require(mode == "isolated" || mode == "serial" || mode == "two-serial" || mode == "overlap", "unknown trial mode");
    require(std::string(argv[7]) == "0" || std::string(argv[7]) == "1", "TIMESTAMPS must be 0 or 1");
    const bool timestamps = std::string(argv[7]) == "1";
    const std::vector<std::string> cancels{"none", "a-before", "b-before", "a-encoder", "b-encoder", "a-decode", "b-decode"};
    require(std::find(cancels.begin(), cancels.end(), cancel) != cancels.end(), "unknown cancellation condition");
    require(cancel == "none" || mode == "overlap", "cancellation diagnostics require overlap mode");
    size_t used = 0;
    const int threads = std::stoi(argv[3], &used);
    require(used == std::string(argv[3]).size() && threads > 0, "THREADS must be a positive integer");
    Wav audio_a = read_wav(argv[4]), audio_b = read_wav(argv[5]);
    require(audio_a.channels == 1 && audio_b.channels == 1 && audio_a.sample_rate == 16000 && audio_b.sample_rate == 16000,
            "031 inputs must be mono 16000 Hz WAVE files");
    const bool two = mode == "two-serial" || mode == "overlap";
    std::string info_a, info_b;
    double load_a = 0, load_b = 0, cold_a = 0, cold_b = 0;
    const std::string memory_before = memory();
    Model model_a = load(argv[1], argv[2], threads, info_a, load_a);
    const std::string memory_loaded_one = memory();
    const std::string expected_a = recognize(model_a.get(), audio_a, timestamps, &cold_a);
    const std::string expected_b = recognize(model_a.get(), audio_b, timestamps);
    require(recognize(model_a.get(), audio_a, timestamps) == expected_a && recognize(model_a.get(), audio_b, timestamps) == expected_b,
            "the repeated shape warm-ups differ on the first handle");
    const std::string memory_warm_one = memory();
    Model model_b(nullptr, speech_model_free);
    if (two) {
        model_b = load(argv[1], argv[2], threads, info_b, load_b);
        require(info_a == info_b, "the two model handles report different information");
        require(recognize(model_b.get(), audio_a, timestamps, &cold_b) == expected_a &&
                recognize(model_b.get(), audio_b, timestamps) == expected_b &&
                recognize(model_b.get(), audio_a, timestamps) == expected_a &&
                recognize(model_b.get(), audio_b, timestamps) == expected_b, "the second handle's warm-ups differ");
    }
    const std::string memory_ready = memory();
    speech_model * follower_model = two ? model_b.get() : model_a.get();
    Job a(model_a.get(), audio_a, timestamps, cancel[0] == 'a' ? cancel.substr(2) : "none");
    Job b(follower_model, audio_b, timestamps, cancel[0] == 'b' ? cancel.substr(2) : "none");
    Gate gate;
    const Time origin = Clock::now();
    a.origin = b.origin = origin;
    if (mode == "overlap") {
        a.gate = &gate;
        auto follower = std::async(std::launch::async, [&] { if (gate.wait()) b.run(); });
        try {
            a.run();
        } catch (...) {
            gate.finish(true);
            follower.wait();
            throw;
        }
        gate.finish();
        follower.get();
    } else {
        a.run();
        if (mode != "isolated") b.run();
    }
    const std::string memory_completed = memory();
    const bool exact = a.matches(expected_a) && (mode == "isolated" || b.matches(expected_b));
    const bool boundary = mode != "overlap" || cancel == "a-before" ||
                          (a.encoder >= 0 && ms(origin, b.start) >= a.encoder);
    bool restart = true;
    if (cancel != "none") {
        restart = recognize(model_a.get(), audio_a, timestamps) == expected_a &&
                  recognize(model_b.get(), audio_b, timestamps) == expected_b;
    }
    const double end = mode == "isolated" ? ms(origin, a.finish) : std::max(ms(origin, a.finish), ms(origin, b.finish));
    const double overlap = mode != "isolated" && a.encoder >= 0 && b.encoder >= 0 && b.frontend >= 0 ?
                           std::max(0.0, std::min(ms(origin, a.finish), b.encoder) - std::max(a.encoder, b.frontend)) : 0;
    const double audio_seconds = (audio_a.samples.size() + (mode == "isolated" ? 0 : audio_b.samples.size())) / 16000.0;
    std::string out = "{\"measure\":\"031\",\"mode\":" + json_string(mode) + ",\"cancel\":" + json_string(cancel) +
        ",\"timestamps\":" + (timestamps ? "true" : "false") + ",\"decode_on_cpu\":true,\"threads\":" + std::to_string(threads) +
        ",\"model_info\":" + info_a + ",\"model_handles\":" + (two ? "2" : "1") +
        ",\"performance_evaluation_eligible\":" + (cancel == "none" && !timestamps ? "true" : "false") +
        ",\"load_ms\":[" + json_number(load_a) + "," + (two ? json_number(load_b) : "null") +
        "],\"first_request_ms\":[" + json_number(cold_a) + "," + (two ? json_number(cold_b) : "null") +
        "],\"memory\":{\"before\":" + memory_before + ",\"loaded_one\":" + memory_loaded_one +
        ",\"warm_one\":" + memory_warm_one + ",\"ready\":" + memory_ready + ",\"completed\":" + memory_completed +
        "},\"a\":" + a.report(expected_a) + ",\"b\":" + (mode == "isolated" ? "null" : b.report(expected_b)) +
        ",\"prepared_pair_completion_ms\":" + json_number(end) + ",\"rtfx\":" + json_number(audio_seconds * 1000 / end) +
        ",\"encoder_decode_wall_interval_overlap_ms\":" + json_number(overlap) +
        ",\"boundary_valid\":" + (boundary ? "true" : "false") + ",\"restart_exact\":" + (restart ? "true" : "false") +
        ",\"outputs_exact\":" + (exact ? "true" : "false") + "}";
    std::puts(out.c_str());
    return exact && boundary && restart ? 0 : 1;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        return trial(argc, argv);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "haas-overlap-031: %s\n", error.what());
        return 1;
    }
}
