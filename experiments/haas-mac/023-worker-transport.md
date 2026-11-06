# 023: diagnose worker transport

This source-only experiment starts at `114c514430c8aa1455c077819f8e72e674a48b75`, with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. No C++ build, worker invocation, model execution, GPU work,
or timing measurement has been performed for this checkpoint. The earlier 002b residual of about 42–50%
does not establish its cause. Its small PCM copy and cleanup/flush intervals do not measure input transport.

The separate instrumentation-free candidate is `codex/exp-haas-023` at
`efa2de7b05d2ebf94e66711ee4fa1669814f5e25`. Both branches use the same startup-only
`SPEECH_HAAS_WORKER_UNSYNC` and `SPEECH_HAAS_WORKER_UNTIE` controls. Each is strict `0`/`1`, absent means
off, and both are validated before either changes a stream. They apply to all worker tasks, not only Haas.
The library and other subcommands do not read them. `tools/cli/main.cpp` still sets Windows streams to binary;
`take_stdout()` still reserves the protocol's separate descriptor; `Protocol::line()` still writes and flushes
that `FILE *`. No parser, base64 algorithm, chunk size, resampling, cancel/retry state, defaults, or C API changes.

`SPEECH_HAAS_WORKER_TRANSPORT_TRACE=1` enables this branch's diagnostic records. It accepts only `0`/`1`,
defaults off, and is validated before model loading and the reader thread. It does not belong in the normal
performance runner's allowlist. The diagnostic binary with tracing off still contains instrumentation branches
and is not the instrumentation-free performance candidate.

## Boundaries

| Process/thread | Recorded boundaries | Meaning and limitation |
|---|---|---|
| Python sender | slice/base64, JSON dumps, TextIO write, flush | Write includes Python text encoding/buffering; it is not a syscall duration. |
| Python stdout reader | readline, JSON loads, queue put | Readline can begin before the request and includes idle/wait time. |
| Python caller | queue get, terminal validation, original request timer | Queue get overlaps the worker's computation. |
| C++ input reader | getline, inbox lock acquisition, parse_json, base64 decode, PCM append, dispatch, queue insertion | Wall and thread CPU are separate; getline can wait through an earlier request. |
| C++ worker | job taken, recognition entry, transcription API, terminal ready, finish returned | Finish includes lock/write/flush and may return after the Python timer ends. |

C++ wall points share one `steady_clock` origin. `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` measures the calling
thread's CPU when supported; unavailable or failed CPU sampling remains `-1` in raw data and `null` in summaries.
CPU timestamps are never subtracted across threads. These clocks do not count syscalls. Wall minus CPU can include
producer idle, scheduling, and waits; it does not identify the cause as stdio synchronization or a pipe syscall.

The Python request timer still uses `perf_counter()` immediately before the first chunk's base64 work and after
terminal validation. PCM slices stay at 32768 bytes; JSON, text UTF-8 pipes, per-line flush, ordering, warmup,
and rotation follow the frozen driver at `09cc03857b4af626fd537ee2be85f78c00d31135`, whose `worker.py` SHA-256 is
`9049921d06e74580db1000e27dc5fbb7e4c07a4c0ed1559b4a6cf87a00b7cb88`. That driver's recorded-environment guard is retained.
Diagnostic calls and bookkeeping add overhead within those boundaries and require their own control runs.

`TransportTrace` stores only fixed timing fields, byte counts, normalized line kinds, line ordinals, and the
existing request serial. A reused ID receives a different serial. The input reader and worker each append to
their own vector. The inbox's closed mutex handoff publishes the reader's final records before the main loop
finishes. Only then, after `close_collecting()`, are records emitted to stderr with `haas-transport ` prefixes.
No trace is written while a request runs. A fatal exit before drain has no complete trace and cannot be summarized.
stdout and its protocol version remain unchanged. Trace records contain no request IDs, PCM, base64, transcripts,
or token strings. The diagnostic result stores a digest of text/tokens/segments/stop after the timer ends.

`transport_summary.py` requires a complete, healthy serialized recognition run and matches every Python input
line to the C++ line and request serial. It rejects missing records, missing stages, duplicate serials, mismatched
bytes/kinds/sequences, and invalid interval ordering. A cancellation/error trace can remain raw evidence but is
not silently accepted as a complete recognition timing. Its interval columns remain separate:

- Do not add client and worker durations or overlapping/nested spans.
- Python and C++ epochs are unrelated. Only the duration from the first chunk's C++ read completion to terminal
  readiness is wholly inside the paired Python request; subtract that duration per request, never two aggregate medians.
- That remainder is **unattributed**, not measured IPC overhead. Never subtract the initial whole getline or
  `finish_returned` from the Python request. The former can start before it and the latter can end after it.

## Checks and next execution

`python3 -B experiments/haas-mac/transport_check.py` passes synthetic accounting fixtures, strict environment
guards, missing CPU handling, and a byte/flush comparison against the pinned driver using an in-memory peer.
It reads the exact pinned driver's Git object and never fetches it or starts a worker; a clone must contain that
object. Both drivers send identical JSON bytes and six per-line flushes for the fixture's warmup and measured
requests, and diagnostic output contains no fixture transcript or PCM payload. These are not runtime C++ checks.

Once the coordinator grants a compute slot, build separate original, diagnostic, and instrumentation-free trees
with `-j4`, holding `/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock` for each build or run only.
Use a small fixed public 2–30 second corpus first. A single existing worker spec is the input to:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock python3 experiments/haas-mac/transport_worker.py SPEC.json
python3 experiments/haas-mac/transport_summary.py RUN_DIRECTORY
```

The spec pins binary/model hashes, threads, records, environment, and a new output directory under `outputs/`.
The diagnostic driver retains the campaign's Metal encoder/CPU decoder invocation; CPU correctness runs use the
existing smoke/API clients separately. Never reuse a measured directory. For a complete C++ summary the trace
switch must be explicitly `1`; an off run intentionally has no worker trace and the summarizer fails.

Control instrumentation before interpreting durations: original binary/original runner; diagnostic binary with
trace absent/original runner; the same diagnostic binary with trace absent/diagnostic driver; then trace on with
the diagnostic driver. Record each driver and binary hash. The middle pair isolates Python probe overhead, and
the last pair isolates C++ collection overhead. Diagnostic memory includes its record buffers.

Use the observed boundaries to choose whether either stdio control is worth screening. Then run the separate
instrumentation-free candidate with the frozen normal runner for three alternating rounds. All four stream
settings need protocol checks, including invalid flags, ready/protocol 4, stdout purity, chunks and errors,
interleaved requests, collecting/waiting/running cancels, retry/ID reuse, and EOF while collecting. Run the existing
recognition, synthesis, enhancement and C API checks since the worker input is shared. Verify exact Haas plain/
boosted text, tokens, times, Q4_K/Q8_0/F16 on CPU4 and Metal, then Windows binary transport and the required Vulkan
checks. These checks and all speed claims remain pending. Do not change parser/base64/chunks/resampler in the same trial.
