# 031: overlap independent prepared requests

Status: source-only prototype. It has not been configured, compiled, run or timed.
The parent runtime is `114c514430c8aa1455c077819f8e72e674a48b75`; ggml remains
`e1a9ac12edbb57402618fd6936598603f5ee6266`. No earlier experiment is included.

## Boundary

This diagnostic executable uses the existing C API. Two separately loaded model
handles own independent backends, graph allocators, CPU decoder thread pools,
weights and position caches. It changes no library operation, public declaration,
worker protocol, server queue, model lock or cancellation rule. Loading and all
shape warm-ups finish before another thread starts, including device discovery
and the lazy CPU decoder weight copy for F16. The explicit thread count is per
handle: two CPU decoders can use twice that number together.

The second request starts when the first reports progress `2/3`, after its full
encoder graph, joint encoder projection and readback have finished. The first
request then decodes on the CPU while the second runs its frontend and encoder.
This is a two-request throughput experiment. An isolated request has no other
request to overlap, and its computation is unchanged. The runtime has global
attention and full-utterance frontend normalization, so feeding partial encoder
outputs into TDT would be a different algorithm.

The existing worker and server retain their FIFO behavior. In particular, the
server's one-model-per-task memory policy is not relaxed by this harness. A
production integration would require a separate queue and memory policy decision.

## Executable

`haas-overlap-031 MODEL DEVICE THREADS A.wav B.wav MODE TIMESTAMPS CANCEL`

Inputs are mono 16 kHz WAVE files of the pinned Haas model. `decode_on_cpu=true`
is explicit for every weight type, including F16. The modes are:

| Mode | Handles | Timed work |
| --- | --- | --- |
| `isolated` | 1 | A alone, without another executing request |
| `serial` | 1 | A then B on the existing single handle |
| `two-serial` | 2 | A on the first handle, then B on the second |
| `overlap` | 2 | B on the second handle after A's encoder boundary |

`TIMESTAMPS=0` and `CANCEL=none` are the performance condition. Diagnostics use
`TIMESTAMPS=1` to compare the public token texts, segment texts and their times.
The C API does not expose token IDs, TDT durations before timestamp processing,
logits or KDA state, so this check does not claim those internal values match.
The operation graph and arithmetic of each request are unchanged.

Each process loads one handle, runs A and B twice in sequence, and retains their
exact serialized public results. A two-handle condition then loads and warms the
second handle the same way, requiring the same results. Each handle's first
request is recorded separately; the second handle can benefit from process-wide
Metal caches and is not an independent cold-start measurement. The final warm
pair is compared with those references. Final results are read only after both
request threads have joined.

The requests and audio copies are prepared before the timer starts. Both are
prepared even in `isolated`, but B does not execute. The reported boundary is
the prepared C API call, not recording-end-to-worker-result latency. Both jobs
are considered available at the pair's common origin. `start_ms` is the service
start/queue delay, `service_ms` covers `speech_transcribe`, and `completion_ms`
includes the wait from that common origin. The overlap condition includes the
cost of launching and waking its follower thread. There is no batch-fill delay.

`prepared_pair_completion_ms` and RTFx describe the pair. A's `decode_tail_ms`
must be reported separately: B can slow A through CPU or unified-memory
contention even when the pair finishes earlier. The intersection of A's decoder
wall interval with B's encoder wall interval is recorded. It is not a GPU busy
trace or proof of simultaneous GPU/CPU execution throughout that interval.

Memory snapshots report the macOS physical footprint, resident bytes and the
process's peak RSS. `ready` follows both handles' warm-ups where applicable;
`warm_one` gives the single-handle state in the same process. Their difference
includes the second handle and its caches/arenas. It is not dedicated VRAM, and
the process peak cannot be attributed to one stage. Capture `/usr/bin/time -l`
as well when measuring. Model file size alone is not a memory measurement.

## Cancellation and lifetime checks

`CANCEL` can be `a-before`, `b-before`, `a-encoder`, `b-encoder`, `a-decode` or
`b-decode`, only with `overlap`. A `before` condition calls the existing cancel
API before running that request and requires no progress callbacks. Encoder
cancellation returns nonzero from that request's `2/3` callback. Decode
cancellation returns nonzero at its first progress above `2/3` and requires it
to be below `1`, so a too-short case cannot silently become a final-only cancel.
The other handle must still produce its reference result. Fresh requests on
both handles must then reproduce the reference results.

If A ends before its encoder boundary, B starts when A ends; this is recorded as
a cancellation diagnostic, not a performance overlap. A failed asynchronous
launch leaves no running follower. If A throws after launch, the gate releases
the follower, and the harness joins it before requests, models or input storage
can be destroyed. Callbacks touch only their own trace and the synchronized gate.
These are callback/pre-start cancellation checks, not a new cross-thread
`speech_request_cancel` or worker queued-cancellation test.

## Execution gate and evaluation plan

Do not execute these commands until the coordinator grants the performance lease.
Use an empty build directory, record its exact source/ggml commit, compiler/SDK,
configure command, generated flags and executable SHA-256, and retain the binary.
The outer CMake file only adds this diagnostic target to the unchanged runtime.

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake \
  -S "$TRIAL_SOURCE/experiments/haas-mac/031" -B "$TRIAL_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake \
  --build "$TRIAL_BUILD" --target haas-overlap-031 -j4
```

First run the same pair in all four modes with timestamps, then the six cancel
conditions and their restart checks. Include distinct A/B recordings and A=B.
Use Q4_K, Q8_0 and F16, CPU decoder 4, with short, medium and long pairs and
reversed input order. Keep all modes on the exact same executable, model and
input bytes. Stop on a failed comparison instead of relaxing it. A Mac result
does not verify Vulkan, and these differential checks do not replace official
stage dumps.

If checks pass, alternate serial/overlap order for at least three rounds, adding
two-serial to separate extra-handle effects and isolated to show A without a
competing request. Preserve raw JSON/stdout/stderr and full environment at each
launch, reject unrecorded `GGML_`/`SPEECH_` keys, and hold the shared lock for each
process. Hash the pinned model and WAV/PCM inputs under that same lease. Report
paired request p50/p90/p99, pair makespan/RTFx, A's tail, B's queue delay, load,
first-request and memory separately. Do not pool these prepared-pair results
into the campaign's primary single-request worker latency or accuracy tables.

The first screen is bounded to representative pairs; it does not require another
650-utterance quality run unless a later integration changes individual request
behavior. There is no measured gain or adoption decision at this checkpoint.
