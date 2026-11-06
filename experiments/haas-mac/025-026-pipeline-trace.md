# 025/026: Observe startup and Metal pipeline reuse

This is a diagnostic checkpoint from speech.cpp
`114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It is not an optimization candidate.
No build, worker invocation, model inference, GPU operation or performance
measurement is part of this checkpoint. No production policy is changed.

## Existing behavior

The worker already loads one model, emits `ready`, and retains the model for
later requests. The server also retains its models and replaces one only after
its requests finish. Both entry points enable standard warmup unless given
`--no-warmup`. FastConformer's warmup uses one second of silence, encodes it, and
decodes on the device. A quantized decoder also warms its lazy CPU copy. An
explicit CPU request on an F16 model can still prepare that copy later.

The primary campaign protocol uses `--no-warmup`, then warms every evaluation
input before timing requests. Model residency and the work that this already
warms are not new warm-latency opportunities. The remaining question for 025 is
the first visit to an unprepared pipeline/shape class and its trade with startup.
Ready without standard warmup, ready with warmup, the first request after ready,
and a repeated request are separate observations. Process launch to ready is not
a pure C API model-load timer.

The pinned Metal backend already retains a process-local pipeline map in
`ggml_metal_library`, guarded by its existing `NSLock`. A get misses the map, and
the compile entry checks the map again under the same lock before creating a
function and a pipeline state. There is no eviction until the owning library is
freed. The static Metal device owners keep their libraries until process exit.
The context's `pipelines_ext` is allocated and freed but has no other use at this
pin. The pinned Metal source contains no `MTLBinaryArchive`, `newBinaryArchive`,
`binaryArchives`, or `serializeToURL` path.

The campaign anchor builds with `GGML_METAL_EMBED_LIBRARY=ON`: this embeds source,
not ready-made pipeline states. `ggml_metal_library_compile_all` creates the
per-kind libraries in parallel with `newLibraryWithSource`. Functions with
specialization constants and compute pipeline states are created separately on
demand. A Metal API call does not establish that Apple's internal compiler cache
missed. The backend also supports precompiled library URLs; this diagnostic
driver deliberately requires the source-embedded campaign configuration.

The 180-second setting belongs to Metal memory residency, not to the pipeline
map. On supported SDK/runtime combinations the residency heartbeat requests
resident allocations after work; a graph refreshes its countdown. A nonpositive
`GGML_METAL_RESIDENCY_KEEP_ALIVE_S` falls back to 180 seconds, while the presence
of `GGML_METAL_NO_RESIDENCY` disables residency, even with value `0`. Existing
007 logs show `keep_alive = 180 s` and `use_residency = true` on the M2 Ultra.
They show activation, not idle-transition latency, residency duration, power, or
memory-pressure behavior. This checkpoint does not run an idle experiment or
change either setting.

The relevant source owners are `tools/worker/worker.cpp`,
`tools/server/served-models.cpp`, `src/engines/fastconformer-engine.cpp`,
`src/api/speech.cpp`, `ggml/src/ggml-metal/ggml-metal-device.{cpp,m}`, and
`ggml/src/ggml-metal/ggml-metal-context.m`. Existing 007 normal-encoder logs are
only an encoder-first-visit observation: the three input boundaries added 15,
2, then 0 pipeline states for each model type, after two setup states. They do
not establish the compile count of a warm worker request or its decoder.

## Instrumentation

`GGML_METAL_PIPELINE_TRACE` is strict `0|1`, read during library construction.
Unset and `0` allocate no trace state and write no trace records. Invalid values
fail. This independent branch leaves kernels, fusion decisions, graph order,
command-buffer count, encoding parallelism, and synchronization unchanged.

The trace keeps cumulative counters inside the original library lock:

| Counter | Meaning |
|---|---|
| `lookups`, `hits`, `misses` | Calls to the existing get wrapper, not individual hash-table probes |
| `compile_entries`, `recheck_hits`, `compile_misses` | Calls to the existing compile wrapper and its second lookup |
| `new_function_calls` | Calls to either form of `newFunctionWithName` |
| `new_compute_pipeline_state_calls` | Calls to `newComputePipelineStateWithFunction` |
| Failure/success/insertion counters | Returned objects, compatibility rejection, and map insertion |

Only integer increments and a bounded copy of an existing name are added under
that lock. There is no per-hit log or hot-lock clock. Each library stores at most
128 miss names of at most 255 bytes; the aggregate count continues past that
limit. Names are emitted once. A unique process-local library ID distinguishes
temporary probe libraries from a later owner at a reused address.

Snapshots copy the counters under the lock and serialize JSON after unlocking.
They occur before and after graph encoding and at library destruction. The end
snapshot follows the existing `dispatch_apply`; it does not wait for the GPU.
The normal command-buffer setting remains `n_cb=1` and the analyzer rejects a
different value for this campaign. `GGML_METAL_FUSION_DEBUG`, graph debug,
capture, and other inherited `GGML_*`/`SPEECH_*` settings are rejected by the
driver. No fusion-stat initialization is used. Snapshots and their final totals
also expose counts outside graph encodings rather than attributing them silently.

Source-library creation is separate. Each parallel task records its own
`newLibraryWithSource` API duration, outside the pipeline lock. One record after
the group joins gives the total parallel wall duration and the sum of these API
durations as different fields. They overlap and must not be added. The existing
single-source device-probe path is also counted. These are local monotonic
durations; no C++ absolute clock is compared with Python's clock.

This tracing has overhead: graph-boundary snapshots, JSON serialization, stderr
writes, and counter updates. Trace-OFF and trace-ON must be compared with one
binary. The OFF build must also be checked against the unmodified anchor before
interpreting its timings. No traced latency is a speed-up result.

## Reproducible driver and analysis

`pipeline-probe/run.py` runs one fresh worker process. It reexecutes under
`/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock` with `/usr/bin/python3`.
It requires a source-verified build, records binary/model/input identities, and
freezes itself and `pipeline_trace.py` in the new output directory. Model and
binary hashes are checked only when that later run is authorized. Raw artifacts
stay outside Git under `outputs/`.

Each worker uses four threads, explicit CPU decoding and the same nine campaign
clips at indices `0,4,9,10,14,19,20,24,29`. It first visits every clip once in
longest-first order, then visits every clip again with the campaign's round
rotation. Thus every timed repeat has been warmed. A separate final phase asks
for public tokens and segments with `timestamps=true`; the first/repeat phases
use `timestamps=false`. The first pass is not a claim that all nine requests are
cold: pipeline classes can be shared across files.

The request timer starts after PCM reading/hash verification and before
base64/JSON/chunk writes, and ends after validated terminal queue receipt.
Python's launch/ready/request markers are written outside request timers.
Markers bound these serialized operations in the log; they do not provide exact
cross-process clock causality or CPU/GPU timing attribution. Launch-to-result
includes the earlier work and diagnostic markers. A fresh process does not mean
the filesystem or Apple's opaque caches are cold; neither is purged.

The analyzer rejects incomplete graphs, changed `n_cb`, counter inconsistencies,
out-of-order or missing requests, missing library-final snapshots, interleaved
records, and model-public-output differences. It preserves source-library wall
and summed times separately. `--compare OFF ON` pairs the same files and round
with one binary, requiring exact text, tokens, segments and stop reason, and
reports diagnostic overhead. No rounding tolerance is introduced.

The small `check.py` fixture validates this parser and pairing, including
timestamp changes, malformed records, and 130 misses with only 128 saved names.
It does not execute ggml and is not evidence that the new Objective-C code builds
or that real model output is unchanged.

## Execution plan after a slot is authorized

1. Build the trace branch and the unmodified anchor with the same recorded
   Release/native/Metal flags under the shared lock, using `-j4`. Keep source
   commits, effective flags, binary hashes and the driver snapshot fixed.
2. Verify unset and `0` produce no trace lines, `1` produces complete records,
   and an invalid trace value fails. Check the ordinary output against the
   anchor. Trace-ON public output must equal trace-OFF exactly. GPU/stage and
   unavailable platform checks retain their own statuses.
3. Start with Q4_K and three alternating OFF/ON rounds, separately for standard
   warmup and `--no-warmup`. Use the fixed nine inputs and CPU decoding with four
   threads. Record ready time, first-pass and repeated request times, and peak
   footprint. Repeat other model types only if the counters show a relevant
   difference or the coordinator authorizes it.
4. Inspect per-request counts and names. A repeated phase with no observed
   compile miss/API call and no unassigned compile count has no measured warm
   compile work for a disk cache to remove. A first-visit miss motivates a
   separately reviewed startup/warmup trade, not a warm-speed claim. If trace
   overhead or correctness is unacceptable, fix the diagnosis before proposing
   an optimization. No disk cache or speculative warmup expansion is included.

Worker command examples, not executed at this checkpoint:

```sh
/usr/bin/python3 -B experiments/haas-mac/pipeline-probe/run.py \
  --build /path/to/source-verified/build.json --assets /path/to/assets.json \
  --out /path/to/outputs/probe-none-off-r0 --warmup none --trace 0 --round 0
/usr/bin/python3 -B experiments/haas-mac/pipeline-probe/run.py \
  --build /path/to/source-verified/build.json --assets /path/to/assets.json \
  --out /path/to/outputs/probe-none-on-r0 --warmup none --trace 1 --round 0
/usr/bin/python3 -B experiments/haas-mac/pipeline-probe/pipeline_trace.py --compare \
  /path/to/outputs/probe-none-off-r0 /path/to/outputs/probe-none-on-r0
```

Only Python AST parsing, synthetic parser fixtures, source inspection and
`git diff --check` have been run. The checkpoint has no measured performance
conclusion. Windows/Vulkan, M5 and idle-residency behavior are not validated.
