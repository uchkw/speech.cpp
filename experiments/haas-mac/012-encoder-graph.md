# Retain the last encoder graph for an exact frame shape

This experiment has been evaluated and is not adopted. The supplied aggregate verdict for measure `012` is `rejected`: Representative requests and the artificial same-length condition do not establish stable latency improvement; Q8_0 and F16 paired latency regress in this screen. The evaluated candidate is `a4819be8c5196c7a16bb7bf247d3d59dd6272d44`, with ggml pinned to `e1a9ac12edbb57402618fd6936598603f5ee6266`, and it includes no other trial's runtime changes. The supplied figures, checks, notes and limitations are summarized in the Evaluation section below.

## Scope and ownership

`SPEECH_HAAS_ENCODER_GRAPH_CACHE=1` enables one graph slot in each loaded Haas-ja-v1.0 KDA recognizer.
Unset and `0` keep the ordinary path; other values are errors. Other model identities keep the ordinary path.
The slot belongs to one model instance and its fixed encoder backend. Its key is the exact number of frontend
feature frames, not the number of subsampled encoder frames. The model fixes the mel count and feature type.
Two recordings with equal frame counts can reuse the graph even when every feature value differs.

On a miss, the old graph is destroyed before building and allocating a graph for the new shape. On a hit,
`Graph::set()` copies the new feature values into the graph's owned upload bytes, then `compute_again()` uploads
the inputs and executes it. The retained graph keeps all constant input uploads as well, including the position
input for a shape outside the existing position cache. Only the final `joint.enc` result is marked as an output.
Readback returns a new owned vector; no request borrows the graph's output buffer.

The existing `Recognizer::allocr_` already outlives individual requests and is used only by `encode()`.
The prototype keeps it, including its existing work-buffer reuse across different shapes. Its gain, if any, is
from avoiding graph construction and `ggml_gallocr_alloc_graph()` on a hit. It does not introduce allocator-object
reuse, cache a CPU backend execution plan, retain Metal command encoding, or cache encoder results.

The recognizer destructor explicitly destroys the retained graph before freeing `allocr_`; model weights and the
encoder's position cache outlive both. A failed cached encode clears the graph slot and propagates the exception.
The allocator's work buffer can remain allocated, as it does in the ordinary path. A shape change is the defined
one-slot replacement policy, not a device fallback or a padded-shape computation.

Decoder states, decoding device choices, keyword trees and request callbacks are outside the slot. The engine still
checks progress after the frontend and after the encoder, then reports decoder progress. An encoder call has no
new mid-graph cancellation point. A request cancelled during decoding leaves a completed encoder graph reusable;
the next decoding starts with its ordinary fresh recurrent state. Request reset does not reset the model's cache.
One model still serves one request at a time; this does not add concurrent use of a recognizer.

## Differential check

`fastconformer-encoder-graph-check` invokes the actual `Recognizer::encode()` method in two independent recognizers
over the same backend, with the switch captured as `0` and `1` when they load. Changing the process switch later
does not change either instance. It is designed to check the following:

- Exact finite final `joint.enc` bytes for fresh graphs, cache hits and evictions, without intermediate output marks.
- Different feature values at the same frame count, and feature frames 129 and 130 that share a subsampled shape
  but must have different graph keys. Cases cross the existing position-cache boundary and return from 3,000
  feature frames to a short request.
- Build/hit counters, the retained feature shape, and the compute-buffer allocation. The compute bytes describe
  the recognizer's allocator even when no graph is retained; they exclude weights, position caches and host metadata.
- Invalid input sizes on a hit and a miss, slot eviction on the exception, unchanged completed-call counters and
  successful retry. Backend allocation or execution failure injection is not part of this check.
- Exact emitted TDT IDs, frames, durations, graph counts and text, fresh requests after cancellation and boosting,
  and changing the decoder device without changing the encoder slot. Before-first cancellation must actually
  return false on callback one and compute no graph. For after-step cancellation, the check first finds supplied
  real audio whose complete decode reaches at least three callbacks. It records every candidate's reachability
  and the selected fixture, then requires both sides to return false at exactly callback `allowed + 1` for
  `allowed=1` and `2`, after at least that many computed graphs. Each cancelled run is followed by a same-shape
  input update and a complete ordinary decode that must recover the pre-cancellation result.
- Destruction of a model with a live cached graph, continued use of the independent ordinary model, and creation
  of a new empty cache whose outputs match the destroyed model's outputs.
- Real mono F32 `.npy` audio supplied explicitly on the command line, at the model sample rate, with direct encoder
  comparisons and a full `Recognizer::recognize()` replay. If no supplied input reaches the after-step cancellation
  condition, including a run without audio arguments, that coverage remains unverified and the check fails.
  These deliberate same-shape replays are numerical checks, not latency evidence.

Every finite-value or bitwise mismatch fails; equal text does not erase a tensor difference. The synthetic cases
use generated features and the supplied model's actual weights. They are not official F32 reference tensors or
recognition-quality samples. Direct boosting checks do not cover the C API's keyword parsing and guard behavior.
The existing recognition API and worker smoke checks must separately cover request reset, cancellation, callbacks,
keywords and timestamps. Candidate-off requires comparison with an unchanged baseline binary.
The evaluation record must retain the model SHA-256 and each supplied audio input's SHA-256 beside the command and
raw check log, including the fixture selected for cancellation. The executed results are recorded in the evaluation record.

Example commands:

```sh
./build/fastconformer-encoder-graph-check "$MODEL" cpu 4 "$SHORT_NPY" "$MEDIUM_NPY" "$LONG_NPY"
./build/fastconformer-encoder-graph-check "$MODEL" gpu 4 "$SHORT_NPY" "$MEDIUM_NPY" "$LONG_NPY"
```

## Request-level evaluation and diagnostic counters

The existing `measure/fastconformer-timing.cpp` builds its encoder graph directly. It does not call
`Recognizer::encode()` and cannot measure this cache. Measure actual worker requests with the frozen common
campaign runner, with both sides' switch values recorded. Do not report the unchanged stage timer as a cache gain.

`SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE=1` prints one cumulative diagnostic line to stderr when the recognizer is
destroyed. It defaults to `0` and rejects values other than `0` and `1`. No per-request logging, clocks or hashes are
added. Builds and hits count successfully completed encoder calls; failed encodes are not counted. Disabled mode
reports every completed encoder call as a build. The worker must close stdin and exit normally to emit the summary;
a forced termination can omit it and is not evidence of zero calls.

Run that trace separately from latency and quality measurement, using the same real worker inputs, ordering and
warmup policy, and retain stderr plus the request manifest. The cumulative counts include warmups. With
`--no-warmup`, no hidden engine warmup is included, but the runner's explicit warmups still count. Ensure their count
plus measured requests equals builds plus hits before treating the trace as complete. A valid low-impact trace
does not by itself provide per-request timing or final encoder bytes.

The campaign runner's allowlist must record both experiment variables before execution; this source checkpoint
does not change the coordinator-owned runner. Keep representative corpus order intact: different lengths cycle
through it, so a last-one exact-shape cache may have few hits. Report that actual hit rate separately from a
deliberately repeated same-shape workload. Miss overhead, retained memory, first request, normal ready warmup and
warm latency all count toward the decision. Do not select only hits to infer a representative gain.

The executed checks and the remaining limits are recorded in the evaluation record; see the Evaluation section below.

## Evaluation

Rendered from the supplied aggregate record for measure `012` (candidate `a4819be8c5196c7a16bb7bf247d3d59dd6272d44`, ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`, runner `a7e227c2b9067ee0b73a4582acf2e453cff5f179`). Verdict: `rejected`: Representative requests and the artificial same-length condition do not establish stable latency improvement; Q8_0 and F16 paired latency regress in this screen. The experiment is evaluated and not adopted.

- q4_k: paired reduction 0.15539188387312386% (positive means faster; not the p50 change), anchor reduction 0.8318893424149709%, p50 81.36883299999997 to 81.18449999999999 ms, p90 132.65430020000002 to 128.86721619999986 ms, p99 151.28747971999985 to 149.09804691999997 ms, RTFx 169.21319258296054 to 169.69162127343048, load 151.797125 to 153.54412500000004 ms, first request 169.92108300000004 to 169.812708 ms, peak memory 453198856 to 460358712 bytes, per-round reductions: -0.02737430328088042%, 0.3045262484502653%, -1.001362123122962%.
- q8_0: paired reduction -2.385233771753059% (positive means faster; not the p50 change), anchor reduction -1.0287924940103776%, p50 79.54537499999992 to 79.81633300000013 ms, p90 124.01357479999993 to 128.42674999999994 ms, p99 144.93494528000002 to 149.59475 ms, RTFx 176.47380357896859 to 172.44207045750832, load 197.20925 to 200.31895800000004 ms, first request 167.50062499999996 to 168.251834 ms, peak memory 763282656 to 757810472 bytes, per-round reductions: -3.649501633127694%, 0.06600455229656133%, -3.9105228510377232%.
- f16: paired reduction -0.9460412888069092% (positive means faster; not the p50 change), anchor reduction -0.9784507353341798%, p50 79.81258300000005 to 76.68804200000024 ms, p90 125.69817519999982 to 126.89986719999985 ms, p99 147.3063047199997 to 148.70113671999974 ms, RTFx 174.62653294067096 to 175.23039978648103, load 284.209125 to 292.57991699999997 ms, first request 169.51804199999998 to 171.54512499999996 ms, peak memory 1346684632 to 1359169240 bytes, per-round reductions: 1.4613050434546304%, 0.8372217633015877%, -3.5339260351445523%.

Checks:

- `Actual encoder graph cache differential checks` (passed): 12 conditions: Q4_K, Q8_0, F16 and F32, each CPU1, CPU4 and Metal. All 1860 assertions pass. Final ordinary joint.enc bytes are finite and bit-identical, maximum absolute difference 0 and zero noise (infinite SNR). Shape hits, changed feature values, 129/130-key separation, long-to-short replacement, exceptions and live-slot destruction/recreation are included.
- `TDT paths, cancellation and recovery` (passed): The dedicated check compares actual recognize paths on three supplied real recordings, IDs, frame/duration decisions, graphs, text and progress. Before-first and after-one/two-step cancellation actually returns false at callback allowed+1 in both paths, followed by same-shape input update and exact recovery. Direct boosting and decoder-device changes preserve the cache contract.
- `Official F32 encoder arithmetic` (passed-separate-scope): 11 official inputs on CPU and Metal: minimum final encoder SNR 104.4 dB and 56.4 dB against unchanged 90 dB and 25 dB limits; maximum absolute differences 0.000187 and 0.0533. The existing direct graph checks features to subsampling/layers/encoded. It does not invoke Recognizer::encode, validate cache reuse against official joint.enc, or validate official KDA/TDT. CPU uses the existing automatic backend thread selection (16 performance cores on this M2 Ultra).
- `Existing C API and worker contracts` (passed): F16 cache off/on: two C API runs and two worker recognition smoke runs pass, including actual cancellation, later request recovery, keywords, guards, reset and timestamps. Worker uses CPU decoder4; API uses automatic load threads. Expected text fixtures come from a frozen original runtime and are differential, not official reference text.
- `Representative screen quality` (passed): Nine public quality inputs per side and type have identical text, token arrays and segment arrays. All 27 latency text comparisons per type match for both same-binary off/on and fixed-anchor comparisons. The screen has three weight types and nine duration-stratified latency inputs over three rounds; no full run was performed.
- `Actual representative worker cache counters` (passed): Separate TRACE1 runs preserve all primary latency input/warmup lists and round rotations. Nine enabled workers make 162 completed encoder calls: 81 explicit warmups plus 81 measured requests; 159 builds and three hits (1.8518518518518516% of the combined calls). Disabled workers make 162 builds and zero hits. Each process exits normally and emits exactly one summary, with builds+hits equal to explicit warmups+requests. These are separate diagnostic executions and exclude quality workers and manual stage timers.
- `Artificial same-length workload` (completed-no-stable-gain): A separate F16 12.84-second input is warmed once then repeated nine times per fresh worker, over three rounds with TRACE0. Median-of-round latency reduction is 0.812225720703%; round reductions are 1.250712697209, 0.812225720703, -0.381169883149%. One separate TRACE1 pair records off 10 builds/0 hits and on one build/nine hits. The single repeated input is an upper-bound condition, not nine distinct quality samples or representative corpus performance.
- `Remaining reference and platform coverage` (not-run): Official cached final joint.enc and complete KDA/TDT references, Windows Vulkan, and the full 650-item quality/30-item latency campaign were not executed. No adoption is claimed.

Notes:

- The same-binary paired metric is a median of per-input candidate/baseline latency ratios after round aggregation. It is distinct from the unpaired p50 change. Q8_0 and F16 paired latency regress in this screen; Q4_K has a small mixed-round signal.
- Each enabled representative trace worker has 18 completed calls (nine warmups plus nine measured requests). All three round-0 workers have one hit; all round-1/2 workers have zero. The cumulative summary does not separate warmup hits from measured-request hits, so the 3/162 ratio is only for their combined calls.
- The recognizer allocator work buffer is 123105792 bytes in both enabled and disabled representative traces. This is the existing encoder allocator, not total process RSS, all retained cache memory, model memory or VRAM. It was already reused before this experiment.
- The enabled slot reports 657408, 1026048 or 1443840 bytes for its final feature input, depending on the final shape. feature_bytes measures only that tensor; graph metadata, constant/position upload storage and allocator overheads are not included. Total retained cache overhead was not isolated.
- The peak-memory values in the profile table are whole-process peak memory footprints from the primary worker runs, reported separately from allocator counters. Their differences cannot be assigned solely to the cache. No standalone GPU-memory measurement was made.
- The manual fastconformer-timing encoder graph bypasses Recognizer::encode. Its raw stage times remain in evidence but are not used to claim a cache gain.
- The artificial repeated-input condition has median latencies 76.461750000000 to 75.862458000000 ms across its 27 repeats per side. It proves the hit path can execute, while its three-round effect is not stable and does not replace the representative result.
- Runtime source and defaults remain unchanged by this evaluation checkpoint. The default-off candidate is not adopted. All new report/script authoring used ZCode GLM-5.3 with effort=max on public notes and fictitious schema only; actual measurements were aggregated and substituted locally.

Limitations:

- No stable speed gain was established for representative requests or the artificial same-length condition. The primary was not repeated or reordered, and no full run followed the rejected screen.
- The fixed anchor uses implicit SDK selection while the new build uses explicit SDKROOT. This is an independent anchor-comparison limitation; the primary uses the same binary on both sides.
- CPU4 is an explicit M2 Ultra campaign setting. Automatic threads and other machines are separate questions.
- Official F32 encoder arithmetic has been checked through the existing direct graph. Official cache-to-final-joint and complete KDA/TDT validation, Windows Vulkan and full-corpus recognition quality remain unverified.
- The p99 of nine duration-stratified inputs is not a production-tail estimate. Fresh-process load and first-request timings do not reset filesystem or Metal caches.
- Cache counters include explicit warmups, and feature/allocator counters cover only their named storage. Complete retained-cache overhead and independent VRAM were not measured.
