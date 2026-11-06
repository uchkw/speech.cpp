# Retain the last encoder graph for an exact frame shape

This is an unbuilt, unmeasured source prototype on `codex/exp-haas-012`, based on
`114c514430c8aa1455c077819f8e72e674a48b75`, with ggml pinned to
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It includes no other trial's runtime changes.

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
does not change either instance. It is designed to check the following after execution is authorized:

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
keywords and timestamps. Candidate-off still needs comparison with an unchanged baseline binary.
The evaluation record must retain the model SHA-256 and each supplied audio input's SHA-256 beside the command and
raw check log, including the fixture selected for cancellation. This source checkpoint supplies no executed result.

Example commands, not executed at this checkpoint:

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

All builds and runs remain deferred, serialized under `/private/tmp/speech-haas-mac-perf.lock` when authorized,
with builds bounded to four jobs. Official F32 dumps and Windows Vulkan are still unverified. No speed, correctness,
memory or adoption result is established by source preparation.
