# Reduce TDT host copies and allocations

This source-only prototype starts at speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It has not been compiled, executed or timed. The switch
`SPEECH_HAAS_TDT_BUFFER_REUSE=0|1` defaults to off and only selects the KDA TDT path. Other prediction networks
do not read this switch. An invalid value on KDA fails explicitly.

## Changed ownership boundaries

| Location | Original operation | Enabled operation |
| --- | --- | --- |
| TDT frame selection | Allocate and copy `joint.hidden()` floats into a vector before each prediction/joint evaluation. | Pass pointer and count into Graph's owned upload. Haas has 640 floats, or 2,560 bytes, per frame. |
| Graph float `input` and `set` | Check the element count and copy/convert into `Upload::bytes` immediately. | Keep the same checks and immediate ownership. Vector signatures remain and delegate to the pointer/count overloads. |
| Projected prediction readback | Allocate a new vector each outer TDT iteration. | Read all values into one request-local vector, retaining its capacity. |
| Logits readback | Allocate each outer iteration and again for each blank-joint result. | Read all token, blank and duration logits into one request-local vector, retaining its capacity. |

The off path retains its temporary frame vectors and fresh prediction/logits results. Every graph input still owns
its bytes before `compute`; no pointer survives an `input` or `set` call. The float input validation, F16 conversion,
and float32-only readback rules remain unchanged. Integer inputs are untouched.

The prediction state still travels through the same host readback and upload. The encoder projection still returns
to the host, as CPU decoding needs it. The blank-joint graph is still rebuilt and computed at the same point. Full
logits are still read, the first of equal maxima still wins, and the existing separate boosted-token vector is still
overwritten for every selection. Progress/cancellation, blank duration zero, maximum symbols, graph counts and the
token/duration update order remain unchanged. This does not combine resident state, cached joint graphs, device
argmax, command reuse, or shared plain/boosted prefixes.

## Prepared checks, not yet run

`fastconformer-tdt-buffer-check` has three modes:

```sh
fastconformer-tdt-buffer-check --graph-only cpu 4
fastconformer-tdt-buffer-check --synthetic "$ARTIFACTS/synthetic" cpu 4
fastconformer-tdt-buffer-check "$MODEL" cpu 4 "$DUMP/joint_encoded.npy"
```

- The graph-only cases compare both public float overloads, F32 and F16 conversion, source mutation before compute,
  source expiration after set, and the same exception category/message for wrong counts, types and foreign inputs.
- The step cases compare complete KDA states, projected predictions and every joint logit bit for bit over 16 changing
  labels and frames. Reusable predictions and logits are deliberately overwritten after each iteration, including
  token boosting and poisoned blank/duration entries, so the next read must replace every value. Their data pointers
  must remain stable after the first allocation.
- The deterministic synthetic F32 fixture and controlled frames cover plain/boosted paths, empty input, zero duration,
  blank chains, max symbols, cancellation before and during decoding, a final token extending past the input, and a
  plain request after a boosted or cancelled one. The same RNN-T greedy fixture retains its expected tokens/frames.
- Each optional real projection compares off/on plain and boosted raw token IDs, frame indices, durations, graph
  counts and exact progress sequences, with cancellation at zero and two callbacks and with no cancellation.

The fixture files are copied from the existing 004a differential check; they are not official checkpoint weights or
official KDA/TDT dumps. The optional `joint_encoded.npy` files may come from the pinned official CPU encoder/joint
projection. Decoding them with two C++ paths remains a differential decoder check, not an official decoder check.

## Slot-dependent validation and timing

All commands that build or execute model work must hold `/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock`.
Build only when the coordinator grants the slot, with `-j4`. Build targets are `fastconformer-tdt-buffer-check`,
`speech-cli`, and `fastconformer-timing`.

1. Run graph-only and synthetic F32 cases on CPU4 and Metal. Run the real-weight step/decoder cases on Q4_K, Q8_0
   and F16 on both CPU4 and Metal, including the 11 existing official encoder projections.
2. Compare unchanged-base and candidate-off worker outputs before comparing candidate off/on. Check plain requests
   and Haas keywords with guard enabled and disabled. Check Metal encoder plus CPU4 decoder as the primary workload,
   and explicit device decoding separately. Keep text/public token/segment times exact wherever exposed; keyword
   requests do not expose timestamps, so their raw decisions are covered by the internal differential check.
3. Run the campaign's alternating screen first with the same verified binary and explicit switch values. Separate
   cold/load, first request and warm request. Only a gain beyond round variation proceeds to full quality/latency runs.
   Recheck the fixed anchor and peak memory. Do not claim an improvement from fewer allocations alone.
4. Existing non-Haas full-stage checks and Windows Vulkan remain required where their weights/dumps/device are
   available; the shared Graph differential cases alone do not establish full-family or Windows compatibility.

The official full KDA/TDT reference is unavailable in this environment and stays unverified. No bound is relaxed.
No speed-up, acceptance verdict or measured HTML result is claimed by this preparation.
