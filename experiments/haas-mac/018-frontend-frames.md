# Parallelize the whole-input NeMo frontend by frame ranges

This source-only prototype starts at speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The implementation and checks are prepared but have not been compiled,
executed on model inputs or timed.
The experiment has its own branch, `codex/exp-haas-018`; it does not include 017 or another prototype.

## Existing computation and ownership

`src/common/nemo-frontend.cpp:89` validates the input, pre-emphasizes it into one float32 buffer, allocates the complete
float64 mel output, and calls `log_mel` for all frames. Each `log_mel` call owns its windowed samples, spectrum and
power scratch (`:59`). Each frame reads the same immutable window/filterbank and writes one output row. Its FFT,
mel summation and logarithm do not depend on another frame. `RealFft::transform` (`src/common/fft.cpp:69`) reads its
constructor-created tables and uses local scratch; it does not mutate the FFT object.

Per-feature normalization (`nemo-frontend.cpp:104`) reduces all frames in a fixed order. Keep it sequential, including
its mean, unbiased variance, guard and float32 conversion. Keep pre-emphasis sequential as well. Parallelize only the
whole-input frame ranges; do not change frame arithmetic, zero padding, FFT butterflies, mel reductions or the output
layout.

## Experiment boundary

- Read `SPEECH_HAAS_FRONTEND_PARALLEL=0|1` at KDA FastConformer recognizer construction, default off; reject other
  values in that scope. Do not read it in Sortformer or non-KDA FastConformer.
- Keep the existing `NemoFrontend` constructor and delegate it to an explicit worker-count overload with one worker.
  The experiment passes the existing model thread preference, capped at four, only when enabled. This adds no C API
  option and changes no FFT interface.
- In `log_mels`, select `max(1, min(thread_limit, 4, frames / 128))` workers. The 128-frame minimum per worker is an
  unmeasured initial grain size, not an accepted performance threshold. The off path always has one worker.
- Give worker `i` the contiguous interval `[frames*i/workers, frames*(i+1)/workers)`. Call the existing `log_mel` with
  the complete pre-emphasized buffer, its original total length, that frame interval and the corresponding output
  offset. Each call owns scratch and writes disjoint complete rows.
- Start `workers - 1` tasks with `std::async(std::launch::async, ...)`; compute one range on the caller. Keep all
  futures alive until their tasks finish and call `get()` to propagate worker failures. A launch or allocation failure
  must not leave work running against destroyed arrays or silently retry serially. Convert a launch failure into the
  appropriate explicit library error, as other host-thread code does.
- Do not add a persistent pool, mutable shared scratch or nested frame parallelism. Preserve cancellation observation
  points outside the frontend. This attempt measures whether the smallest range split is sufficient.

For a few seconds of audio, task creation and joining can cost as much as the work saved. The candidate grain size
therefore keeps small inputs serial, and its transition points must be measured explicitly. Existing campaign stage
records put the whole frontend around 2 to 6 ms for their three 6 to 28 s inputs; only part of that time is eligible
because pre-emphasis and normalization remain serial. A reduction in frontend time alone is not an end-to-end gain.

## Sortformer and other call sites

Sortformer's constructor (`src/families/sortformer/diarizer.cpp:27`) retains the one-worker default. Its streaming path
calls `log_mel` directly for ready frames (`:70`), including a nonzero `x_begin_` after old samples are removed. Leave
that method and its existing buffer contract untouched. Its whole-input `features()` call also remains serial by
default. A check may explicitly select parallel whole-input computation to verify that the common code still agrees
with incremental computation, without enabling it for Sortformer in the runtime.

No frame range may assume that every audio input is an exact multiple of the hop. The existing minimum is two frames
when normalized and one otherwise. Preserve the exception category, message and `audio` input attribution.

## Prepared validation, not yet run

The check target loads only the frontend window and filterbank, with the complete model layout still validated:

```sh
nemo-frontend-parallel-check fastconformer "$MODEL" "$REFERENCE_OUT"
nemo-frontend-parallel-check sortformer "$SORTFORMER_MODEL" "$SORTFORMER_REFERENCE_OUT"
```

Omitting the reference root runs differential cases and explicitly reports the official comparison as unchecked.
Both modes execute CPU work and must wait for the shared performance lock and a granted compute slot.

1. Compare the original serial output with worker limits one, two and four. Require bit identity for all raw float64
   log-mel values and final float32 features, not only SNR or final text. Include silence, an impulse at both ends,
   low-amplitude alternating samples, deterministic noise and real speech.
2. Cover minimum-length rejection, the minimum valid input, hop boundaries, frame-grain transitions and one sample on
   each side, lengths not divisible by the number of workers, and the 2, 4 and 30 s workload. Verify every output row is
   written and remains finite. Repeat calls of different lengths on the same frontend, and compare independent
   concurrent calls that share its immutable FFT/tables to isolated calls.
3. Run the existing official Haas frontend check on the 11 CPU reference inputs with its unchanged 90 dB bound.
   Compare parallel and serial output exactly first. The official full KDA/TDT reference remains outside this check.
4. Preserve Sortformer's whole-input versus incremental checks with pieces of up to 1, 333 and 4,800 samples. Include
   a streaming check that discards old samples, exercising nonzero `x_begin_`. Keep the existing official 90 dB bound
   where Sortformer weights and dumps are available; unavailable assets are not a pass.
5. Check switch parsing and the unchanged non-KDA/Sortformer default. Before timing, compare candidate-off with the
   unchanged base, then off/on worker results for Q4_K, Q8_0 and F16, plain and keyword requests, preserving all exposed
   text and times. Frontend code stays on the host for both CPU and Metal model execution.

Timing waits for a compute lease and the shared performance lock. Count task creation, joining and normalization in
the frontend timer. Hold model/decoder thread counts fixed, separate first-use and warm requests, and alternate the
same binary off/on. Report the short-input overhead, grain-size transitions, frontend duration, total request latency
and peak memory. Any future threshold adjustment requires a fresh paired measurement. CPU/Metal/Windows validation
and a measured acceptance verdict are still outstanding.
