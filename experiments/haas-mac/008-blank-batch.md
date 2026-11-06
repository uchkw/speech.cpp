# Reject four-frame blank joint batching after the latency screen

On 2026-11-07, this fixed-four-row prototype did not improve warm latency on the M2 Ultra. The same-binary screen
moved in the slower direction for Q4_K, Q8_0 and F16. No full-corpus or short-prefix evaluation follows this screen.
This rejects the measured prototype for lack of a speed benefit; its small internal rounding differences are
recorded separately and are not treated as observed recognition errors.

## Frozen implementation and conditions

- Runtime source: `17de317264b9fdc197cc8f2569695742e2d986c6`, independently based on campaign `114c514`.
- ggml: `e1a9ac12edbb57402618fd6936598603f5ee6266`, with no other trial's changes.
- Frozen runner: `b899922d5f71c5f66be33e0add13f1d310b5f74b`; Python 3.9.6 via `/usr/bin/python3`.
- Hardware: Apple M2 Ultra, 24 CPU cores, 76 GPU cores and 192 GB memory. Metal encodes; CPU decoding explicitly uses
  four threads. Four is the campaign setting, not the runtime's automatic default.
- Build: Release, `GGML_NATIVE=ON`, `GGML_METAL=ON`, bounded to four compile jobs. Build and inference use the shared
  campaign lock. The build enables dotprod and i8mm.
- Primary comparison: one verified binary with `SPEECH_HAAS_TDT_BLANK_BATCH=0` versus `=1`. Both unset and `0` retain
  scalar blank joints; `1` uses blocks of up to four rows, including shorter tails. No defaults change.
- Screen: nine existing duration-stratified recordings over three alternating rounds, plus nine separate requests
  with timestamps for public-quality comparison, for each precision and side. The fixed anchor is also rerun.

The fixed anchor's SDK cache setting is implicit; this build records the explicit Xcode SDK path inherited through
`SDKROOT`. This distinction remains a limitation of the separate anchor comparison. Primary off/on uses one binary.

## Internal rounding and output comparisons

The dedicated check runs four prediction histories with batches one through four, on CPU1, CPU4 and Metal, using a
synthetic F32 fixture and each of the three real model files. It completes 1,884 assertions. The only 68 failures
are strict logit bit comparisons; all 192 direct label/duration argmax comparisons agree.

| Weights and backend | Strict differences per 16 direct cases | Different batch sizes | Largest printed absolute difference | Lowest printed SNR |
|---|---:|---|---:|---:|
| Q4_K and Q8_0, CPU1 and CPU4 | 8 each | 2, 4 | 1.14e-5 | 141.5 dB |
| Q4_K and Q8_0, Metal | 12 each | 2, 3, 4 | 1.53e-5 | 141.3 dB |
| F16, CPU1 and CPU4 | 0 each | None | 0 | Exact |
| F16, Metal | 12 | 2, 3, 4 | 1.53e-5 | 141.7 dB |

Synthetic F32 has no direct mismatch on any of the three configurations. Its controlled cases cover blank duration
zero, skipped rows, tails, nonblank invalidation, forced max-symbol advance, cancellation and keyword boosting.
All 486 dedicated TDT pairs agree on emitted IDs, frames, durations and callbacks. The generated real-model inputs
rarely take a blank batch, so they do not replace real-audio coverage. Intermediate logits for every blank in a
complete decoding are not recorded. The original strict failures, exit statuses and thresholds remain unchanged.

A separate driver calls ordinary `Recognizer::recognize()` with the switch fixed for each process, Metal encoding
and CPU4 decoding. All nine pairs from the existing 7.98, 14.52 and 24.72 second stage recordings match raw token IDs,
frame indices, duration decisions and text. This driver compares paths and treats graph counts as metrics.

The worker screen matches all 81 off/on latency texts and all 27 public-quality outputs, including token/segment
texts and times. The independent anchor comparisons also match. Literal and repository CER are unchanged within
this nine-input screen; this is not a full-corpus accuracy result.

Separate direct-joint runs with `GGML_METAL_GRAPH_DEBUG=2` record actual pipeline creation and graph execution:

| Joint output weights | Batch 1 | Batch 2 | Batches 3 and 4 |
|---|---|---|---|
| Q8_0, in both Q4_K and Q8_0 files | `kernel_mul_mv_q8_0_f32` | `kernel_mul_mv_mma_q8_0_f32_nt2_rt1` | Same few-row MMA |
| F16 | `kernel_mul_mv_f16_f32_4` | `kernel_mul_mv_ext_f16_f32_r1_2` | `kernel_mul_mv_mma_f16_f32_nt2_rt1` |

These observed direct-graph logs are distinct from the source analysis and run outside timing. CPU source conditions
select the Q8_0 two-row i8mm implementation for even batches and the one-row path for odd batches, consistent with
the observed batch-two/four differences; CPU kernel dispatch is not instrumented. No broad Metal matrix-multiply
half-staging explanation is substituted for the observed few-row kernels.

## Latency and speculative work

Positive paired change means slower. It is the median of per-input median latency ratios, not the percentage change
in p50. Each of the nine per-round paired medians also moves in the slower direction. These bounded samples show
no gain; they do not establish a precise population slowdown.

| Precision | Paired latency change | p50 off / on | RTFx off / on | Fixed-anchor paired change |
|---|---:|---:|---:|---:|
| Q4_K | +0.584% | 77.29 / 77.74 ms | 177.77 / 176.77 | +0.425% |
| Q8_0 | +0.829% | 74.03 / 75.59 ms | 185.07 / 182.50 | +0.903% |
| F16 | +0.613% | 72.26 / 72.71 ms | 188.19 / 186.89 | +0.267% |

The ordinary three-recording path diagnostic shows little reuse of the computed future rows:

| Precision | Total graphs off / on | Future rows used / computed | Unused computed rows |
|---|---:|---:|---:|
| Q4_K | 186 / 184 | 46 / 174 | 73.6% |
| Q8_0 | 187 / 185 | 48 / 182 | 73.6% |
| F16 | 187 / 185 | 47 / 177 | 73.4% |

The candidate computes substantially more joint rows while saving only two graphs per precision on these inputs.
Those counters describe the speculative work; the independent screen supplies the latency comparison. F16 executes
the changed algorithm too and is not an unchanged runtime control. Load, first request and process peak memory are
recorded separately in the artifact report; no memory improvement or GPU VRAM result is claimed.

## Evidence and remaining limits

Ignored artifacts are under `outputs/haas-mac-campaign/008/`: `evaluation.html` and `evaluation.json`, the frozen
runner's `screen/summary.json`, raw worker and stage logs, `audio-paths/result.json`, `dispatch/result.json`, the
earlier `check-report.html`, and `evidence-evaluation.json`. The earlier 40 evidence hashes remain intact; the later
manifest adds the exploratory continuation. The build manifest is in `builds/008-blank-batch/` beside that directory.

Official NeMo/FLA F32 dumps, Windows Vulkan, the 650-input full corpus and short-utterance quality remain unverified.
The screen supports rejection of this variant, not adoption or a general recognition-quality guarantee.
