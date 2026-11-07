# Double-precision vDSP frontend for Haas

The frozen trial source is `63b8a4e2e4f2895f8d026340c57d58d63732269e`, on the model repository base speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` with ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`.

The strict selector `SPEECH_HAAS_FRONTEND_VDSP` accepts exactly 0 or 1. The FastConformer factory reads it only when `general.name` is `Haas-ja-v1.0` and `fastconformer.decoder.prediction` is `kda`. On an eligible model, unset or 0 uses the ordinary FFT, 1 selects vDSP, and all other present strings are invalid. Eligible selection constructs one vDSP setup at frontend construction, held for the frontend's lifetime.

Only the double FFT changes. The helper packs the windowed frame into real and imaginary halves of N/2 values and runs the double real DFT in place. Unpacking scales the ordinary N/2+1 spectrum by the binary-exact constant 0.5 to undo vDSP's real-forward factor of two, with DC from the real slot and Nyquist from the imaginary slot. The ordinary `RealFft` object remains constructed on both paths, and the other stages keep their existing precision and order. The selected path adds one setup per frontend and a per-call scratch of N doubles, 4,096 bytes at 512 points.

Ordinary-versus-selected comparisons of the raw float64 log-mel and normalized float32 features require finite, equal-shaped values and a signal-to-noise ratio above 90 dB. The official features.npy comparison checks the final normalized features of each path at the same bound, with no official raw log-mel tensor in this check. The restricted repeated and split calls, unset against zero, and shared constructor cases compare bytes.

## Decision and measurement

Keep the default off and reject this candidate for the measured workload. All nine frontend stage comparisons were faster. Whole-request changes were small and each weight type changed direction across rounds.

Apple M2 Ultra with 24 CPU cores, 76 GPU cores and 192 GB. Native arm64 Release, Metal encoder and greedy CPU4 decoder. Fresh workers warmed all nine measured inputs. The same candidate binary ran with selector OFF and ON. Positive paired percentages mean a shorter request. Each input contributes its median over three rounds before the cohort percentiles and paired ratios are computed.

## Nine-input worker screen

| Weight | Paired reduction | Faster | p50 ms OFF / ON | p90 ms OFF / ON | p99 ms OFF / ON | RTFx OFF / ON |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_K | -0.366% | 4/9 | 78.48 / 79.70 | 125.30 / 124.84 | 146.74 / 146.15 | 174.7 / 174.3 |
| Q8_0 | -0.286% | 3/9 | 75.80 / 75.49 | 121.60 / 122.10 | 142.11 / 143.20 | 180.8 / 180.6 |
| F16 | +0.236% | 6/9 | 76.06 / 75.88 | 121.99 / 120.85 | 141.77 / 142.73 | 182.3 / 182.7 |

## Rounds and input lengths

| Weight | Rounds 1 / 2 / 3 | 6–10 / 10–20 / 20–30 seconds | Separate anchor reduction |
| --- | --- | --- | --- |
| Q4_K | -1.52% / -0.12% / +2.63% | +0.12% / -1.42% / +0.36% | +0.508% |
| Q8_0 | -0.12% / +0.37% / -0.22% | -0.53% / +0.41% / -0.29% | +0.469% |
| F16 | +0.06% / -2.92% / +1.48% | -1.05% / +0.24% / +0.70% | +3.788% |

## Frontend stage duration

| Weight | 7.98 s OFF / ON ms | 14.52 s OFF / ON ms | 24.72 s OFF / ON ms |
| --- | --- | --- | --- |
| Q4_K | 2.128 / 1.747 (+17.89%) | 3.597 / 3.171 (+11.86%) | 6.030 / 5.396 (+10.51%) |
| Q8_0 | 1.976 / 1.748 (+11.50%) | 3.599 / 3.263 (+9.31%) | 6.057 / 5.405 (+10.76%) |
| F16 | 1.986 / 1.761 (+11.34%) | 3.606 / 3.182 (+11.78%) | 6.040 / 5.412 (+10.41%) |

## Load and process memory

| Weight | Ready ms OFF / ON | First request ms OFF / ON | Peak MiB OFF / ON |
| --- | --- | --- | --- |
| Q4_K | 144.32 / 142.63 | 169.14 / 165.61 | 430.47 / 431.88 |
| Q8_0 | 204.64 / 186.75 | 161.06 / 161.51 | 725.19 / 721.20 |
| F16 | 277.31 / 277.47 | 160.69 / 161.63 | 1290.08 / 1284.44 |

## Nine-input public recognition quality

| Weight | Literal micro-CER | Repository micro-CER | Timestamp-enabled output matches | Latency text matches |
| --- | --- | --- | --- | --- |
| Q4_K | 40/532 (7.519%) | 22/493 (4.462%) | 9/9 | 27/27 |
| Q8_0 | 39/532 (7.331%) | 21/493 (4.260%) | 9/9 | 27/27 |
| F16 | 39/532 (7.331%) | 21/493 (4.260%) | 9/9 | 27/27 |

## Checks

- The original FFT check passed 27 cases. The vDSP check passed 964 cases across six transform sizes under its existing relative error bound, including DC and Nyquist bins, buffer guards and explicit refusal cases.
- F32, Q4_K, Q8_0 and F16 each passed 163 frontend checks. Ordinary and selected features each exceeded 90 dB for all eleven pinned official inputs, with a minimum of 103.3 dB. The finite differential measurements reached at least 275.7 dB with maximum absolute difference 2.27e-12. The byte comparison applies to the designated constructor, selector, repeat, split and revisited-frame cases.
- On three public recordings, 72 new candidate launches and 24 source-114 controls retained from trial 018 accounted for 288 requests. Unset, OFF and ON matched the complete reference JSON for three weight types on CPU and Metal in plain, timestamp and both keyword-guard modes. The in-scope invalid selector returned invalid_argument.
- Eight new F16 C API and worker checks and four retained original controls passed on CPU and Metal. All six worker runs, including the two retained controls, observed collecting, waiting and running cancellation plus the next successful request. The API used automatic load threads and the worker used four.
- The matched original supplied nine retained joint.enc tensors from three public inputs and three weights. Each of the nine candidate OFF tensors and each of the nine ON tensors was finite and matched its original F32 bytes. Greedy public text also matched. The helper exposes the projected encoder output and final text.
- For each type, all nine timestamp-enabled texts, public tokens and segments and all 27 timestamp-disabled texts matched between OFF, ON and the historical anchor. Every type also matched the nine stage rows for visible text, token count and graph count.

## Interpreting the measurements

- Each type timed nine FLEURS inputs spanning three duration strata, with three inputs in each. The three stage clips last 7.98, 14.52 and 24.72 seconds. Timestamps were off for timing and on for the separate nine-input quality comparison.
- The matched original and candidate used native arm64 Release with explicit SDK and architecture. After normalizing build paths, 73 of 74 common compiler flag files and 15 of 74 common link files matched literally. The entire token comparison accounted for one SDK framework search path, a change in Accelerate link order in 57 files, the frontend factory object in one archive and the vDSP object in a second archive. The remaining architecture, SDK and optimization flags matched.
- The fixed historical timing anchor used source f976ebd8 with a different SDK selection. Its direct comparison uses the same public screen and is shown separately. The source-114 original controls establish semantic continuity only.
- The worker timer starts before the first PCM16 chunk and ends at receipt of the terminal result. Loading, input file reads and warmup are outside it. The stage timer covers the complete host frontend. First request, ready and peak process memory describe separate parts of the run.

## Coverage limits

- The full 30-input latency and 650-input accuracy evaluation and the six 2/4-second prefixes were not run. Percentiles and CER in this report use the nine-input screen only. Each duration stratum has three inputs.
- Official reference comparisons cover the frontend of the pinned encoder path. Full official KDA/TDT, raw decoder IDs, frame indices and durations remain unverified. Real non-Haas and Sortformer runtime, Windows/Vulkan, non-Apple refusal and allocation-fault injection were untested.
- The screen used a Metal encoder and greedy CPU4 decoder on one Mac. Voimem default decoder placement and automatic thread settings were untested. Gross process peak memory includes every worker allocation, including the opaque Accelerate setup.
