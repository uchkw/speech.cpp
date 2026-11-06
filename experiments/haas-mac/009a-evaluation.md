# Reject packed KDA QKV projections after the bounded screen

Trial 009a preserves the outputs checked, but does not demonstrate a clear warm-latency gain. The Q4_K and F16
screen changes are small and Q8_0 moves in the slower direction. This rejects the measured projection variant;
the separate CPU small-operation fusion candidate under measure 009 has not been evaluated here.

## Frozen conditions

The executed runtime/check source is `9407d19f7f83a3326264d9abf93ac3d5fb23ca47`, independently based on campaign
`114c514430c8aa1455c077819f8e72e674a48b75`, with ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`. The evaluation
commit changes only experiment notes. No runtime from 003, 004, 008 or 009b is included.

The runner is frozen at `09cc03857b4af626fd537ee2be85f78c00d31135`, including worker SHA-256
`9049921d06e74580db1000e27dc5fbb7e4c07a4c0ed1559b4a6cf87a00b7cb88`. Python is 3.9.6 via `/usr/bin/python3`.
The machine is the Apple M2 Ultra with 24 CPU cores, 76 GPU cores and 192 GB memory. Builds use Release, native CPU
features, Metal and four compile jobs under the shared campaign lock. Inference is serialized under the same lock.

The primary comparison uses the same verified binary with `SPEECH_HAAS_KDA_QKV=0` and `=1`, Metal encoding and explicit
CPU4 decoding. The fixed campaign anchor is also measured. Its SDK cache setting is implicit while this candidate
has an explicit Xcode SDK path; primary off/on is unaffected because it shares one binary, and separate anchor
timings retain that build-environment limitation. Four threads are the tuned campaign setting, not the automatic
runtime default. F16 also executes packing and is not an unchanged runtime control.

## Numerical and recognition checks

The dedicated check passes all 1,983 assertions across twelve configurations: synthetic F32 and each of the real
Q4_K, Q8_0 and F16 files, on CPU1, CPU4 and Metal. Each synthetic case passes 166 assertions and each real-file case
165. Every case completes metadata-only inspection, invalid/mixed-type/shape/stride refusal, packed-byte and buffer
checks, actual production views, unselected operand checks, and packed-owner destruction with original bytes intact.

Normal graphs and graphs retaining intermediate products are compared separately. All normal outputs, intermediate
Q/K/V values, recurrent and convolution states, projected predictions and logits are bitwise equal. Two independent,
interleaved histories propagate their own states. Their 408 step-logit comparisons and label/duration argmax
comparisons have no difference. The synthetic history-dependence assertion also passes. No threshold is changed.

The existing 7.98, 14.52 and 24.72 second stage recordings then pass nine ordinary off/on comparisons of raw token
IDs, frame indices, duration decisions and text through `Recognizer::recognize()`. All graph counts remain equal:
the six-to-two Q/K/V product change removes four matrix operations per prediction step, not four whole graphs.

The same recordings also pass nine timestamped public-worker comparisons, with GPU-loaded models and explicit CPU4
decoding through the lazy CPU-copy route. Three `speech info --json` off/on comparisons agree. These public requests
exercise the GPU and lazily created CPU inference owners, beyond the dedicated check's directly loaded CPU tensors.

In the nine-input screen, all 81 latency texts and all 27 public-quality outputs, including token and segment text
and times, match off/on. Fixed-anchor output comparisons also match. Literal and repository CER remain unchanged
within this bounded screen; no full-corpus quality claim follows.

## Latency screen

Each precision uses nine existing duration-stratified recordings over three alternating rounds, plus nine separate
timestamp-enabled quality requests. Positive paired reduction means faster. It is the median of per-input median
latency ratios, not the percentage change in p50.

| Precision | Paired latency reduction | p50 off / on | RTFx off / on | Reduction against fixed anchor |
|---|---:|---:|---:|---:|
| Q4_K | +0.417% | 77.83 / 77.54 ms | 176.86 / 177.01 | +0.211% |
| Q8_0 | -0.701% | 74.10 / 74.95 ms | 183.53 / 182.83 | -0.875% |
| F16 | +0.197% | 73.20 / 74.45 ms | 186.86 / 187.47 | +0.201% |

| Precision | Round 1 paired reduction | Round 2 | Round 3 |
|---|---:|---:|---:|
| Q4_K | +1.263% | -0.031% | +0.375% |
| Q8_0 | -0.199% | -1.152% | -1.295% |
| F16 | +0.997% | +1.464% | +0.223% |

The small primary changes and their variation do not demonstrate a speed benefit beyond the noise. No 650-input
full or short-prefix run follows this screen. These observations do not establish a precise population slowdown.

Packing retains an additional 1,305,600 weight bytes per prediction network in Q4_K/Q8_0 and 2,457,600 in F16,
excluding alignment and object overhead. A GPU network and its lazy CPU copy each own a packed buffer. Load time,
first request and process peak memory are recorded separately in the artifact report. Their process-wide values
do not isolate these allocations or establish a memory improvement; no GPU VRAM result is claimed.

## Evidence and remaining candidate

Ignored artifacts are under `outputs/haas-mac-campaign/009a/`: `evaluation.html`, `evaluation.json`,
`checks-evaluation.json`, `screen/summary.json`, `dedicated-checks/result.json`, `audio-paths/result.json`, raw logs
and `evidence.json`. The verified build manifest is in `builds/009a-kda-qkv/` beside that artifact directory.

Official changed-stage KDA/prediction reference checks and Windows Vulkan remain unrun. This does not assert that
existing frontend, encoder or `joint.enc` references are absent. Synthetic F32 is not an official reference.
The remaining 009b candidate concerns CPU `NORM -> MUL -> ADD` fusion and requires its own independent source and
evaluation; none of its proposed barrier savings are attributed to 009a.
