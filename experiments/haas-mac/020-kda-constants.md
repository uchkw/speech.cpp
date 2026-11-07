# 020: Prepare KDA constants once per prediction backend

Rejected at the exact same-backend correctness gate. The original real Q4_K Metal comparison failed 104 of 173 assertions with default fusion. Recognition accuracy and end-to-end performance were not evaluated.

This is a default-off experiment on speech.cpp source 980269a and ggml e1a9ac1.

## Selector

`SPEECH_HAAS_KDA_CONSTANTS` is read once during full `Recognizer` construction.
Unset and `0` keep the original path, so the prototype is disabled by default.
`1` enables it only for `general.name=Haas-ja-v1.0` with KDA prediction; any
other value fails construction. The metadata factory and the one-argument
prediction constructor stay on the original path and prepare nothing even when
the variable is set, so inspecting model metadata alone never prepares this prototype.

## When preparation actually happens

This is first-prediction preparation, not load-time precomputation.

- Metadata-only inspection and a full load without warmup prepare nothing; the
  switch is only captured.
- The first prediction graph on a backend runs one setup graph and allocates
  one owned buffer; later graphs and requests on that backend reuse the
  retained values and that buffer. The temporary setup graph and its
  allocator are destroyed after preparation.
- The lazy CPU decoder prepares separately, from its copied weights on its own
  backend, so a GPU-loaded model can retain both device and CPU constants
  while a CPU-loaded model owns only one set.
- Standard warmup decodes on the device, then also on the CPU when the default
  decoding weights are quantized, so `ready` with warmup can include two
  preparations and `--no-warmup` ready none. An F16 warmup skips the CPU copy,
  which an explicit later CPU request can still prepare.
- Pure load, ready and the first request on each backend have separate cost
  boundaries. A separately loaded model derives its own constants.

## Payload and retention

Each KDA layer otherwise rebuilds `reshape(a_log, 1, heads) -> EXP -> NEG` into
every prediction graph. The prototype computes the same expression once on the
same backend and retains the F32 result in a model-owned tensor with the
original `[1, heads]` shape: two layers with five heads here, 40 bytes of
retained values per active prediction backend, in a larger actual allocation
because of alignment. There is no host `exp`, no dtype change and no
concatenation into another kernel shape; the gate multiplies by the negative
exponential at the original point. A buffer is published only after allocation
and synchronized computation succeed, then destroyed before its weights and backend.

## Graph-count limits

`Decoding::graphs` counts decoding graphs only and does not count the
once-per-backend preparation, so graph-count reports must be accompanied by
the separate preparation counter. Removing four EXP/NEG nodes per prediction
graph is a structural observation, not a speed measurement.

## Check design

`fastconformer-constants-check` compares each retained tensor bit for bit
against an independently built EXP-then-NEG expression on the same backend;
CPU and GPU results are compared separately and cross-backend identity is not
assumed. Decoding comparisons cover the first output and state, two distinct
interleaved 17-step histories, all recurrent and convolution state values,
projected predictions, every joint logit, TDT tokens, frames and durations,
progress and cancellation, a second model with changed `a_log`, and real Haas
models across decoding orders. The direct constant and initial-prediction
checks mark their results as graph outputs, so their allocation and fusion can
differ from the ordinary decoding graph. The separate `DecoderStep`, TDT and
real-Recognizer comparisons are ordinary, with normal output markings: separate evidence.

| Input | Backend | Threads | Assertions | Failures |
| --- | --- | --- | --- | --- |
| synthetic F32 | CPU | 1 | 185 | 0 |
| synthetic F32 | CPU | 4 | 185 | 0 |
| synthetic F32 | MTL0 | 4 | 178 | 0 |
| Haas Q4_K weights | CPU | 4 | 177 | 0 |
| Haas Q4_K weights | MTL0 | 4 | 173 | 104 |

## Numerical comparison

| Q4_K Metal comparison | Failures | Minimum SNR (dB) | Maximum absolute difference |
| --- | --- | --- | --- |
| Initial prediction output | 1 | 142.7 | 4.77e-07 |
| Initial prediction state | 1 | 146.7 | 2.38e-07 |
| Reused step, state | 34 | 145.3 | 9.54e-07 |
| Reused step, projected prediction | 34 | 133.1 | 2.86e-06 |
| Reused step, joint logits | 34 | 139.1 | 1.14e-05 |

The initial plain and prepared prediction graphs use the same output markings.
Their encoding trace can expose a difference in fusion length. It contains no
intermediate tensor values that locate the first numerical divergence.

## Diagnostics

- Five primary configurations ran 898 assertions with 104 failures. The retained constants matched independent same-backend EXP then NEG bit for bit. Each direct run reported one preparation, 40 bytes of F32 payload and a 64-byte backend allocation. The allocation excludes graph, model and process memory.
- The initial comparison used label 1 and a zero old state. The 34 comparisons per later tensor family came from two 17-step histories. After the first step each side receives its own preceding state. The minimum SNR and maximum absolute difference in the table come from the printed results for each family. Passing generated-input TDT path and progress assertions do not establish recognition on recordings.
- Disabling Metal fusion for one separate invocation of the same dedicated binary passed all 173 assertions. The original failure remains the primary result. The check is sensitive to fusion under these conditions.
- One log-only launcher with default fusion and GRAPH_DEBUG=1 reproduced the original 12,124 stdout bytes exactly, including every printed difference. Seven retained runtime archives and the original check binary were byte-identical across diagnostic linking. The check logic, inputs, numerical threshold and graph-output markings stayed identical.
- After excluding the setup graph (trace records 58–69), the initial ordinary and prepared graphs occupy records 70–184 and 185–268. Each has 15 logged multi-operation fusion groups. Only the eighth group, the second NORM, changes length. Ordinary node_65 at local index 0 fuses three operations. Prepared node_62 at local index 37 fuses two, then ADD node_64 starts at the next local index 0. The three NORM group lengths are 3/3/3 and 3/2/3 respectively.
- The KDA source order identifies the second NORM as the pre-normalization of layer 1, pred.kda.1.norm. The log and backend source point to NORM+MUL+ADD becoming NORM+MUL followed by a separate ADD at a local partition boundary. No intermediate tensor or instruction trace locates the first numerical divergence, proves FMA execution or identifies a sole numerical cause.
- The first diagnostic compile stopped before linking because the added repository include root shadowed the SDK header <version> with VERSION on this case-insensitive filesystem. A source-local checks symlink resolved the check include, and the final compile used the exact original compiler flags. Its seven-archive link order also matched. All failed-attempt logs remain available.

## Limits

- The six planned real-model checks for Q8_0, F16 and F32 on CPU4 and Metal did not run. The passing real Q4_K CPU4 assertions describe this prototype under the dedicated test. A CPU-prediction-only implementation and its performance require a separate trial.
- The official F32 encoder comparison did not run. Available reference inputs cover the encoder boundary. Matching official KDA prediction, recurrent-state and complete TDT stage references were unavailable for this evaluation.
- Public metadata, API/worker cancellation, recorded-speech keywords, timestamps and public output equivalence were not evaluated. The lifecycle and generated-input path checks inside the dedicated executable have narrower scope.
- Pure load, warmup-ready time, first request, warm 9-input/3-round screening, the full corpus and peak process memory were not measured. There is no fixed-anchor or paired speed result. Windows Vulkan and M5 were untested.
- The matched M2 Ultra arm64 Release build matched all 74 common compile-flag files and all 74 common link recipes of the retained 014 original, after normalizing source/build roots. The original build served as a build-configuration reference; no new original worker or end-to-end comparison ran.

## Evidence

The rendered evidence page for this evaluation is at
`outputs/haas-mac-campaign/020/evaluation.html`.
