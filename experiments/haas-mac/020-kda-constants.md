# 020: Prepare KDA constants once per prediction backend

This is an unmeasured, default-OFF prototype from speech.cpp
`114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The ggml source is unchanged. No build,
inference, stage check or performance run is part of this source checkpoint.

## Change

`SPEECH_HAAS_KDA_CONSTANTS=0|1` is read once during the full `Recognizer`
construction. Unset and `0` keep the original path. `1` enables this experiment
only for `general.name=Haas-ja-v1.0` with KDA prediction. A different value fails.
The existing metadata factory and the one-argument prediction constructor remain
OFF and do no computation, even when the environment variable is set.

Each KDA layer currently builds `reshape(a_log, 1, heads) -> EXP -> NEG` into every
prediction graph. `DecoderStep` already retains its graph within a request, but
those EXP and NEG nodes still execute whenever that graph computes. The
experiment prepares the same expression on the same backend once and retains its
F32 result in a model-owned tensor. Each layer keeps the original `[1, heads]`
shape. There is no host `exp`, fused arithmetic, change of dtype, or concatenation
of the layers into another kernel shape. The gate still multiplies by the
negative exponential at the original point.

The Haas Q4_K file's metadata has two KDA layers and five heads, with F32 `a_log`
in both layers. This makes 40 bytes of retained values per active prediction
backend and removes two EXP and two NEG nodes from each prediction graph. The
backend allocation is larger than the payload because of alignment; the check
reports the actual buffer size. The setup graph has one copy into the retained
tensor per layer. `Decoding::graphs` continues to count the existing decoding
graphs and does not count this additional once-per-backend preparation. Its
separate preparation counter must accompany graph-count reports.

The existing host calculations of epsilon and square-root factors are unchanged.
This trial does not include state retention, decoder graph/plan reuse, matrix
packing, or any other optimization candidate.

## Lifetime and first-request accounting

This is **first prediction preparation**, not load-time precomputation.

| Boundary | Constant preparation |
|---|---|
| Metadata-only model information and default-option inspection | None; no prediction backend exists |
| Full model load without warmup | None; the switch is captured, but there is no constant buffer |
| First prediction graph on one backend | One successful setup graph and one owned buffer |
| Later graphs and requests on that backend | Reuse the same values and allocation |
| First prediction on the lazy CPU decoder | A separate setup using its copied weights and its backend |
| A separate loaded model | Independent constants derived from that model's weights |

The lazy CPU decoder installs its existing thread pool before any call can build
a prediction step. A model loaded directly on the CPU has only that one
prediction owner. A model loaded on a GPU can retain both device and CPU owners.
The standard engine warmup first decodes on the device, then also on the CPU when
the file's default decoding weights are quantized. Thus `ready` with standard
warmup can include two preparations, while `--no-warmup` ready includes none.
An F16 model's standard warmup does not prepare the optional CPU copy; an explicit
CPU request can still prepare it later. Pure load, ready with warmup, and the
first request on each backend must be reported separately.

Preparation takes place inside the first `const build` call. The existing C API
contract serializes requests on one model. The owner publishes its cache and
increments the success counter only after allocation and synchronized graph
computation succeed. An exception destroys the temporary cache; the next build
can retry. `ggml_backend_graph_compute` synchronizes before it returns, so the
temporary graph and allocator do not outlive their GPU work. The retained buffer
is destroyed before its context, model weights and owning backend. There is no
global cache, extra lock, changed CPU/GPU fallback, or reread of the file.

## Differential check

`fastconformer-constants-check` is a new check target. Its sources are
`checks/fastconformer/fastconformer-constants-check.cpp`, `constants-fixture.h`
and `constants-cases.h`. They are prepared but have not been compiled or run.

The synthetic fixture contains two F32 KDA layers with five different `a_log`
values per layer, nonzero recurrent behavior and a nonzero prediction-to-joint
projection. The test is designed to verify:

- Metadata construction and option inspection before backend initialization,
  and plain/prepared constructors with preparation count zero.
- Each retained tensor against an independently built original EXP then NEG
  expression on the same backend, bit for bit and finite. CPU and GPU results
  are compared separately; cross-backend identity is not assumed.
- The original prediction graph's per-layer EXP and NEG nodes and their absence
  from the prepared graph. This is a graph-structure check, not a dispatch timing.
- First prediction output/state, two interleaved 16-step histories, all recurrent
  and convolution state values, projected predictions and every joint logit.
  The same final label/frame must still depend on the preceding history.
- TDT tokens, frames, durations, graph counts and progress/cancellation points,
  including empty input, zero duration, max-symbol handling, repeated blank
  chains, a duration past the last frame, cancellation, another request, and
  custom-vocabulary boosting. A synthetic RNN-T comparison also retains its
  original host-state behavior.
- A CPU abort during preparation leaves count zero and no published buffer;
  retry succeeds once. This exercises a compute failure, not an injected
  allocator failure.
- A second synthetic model copy with changed `a_log` owns distinct tensors,
  derives different constants from its own weights and leaves the first model
  untouched.
- Real Haas models: pure load count zero; Device/CPU/Device/CPU decoding preserves
  text and the emitted path; counts remain one and allocations remain stable.
  Changing the environment after load does not change either model's captured
  setting. An unset switch preserves the original path. On a GPU the lazy CPU
  owner is distinct; on the CPU both decoding choices use the same owner.

The direct constant and initial prediction checks mark their results as graph
outputs. Their allocation/fusion can differ from the ordinary decoding graph.
The separate `DecoderStep`, TDT and real-Recognizer comparisons use their normal
output markings and are separate evidence. Public worker text/timestamp
comparisons still belong to the campaign screen/full evaluation. The metadata
check exercises the underlying factory path; it is not a claim that a public
`speech_model_info_open` smoke test has been run.

After a build slot is authorized, run the synthetic check on CPU F32 and Metal,
then the real-model check for Q4_K, Q8_0 and F16 on both CPU and Metal with four
threads. Each synthetic run needs a new output directory. For example, under
the shared performance lock:

```sh
build/fastconformer-constants-check --synthetic outputs/020/synthetic-cpu cpu 4
build/fastconformer-constants-check --synthetic outputs/020/synthetic-metal gpu 4
build/fastconformer-constants-check /path/to/Haas-616M-ja-v1.0-Q4_K.gguf gpu 4
```

The executable location follows the configured build layout; these are command
templates, not completed runs. Do not relax exact comparison on a difference.
The campaign has 11 official F32 frontend/encoder/joint-encoder-projection dumps
in `reference-mac`. Official KDA/prediction and complete TDT references needed for
this change, and Windows Vulkan validation, remain unavailable and must not be
reported as passed. M5 is unmeasured.

## Later performance gate

No speed claim follows from removing four nodes. Use the frozen common runner,
record its source and binary identities, explicitly pass the switch to both
variants, and reject unrecorded inherited `GGML_*`/`SPEECH_*` variables. Start with
the same nine clips, three alternating rounds, Q4_K/Q8_0/F16, CPU decoding with
four threads, and all clips warmed before measurement. Keep the fixed campaign
anchor as a separate comparison. Record first request, pure load and peak
footprint in addition to warm request latency. A small decoder stage change is
not by itself an end-to-end gain; use only the separately authorized diagnostic
or full evaluation when the screen warrants it.

All builds and inference runs use the campaign's shared lock, `-j4` for builds,
and `/usr/bin/python3` for the frozen runner. Large artifacts stay in ignored
`outputs/`; this branch is a measurement prototype and is not an adoption change.
