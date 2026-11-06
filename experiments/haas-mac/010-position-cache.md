# Extend the Haas position projection cache

This prototype on `codex/exp-haas-010` is based on
`114c514430c8aa1455c077819f8e72e674a48b75` with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It contains no other campaign runtime change.
Source checkpoint `b47696771ff816224701e8942a81aeeda64a4fd4` was evaluated without runtime changes.
The separate `010-position-cache-evaluation.md` records the completed checks and measurements. Q4 is a conditional
candidate for longer inputs; this prototype changes no production default or measurement baseline.

## Change and size

`SPEECH_HAAS_POSITION_CACHE=0` retains the existing 32 MiB position budget; `1` selects 72 MiB for
`Haas-ja-v1.0` with KDA and global `rel_pos` attention. Unset means `0`; any other value is an error.
Other models keep their existing budget and local attention keeps its fixed context. This is a prototype switch,
not a new C API option or a change to the default model type.

`Encoder::attention_inputs()` already reuses each layer's position projection across requests. This experiment only
extends its range. If a request needs a greater span, it still computes positions in the utterance graph by the
existing path. The sinusoidal positions, weights, per-layer projection operations, relative shift, attention and
decoder are unchanged. Capturing projection tensor pointers in `EncoderStages` adds no nodes or output marks.

For 24 layers of width 1024, each retained position takes `24 * 1024 * 4 = 98,304` bytes. The budget gives
`rows = floor(budget / 98,304)`, `span = floor((rows - 1) / 2)` and `2 * span + 1` retained positions. A request
with `T` encoder frames uses the cache when `T - 1 <= span`.

Inputs at `T <= 171` already reuse projections. Only `172 <= T <= 384` removes the 24 per-request position products
relative to the baseline. Inputs at `T > 384` keep the original uncached path. The larger retained allocation is paid
as soon as any candidate-eligible request creates it, including short requests that already fit the original cache.

| Budget | Integer rows | Span | Stored positions | Eligible T | Tensor payload |
|---|---:|---:|---:|---:|---:|
| 32 MiB | 341 | 170 | 341 | 1 to 171 | 33,521,664 bytes / 31.96875 MiB |
| 72 MiB | 768 | 383 | 767 | 1 to 384 | 75,399,168 bytes / 71.90625 MiB |

The payload increases by 41,877,504 bytes, or **39.9375 MiB**. The check also prints the actual retained backend
buffer size; alignment and other process allocations are not part of the payload calculation. Cache construction
has temporary graph and input allocations, so process peak memory must be measured separately.

The frontend gives `samples / hop_length` frames and the three stride-2 subsampling layers round upward. With this
model's 16 kHz rate and 160-sample hop, 30 seconds gives 3,000 feature frames and `T = 375`, inside the candidate's
range. The check calls these actual frame-count functions and prints the actual sample, feature and encoder frame
counts for every supplied audio input. Its synthetic 3,000-feature-frame case checks the same encoder shape; it is
not speech audio or a CER sample.

## Lazy initialization and timing

`Positions` is constructed inside the first cache-eligible `Encoder::build()`, not during `Encoder` or model
construction. Keep the following times distinct:

1. Pure load to `ready` with the worker's `--no-warmup`: the position cache is still empty.
2. The normal engine's load-to-ready including its one-second warmup: that warmup creates the cache and pays the
   projection cost. The increased budget can change this time and the retained memory at `ready`.
3. The first cache-eligible request after a no-warmup load: the cache construction belongs to this request, even
   when one or more earlier requests were too long to use the cache.
4. Warm request latency after every latency input has been warmed once: this is the campaign's primary metric.

The first eligible request can be a different file on each side. If the first warmup has `171 < T <= 384`, only the
candidate creates its cache then; the baseline creates its cache on a later shorter input. The common worker runner
currently retains `first_request` and `warmup_count`, not each later warmup's timing. Do not infer the later creation
time from those fields. Record a separate bounded first-eligible diagnostic, or explicitly mark that time unavailable.
The check reports `cache_created` and retained bytes for each of its calls, but its timings are not worker latencies.

## Differential checks

`fastconformer-position-cache-check` has two independent modes. Both use one GGUF's weights for the two encoders,
four CPU threads, and the existing greedy decoder. They check the switch selection, lazy constructor state and the
30-second frame calculation. All output must be finite. There is no adjustable numeric tolerance.

The `positions` mode captures the actual tensors returned by the encoder's global attention construction. It runs
32 MiB, 72 MiB and explicit zero-cache encoders through
`T = 385, 1, 170, 171, 172, 375, 383, 384, 385, 171, 1, 384, 1`.
This covers both limits, an initial input beyond both limits, and returns to short inputs after longer ones.

- The actual cached tensor must be the central contiguous view with the expected parent extent and offset. Each
  layer must retain the same parent tensor across requests. The uncached product must read that layer's exact
  `attn_pos.weight` tensor, and all layers must share the same positional input.
- Uncached positional inputs are compared exactly with central slices of the first `T = 385` input. This tests the
  position coordinates across lengths without substituting a second sine/cosine implementation.
- Candidate versus baseline projections are compared bit for bit at every layer. Repeated `T = 1` values must also
  match their own earlier values exactly. Any difference fails this gate and is printed with SNR and maximum error.
- Comparisons of each cache with the zero-cache encoder are separately labeled diagnostics. The existing
  341-column cache product can already round differently from a request's narrower product. Those differences are
  neither passes nor an automatic reason to relax the candidate-versus-baseline gate. Nonfinite results fail.
- The check prints graph node counts and requires zero utterance-graph position products for cached inputs and
  exactly 24 for uncached inputs. Cache construction itself runs a separate projection graph before that graph.

The projection diagnostic marks the subsampling and position outputs to allocate its inputs and read the projections.
It does not compute the rest of the conformer and **does not reproduce the normal graph's allocator or fusion
decisions**. It cannot establish ordinary-path equality or speed by itself.

The `outputs` mode builds the ordinary encoder-to-joint graph with only its final output marked, just as
`Recognizer::encode()` does. It first compares nonconstant synthetic features at `T = 385` and `T = 375`, then the
supplied real audio inputs in order, and then the first real input again. Supply a short input before longer ones to
exercise a return to short audio. It compares final F32 tensors exactly, and separately compares greedy token IDs,
emission frames, TDT durations, decoder graph counts and text. These internal path checks supplement the worker's
public text and timestamp comparison; they are not CER estimates. Synthetic features are never scored as speech.

Changing the projection from 341 or `2T - 1` columns to 767 columns may select another GEMM shape or backend kernel.
Removing the 24 position products from a request can also change graph scheduling, fusion and allocation. Either can
change floating-point rounding even though the represented positions and weights are the same. Preserve and report
any differences; do not loosen a threshold or reinterpret nonmatching public output as exact.

After build authorization, run both modes on CPU and Metal for Q4_K, Q8_0 and F16. Every invocation holds the shared
lock, for example:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$BUILD/fastconformer-position-cache-check" "$MODEL" gpu positions
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$BUILD/fastconformer-position-cache-check" "$MODEL" gpu outputs "$SHORT_NPY" "$MEDIUM_NPY" "$LONG_NPY"
```

The `.npy` inputs are finite mono float32 samples at the model's sample rate. Preserve their hashes and source PCM
identity. An official F32 stage check requires the pinned official dumps and matching F32 model. The completed
evaluation uses those separately from the differential Q4_K/Q8_0/F16 checks. Windows Vulkan remains unverified.

## Evaluation protocol

Build from the clean checkpoint with the campaign helper, `/usr/bin/python3`, Release/native/Metal and four compile
jobs under the shared lock. Freeze the current common runner independently from this prototype's source identity.
Its allowlist includes `SPEECH_HAAS_POSITION_CACHE=0|1` from campaign commit `0656426`.

Start with the existing nine-input, three-alternating-round screen using four CPU decoder threads and the fixed
campaign anchor. Record outputs, CER, paired per-file latency, length groups, round variation, pure load, first
request, first cache-eligible request when available, and process peak memory. Separate the normal ready warmup
diagnostic from the no-warmup worker comparison. Native F16 is also affected by this cache change and is not an
unchanged runtime control. Every disabled-switch result must be checked against the unchanged anchor.

Only a reproducible request-level gain justifies 30 inputs over three rounds, the 650-input quality check and short
inputs. A speed result from this prototype is not an adopted implementation. Any later implementation without a
switch needs a fresh build and evaluation. The evaluation record preserves the original screen, independent repeat,
full comparison and short supplement separately, including regressions and startup costs.
