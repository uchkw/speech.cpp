# Prepare channel-first depthwise weights once

This prototype on `codex/exp-haas-014` is based on
`114c514430c8aa1455c077819f8e72e674a48b75`, with ggml pinned to
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It includes no other trial's runtime or ggml fusion changes.
Source checkpoint `dd0a27e412842bde002dc76cb02aaea82c94799f` was evaluated without runtime changes and rejected:
ordinary Metal output does not meet the strict equivalence gate. `014-depthwise-prepack-evaluation.md` records the
checks and bounded diagnostics. No performance or public ASR quality measurement ran, and no default changed.

## Existing weights and byte mapping

A read-only GGUF tensor-header inspection on 2026-11-07 found the same selected storage in each existing
Haas-616M-ja-v1.0 Q4_K, Q8_0 and F16 file:

| Tensors | Count | Original shape | Type | Bytes each |
|---|---:|---|---|---:|
| `sub.conv.{1,2}.dw.weight` | 2 | `[3,3,1,256]` | F32 | 9,216 |
| `blk.{0..23}.conv_dw.weight` | 24 | `[9,1024]` | F32 | 36,864 |

The total selected payload is 903,168 bytes, exactly 882 KiB. These are F32 tensors even in the Q4_K and Q8_0 files;
the filename's overall quantization type does not describe them. The layout already requires their F32 storage.
The source audit read headers only. The evaluation separately verified the existing model files by their full hashes;
only Q4_K proceeded to inference checks before the strict Metal failure stopped evaluation.

Every selected convolution calls the existing `depthwise()` helper, which constructs
`PERMUTE(2,3,1,0) -> CONT -> PERMUTE(3,2,0,1)` for its kernel. The encoder first views its `[9,1024]` matrix as
`[kw,1,1,C]`. For a general original kernel `[kw,kh,1,C]`, the first permutation and CONT produce contiguous
`[C,1,kw,kh]`. With `tap = y * kw + x`, the mapping is:

```text
destination[(tap * C + channel) * 4 : +4] = source[(channel * (kw * kh) + tap) * 4 : +4]
```

Preparation performs that mapping with `memcpy` of four bytes per element. It evaluates no floating-point
operation, so it preserves every F32 bit, including signed zeros. The request graph keeps the final permutation
view, with the same kernel shape and byte strides as before. The decoder's convolution weights are not selected.

## Ownership and load cost

`SPEECH_HAAS_DEPTHWISE_PREPACK=1` enables preparation only for Haas-ja-v1.0 with KDA. Unset and `0` keep the ordinary
path, and other switch values are errors. The encoder allocates a context and one owned weight buffer on the source
weights' buffer type, retains the original tensors unchanged, and rejects incompatible source storage or buffers.
Graphs borrow the prepared tensors while the encoder lives. The owned buffer is freed before its tensor context;
the model and its backend outlive the encoder. Input length and cancellation do not mutate the prepared weights.

Preparation occurs during encoder construction, after the model's weights have loaded. It adds host byte permutation,
allocation, source readback and destination upload to load time, plus 882 KiB of retained payload and backend alignment.
One original and one permuted matrix form the temporary host copy. It launches no ggml arithmetic graph for this
preparation, but backend transfers still cost time and memory. The existing lazy position-cache graph is unchanged.

Metadata-only model information and quantized-decoder inspection do not construct an encoder and do not prepare
these weights. CPU and GPU encoder instances prepare on their respective source buffer types. The lazy CPU copy
used only for decoding does not contain an encoder and receives no prepared depthwise tensors. Request progress,
decoder state, keywords and cancellation points are unchanged. API cancellation still needs its separate smoke check.

## Differential check

`fastconformer-depthwise-prepack-check` uses the actual encoder graph to identify every depthwise operand. It checks:

- Metadata inspection with the switch off and on, selector refusal, and the additional retained allocation.
- Exactly 26 depthwise convolutions in both graphs, exactly 26 fewer total CONT nodes, and the same final kernel
  types, shapes and byte strides. The candidate operands must point to the owned prepared weights on the original
  buffer type, with weight-buffer usage.
- A separate retention diagnostic marks the ordinary weight CONT outputs, computes the full graph, and compares
  their raw bytes with every candidate weight. It requires the total compared payload to be exactly 882 KiB.
- Normal graphs mark only their final `joint.enc` output, as recognition does. Their finite output vectors are
  compared bit for bit on short inputs, both sides of the existing position-cache boundary, 3,000 feature frames,
  returns to the first short input, and optional real F32 mono audio at the model's sample rate.
- Prepared tensor addresses and retained allocation remain fixed across requests. After destroying the candidate
  encoder, the original encoder still produces the original bytes, and all original weight payloads are unchanged.

The diagnostic also prints its final-output difference from the corresponding normal graph. Those differences
are explicitly labeled diagnostics, not ordinary-path passes or automatic failures of the normal-output gate;
nonfinite diagnostic values or mismatched shapes still fail. Intermediate retention can change allocation and
fusion, so only the separate normal graphs establish the candidate-versus-baseline output comparison.
Byte-preserving preparation alone does not prove that different graph scheduling preserves the final output.

This is a candidate-off/candidate-on differential check. It is not an original-binary comparison, an official F32
reference check, a recognition-quality result, or an API cancellation check. Record model/input hashes and raw logs.
Official F32 inference and Windows Vulkan remain unverified. The Q4_K evaluation uses these command interfaces:

```sh
./build/fastconformer-depthwise-prepack-check "$MODEL" cpu 4 "$SHORT_NPY" "$MEDIUM_NPY" "$LONG_NPY"
./build/fastconformer-depthwise-prepack-check "$MODEL" gpu 4 "$SHORT_NPY" "$MEDIUM_NPY" "$LONG_NPY"
```

## Evaluation boundary

Runner `a7e227c2b9067ee0b73a4582acf2e453cff5f179` was frozen with the switch allowed. Builds and checks used
`/private/tmp/speech-haas-mac-perf.lock` and four build jobs. The ordinary candidate-off/candidate-on check failed
on Metal, so original-binary continuity, Q8_0/F16 checks, official encoder checks, fixed-anchor comparisons and
performance/quality runs were not executed. The additional backend allocation was observed as 882 KiB, but load
time, process peak memory and latency were not measured. F16 would also be affected by this runtime change.
The 26 removed copies are graph nodes, not 26 removed whole encoder graphs.

This rejection concerns the current prototype's unmet strict gate. It is not a permanent rejection of depthwise
weight preparation or a reason to relax the numerical gate. A different graph or fusion design would be a separate
experiment; this evaluation includes no such change.
