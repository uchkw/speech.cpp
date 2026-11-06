# Trial 009a: pack KDA Q/K/V rows

This is an unbuilt, unmeasured source checkpoint for [campaign issue 1](https://github.com/uchkw/speech.cpp/issues/1).
The experiment starts from speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75`, whose inference baseline is
`03555100287c118b41d1b475044e5ab1e07baa8c`, and keeps ggml `e1a9ac12edbb57402618fd6936598603f5ee6266` unchanged.
It contains none of the runtime changes from trials 003, 004, 008 or the proposed CPU normalization trial 009b.
No speed or recognition-accuracy result is established by this checkpoint.

## Scope and ownership

`SPEECH_HAAS_KDA_QKV=1` prepares one additional matrix per KDA prediction layer. Its rows are the original Q, K and V
rows in that order, with the original storage type and bytes. One matrix-vector product replaces three; three views
feed the existing convolutions. Two Haas layers therefore replace six matrix products with two. This removes four
matrix operations from each prediction step; it does not remove four whole graphs or establish a CPU thread-start
saving. Normalization, gates, recurrence, convolution history, state transfers and the joint remain unchanged.

The unset switch and `0` keep the ordinary path. An inference instance rejects other values. Explicit `1` rejects
Q/K/V matrices with different types, shapes, noncontiguous rows or different source buffer types. It does not silently
disable packing. The ordinary GGUF layout validation still supplies the model-specific dimensions.

Only the prediction network used for inference prepares weights. The factory's metadata-only mode, including
`quantized_decoding()`, neither parses the switch nor loads or allocates tensors. The packed context and buffer belong
to that prediction network and use the source weights' buffer type. They outlive its graphs. The original model
tensors remain alive and unchanged. A GPU model and its lazily created CPU decoding copy can each own a packed copy
on their respective buffer types; this experiment does not change the existing choice of decoding device.

## Existing weight audit

A read-only GGUF header and selected-payload audit on 2026-11-07 found these six tensors:
`pred.kda.{0,1}.{q,k,v}.weight`. Every shape is `[640,320]`; the packed shape is `[640,960]`.

| Existing model file | Actual Q/K/V type | Bytes per original matrix | Additional bytes per prediction network |
|---|---|---:|---:|
| Haas-616M-ja-v1.0-Q4_K.gguf | Q8_0 | 217,600 | 1,305,600 |
| Haas-616M-ja-v1.0-Q8_0.gguf | Q8_0 | 217,600 | 1,305,600 |
| Haas-616M-ja-v1.0-F16.gguf | F16 | 409,600 | 2,457,600 |

These are packed tensor payload sizes, excluding backend allocation alignment and object overhead. The temporary
host copy holds one original matrix at a time. The six selected payload SHA-256 values match between the Q4_K and
Q8_0 files. The audit and per-tensor hashes are in ignored `outputs/009a-weight-audit.json` in this worktree. Its
whole-file hashes come from the campaign's model catalog and were not rehashed for this header audit.

The Q4_K model's selected path is therefore Q8_0 matrix-vector arithmetic, not Q4_K matrix-vector arithmetic.
No dequantization or requantization occurs during packing. The larger output row count may nevertheless change a
CPU or GPU kernel's dispatch, row grouping or accumulation path. Identical weight bytes alone do not establish
bitwise output equality. F32 synthetic, F16 and Q8_0 arithmetic must be reported separately, including CPU thread
counts. The unchanged ggml kernel sources do not imply an unchanged dispatch for the new shape.

## Prepared differential check

`fastconformer-kda-qkv-check` accepts a real model or writes a deterministic decoder-only all-F32 synthetic model.
The fixture keeps the actual two-layer, 46,720-float recurrent and convolution state layout. Its nonzero joint
prediction projection makes recurrent history observable in logits. It comes from the campaign's state-check
fixture; it does not contain another trial's inference implementation.

The check is designed to perform these checks when execution is authorized:

- Inspect metadata without loaded tensors with packing both off and on.
- Reject an invalid switch and deliberately altered K matrix descriptors for mixed type, shape and stride. Restore
  the descriptors before inference. These are constructor guard checks, not additional converted model files.
- Compare every packed byte with the three original tensors, their storage and buffer types, and weight-buffer
  usage. Count the six-to-two matrix products and verify the offsets, size and actual use of all three production
  views. Compare every other named model operand's pointer and use count.
- Run ordinary and packed graphs with the same three retained outputs and output order as `DecoderStep`. Compare
  each with its instrumented version, which additionally exposes raw Q/K/V products and the prediction network's
  output. Compare those intermediate values between the ordinary and packed paths.
- Run two independent, interleaved `DecoderStep` histories through at least sixteen updates, then give both the same
  label and frame. Propagate each path's own state, compare every recurrent and convolution state value, projected
  prediction and joint logit, and require the synthetic histories to produce different history-dependent outputs.
- Count strict logits differences separately from label and duration argmax differences. Nonfinite values fail.
  Any bitwise discrepancy makes the check fail even when the argmax is unchanged; it is evidence to inspect, not an
  automatically accepted rounding allowance.
- Destroy the packed prediction owner, rerun the original network, and compare all six original tensor payloads.

This is a candidate-off/candidate-on differential check. It does not establish equality with a separately built
original binary, an official NeMo/FLA stage reference, or every TDT blank/duration/cancellation decision. Those
boundaries remain explicit even if this check later passes. It also makes no concurrent-model-thread claim.

## Pending execution

No compiler, inference check, CPU timing, Metal timing or Vulkan run has been performed for this checkpoint.
Only source inspection, `git diff --check`, and the read-only existing-model header/payload audit are complete.
The coordinator schedules the serial execution after earlier trials, using the shared
`/private/tmp/speech-haas-mac-perf.lock` and builds bounded to `-j4`.

Once authorized, build `fastconformer-kda-qkv-check`, then run the synthetic fixture on CPU with one and four
threads and on Metal. Run the same differential check for each of the existing F16, Q8_0 and Q4_K files with CPU
one/four threads and Metal. Example check commands, not executed here:

```sh
./build/fastconformer-kda-qkv-check --synthetic outputs/009a-fixture cpu 1
./build/fastconformer-kda-qkv-check --synthetic outputs/009a-fixture cpu 4
./build/fastconformer-kda-qkv-check --synthetic outputs/009a-fixture gpu 4
./build/fastconformer-kda-qkv-check /path/to/Haas-616M-ja-v1.0-Q4_K.gguf cpu 4
```

Keep original-binary comparisons, actual recognition tokens/times/text, cold preparation cost, warm prediction and
whole-request timing, and peak memory separate from these numerical checks. Alternate the measured variants and
record type, device and thread count. Official dumps and Vulkan hardware validation remain unperformed until their
own runs provide evidence. No adoption decision or numerical tolerance is changed by this source checkpoint.
