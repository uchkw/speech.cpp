# Trial 007b: tile content attention with a dynamic relative-score mask

This source-only experiment starts from speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and unchanged ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It has not been compiled or run. It is independent of rejected trial 007a,
which fused ADD and softmax without tiling QK or PV. No speed or accuracy result is claimed here.

## Scope

`SPEECH_HAAS_REL_FA=1` enables the graph variant for Haas-ja-v1.0 only: global relative attention, 24 unbiased layers,
width 1024, eight heads of 128, KDA prediction, on Metal, and 1 to 384 subsampled frames. An unset switch or `0`
keeps the original graph. Any other switch value, another model, another backend, an out-of-range input or a backend
that cannot compute the constructed FA_EXT throws a named error. Enabling the trial never silently selects the old
graph. The switch is read when an encoder is constructed; use fresh processes for the paired campaign.

Each layer still computes the projected positions and the dense positional product `(q + v) P^T`, of shape
`[2T - 1, T, 8]`. The original relative shift is the same view: query `i`, key `j` reads position index
`T - 1 - i + j`. The view becomes a contiguous F32 copy, is multiplied by `1/sqrt(128)`, and is cast to a contiguous
F16 mask `[T, T, 8, 1]`. It is recomputed from that layer's queries, not cached as a static positional mask.

FA_EXT receives `(q + u)`, K and V as F32 tensors `[128, T, 8, 1]`, the mask above, scale `1/sqrt(128)`, max_bias 0,
and logit_softcap 0. K and V each have an explicit CONT into this canonical layout, without a type conversion.
The two copies and their dispatches remain part of the measured variant cost. Thus every query sees every real key, the two query biases remain distinct, and no ALiBi slope,
causal mask, local window, sink or sparse-key hint is introduced. Its output `[128, 8, T, 1]` already has the layout
the output projection takes. The old content-score tensor, separate softmax, transposed V copy and PV product are
absent from the enabled graph.

This is tiled QK/online-softmax/PV with a dense positional prepass. It is not fully fused relative-position attention
and does not make all attention memory linear: the dense positional product, F32 preparation and F16 mask remain
quadratic in T. FA_EXT also scans mask blocks and pads the final K/V block when T does not fill one. A later fully
tiled positional kernel would need a query-dependent relative-position gather; concatenating one static P with K
cannot express it.

At 16 kHz with the model's hop of 160 and subsampling of eight, exactly 30 seconds is 480000 samples,
3000 feature frames and 375 encoder frames. `NemoFrontend::frames()` drops the extra centered-STFT frame before
three ceiling halvings. The experimental bound of 384 also covers neighboring tail fixtures, including T=376.
The existing Q4_K file's metadata confirms these constants; no weights were read to establish the bound.

The pinned FA tail-padding source copies a K/V time stride instead of only the logical channel row. For the
encoder's original permuted layout these differ. That source finding has not been reproduced here and does not
establish an impact on the original Haas graph, which does not use FA. The backend cause is being investigated
separately; choosing a canonical contiguous FA layout here does not claim to fix ggml. Production review must
include that investigation.

## Arithmetic changes to measure

- The original logits round `(AC + BD) * scale` in F32. FA_EXT rounds `AC * scale + F16(BD * scale)`. Distributing the
  scale changes rounding even before the mask cast. Q is not prescaled in this variant.
- The two explicit K/V CONT operations preserve F32 values; they change memory layout and incur a copy cost.
  There is no explicit K/V cast. The pinned Metal F32 FA kernels stage Q in half, while reading K and V in F32
  (`fa_f32.metal:23-29` and `fa_vec_f32.metal:13-19`). The ordinary large-batch GEMMs stage both operands in half
  (`mul_mm.metal`), so K, V and softmax probabilities do not have the same rounding in the two paths. Small batches
  also cross the original mat-vec/MMA dispatch boundaries; they require their own checks.
- FA uses online maxima, exponential sums, output rescaling and a different reduction order. Its non-vector F32
  kernel keeps probabilities and output accumulation in F32. This differs from the old separate softmax followed by
  a GEMM whose operands are staged in half. `ggml_prec_set_acc(F32)` does not undo these differences.
- For fewer than 20 queries Metal uses its vector FA kernel. Otherwise M2 uses the simdgroup kernel, not the M5
  tensor API. Both path families and partial query/key tiles need coverage. F16 mask overflow, underflow and finite
  range must be recorded on real inputs; no clamp or accuracy fallback is added.
- The positional product and its projection/cache have the original arithmetic. No q/v sharing, position reversal,
  windowing, or change to the scale/head count is part of the trial.

## Validation gate, after a compute lease

1. Compile original and candidate with identical flags and link the same diagnostic source separately against each.
   Keep original, candidate OFF and candidate ON as separate fresh processes. Verify OFF against original first.
2. Run `007b-encoder-check` with expected FA counts 0, 0 and 24. It checks the actual graph's F32 Q/K/V and
   contiguous F32 K/V and head-specific F16 mask, saves ordinary projected output and retained subsampling/layer/encoder/projected tensors,
   records every retained mask's finite range, zero and F16-subnormal counts, and requires ordinary and retained
   projected bytes to agree within each side. Synthetic T values are
   `1,16,17,19,20,31,32,33,63,64,65,171,172,375,376,384`; real `features.npy` files can replace the synthetic inputs.
   Its channel argmax is an activation diagnostic, not a vocabulary or duration decision.
3. Compare every saved tensor with original: finite values, exact-byte count, max absolute/relative difference, NMSE,
   SNR and frame-wise channel argmax. Run `007b-attention-check` on CPU and Metal: it uses asymmetric dyadic Q/K/V/P with distinct
   u/v biases, an independent scalar formula and explicit shared-head/reversed-position negative controls. It checks
   every F16 mask entry against the scalar relative index and uses the pinned ggml FA_EXT NMSE bound of `5e-4`.
   It includes the T=19/20 vector-kernel boundary, partial tiles, T=376 and larger finite logits. The direct CPU FA
   fixture is not a CPU execution of the family variant, which deliberately rejects CPU ON.
4. Keep the official encoder thresholds unchanged: CPU F32 90 dB with the variant OFF, and Metal F32/F16 25 dB with
   OFF and ON, through `fastconformer-encoder-check`. Capture every layer's SNR as well as the final pass/fail. The
   CPU ON rejection is intentional. The check does not admit Q4_K/Q8_0 by loosening its existing type gate.
5. Capture raw TDT ids, frames and durations with the existing 007a stage-path diagnostic, compiled against all
   three runtime sides. It uses only unchanged family interfaces. Compare full public text/tokens/times and both
   literal and repository-normalized CER on the same 650 quality inputs for Q4_K, Q8_0 and F16. Investigate all
   changed decisions; tensor SNR alone cannot approve a recognition regression.
6. Check invalid switch values, CPU ON, another model, T=385 and backend support rejection before timing. Run the
   ordinary non-instrumented end-to-end screen and alternating stage measurements only after numerical checks.
   Report warm latency p50/p90/p99, RTFx, first request, load time and peak memory. Escalate to the full/short-prefix
   campaign only for a stable gain beyond round noise. Vulkan and M5 are untested by this Mac experiment.

Raw artifacts belong under `outputs/haas-mac-campaign/007b/`, not in Git. The coordinator owns compute allocation,
publication, later evaluation reports and any decision to replace this experiment with production code.
