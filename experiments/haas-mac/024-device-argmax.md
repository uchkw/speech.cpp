# 024: Reduce the TDT joint row before host readback

## Status and scope

Source-only prototype from speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75`, with ggml fixed at
`e1a9ac12edbb57402618fd6936598603f5ee6266`. No build, C++ check, inference, GPU work or performance measurement has
run for this checkpoint. This is not an adopted optimization and gives no measured gain.

`SPEECH_HAAS_TDT_DEVICE_ARGMAX` is unset or `0` by default and accepts only `0` or `1`. A TDT decoder reads it only
when `general.name` is exactly `Haas-ja-v1.0` and `fastconformer.decoder.prediction` is `kda`. Other models ignore even
invalid values. A metadata-only model read constructs no decoder and does not read the switch. The constructor
captures the setting; a request does not read the environment again.

With `1`, an unboosted request uses the new path only when its actual decoding backend belongs to the `MTL` registry.
CPU decoding, Vulkan and every boosted decoding, including alpha zero, keep the full-logits path. The principal
campaign workload decodes on the CPU with four threads, so this experiment does not improve that path. Device decoding
must be evaluated as a separate condition; its gain cannot be added to CPU-decoding gains.

This variant changes neither state readback nor host frame/state buffers and does not keep a blank-joint graph
resident. It is independent of 003 and 017. It keeps the ordinary prediction graph, the allocator shared by the
recreated blank-joint graphs, their count and the position of progress/cancellation checks.

## Graph and ownership

Each joint row remains an F32 `OUTPUT` exactly as before. After the original outputs, the step graph adds two
one-dimensional views and two `ggml_argmax` outputs. The blank-joint graph adds the same pair after its ordinary
logits output. No scalar-copy, concatenation or extra graph is introduced. The only additional arithmetic operations
are the two reductions.

The first range is `[0, blank + 1)`, tokens plus blank. The second is `[blank + 1, outputs)`, the duration logits.
Haas has 3,073 and 5 entries. `TdtArgmax` holds non-owning tensor pointers into the caller's graph; it reads them after
that graph's synchronous compute and before the graph is destroyed or its allocator is reused. Source logits are
retained so the two leading floats are still available after both reductions, and checks can read the complete row
without adding diagnostic retention to the graph.

One decision performs four tensor gets: two I32 indices and the F32 leading value of each range. This is 16 bytes,
against the old 12,312-byte F32 row, with four calls instead of one. The host replaces an index with zero when that
range's leading value is NaN, then checks both indices before they are used. It does not read a finite certificate,
scan the rest of the row or switch back to full readback. Unsupported ARGMAX on an explicitly selected backend throws.

Retaining logits is deliberate. Actual graph allocation, additional reduction outputs, command scheduling and the
cost of four small gets remain to be measured. Reduced byte count does not establish a speedup on unified memory.

## Source semantics and limits

At the pinned ggml commit, [CPU argmax](https://github.com/nyosegawa/ggml/blob/e1a9ac12edbb57402618fd6936598603f5ee6266/src/ggml-cpu/vec.h#L1618)
starts with negative infinity and changes the index only on a strict greater-than comparison. The
[Metal kernel](https://github.com/nyosegawa/ggml/blob/e1a9ac12edbb57402618fd6936598603f5ee6266/src/ggml-metal/kernels/misc.metal#L3)
keeps the first maximum in each thread and reduces equal maxima to the smallest index. When no entry exceeds negative
infinity, it returns zero. Its bounds check covers the final partial SIMD group. The F32 view offset of the duration
range is 12,292 bytes for Haas.

Under ordered IEEE comparisons, both ignore nonleading NaNs. All negative infinity, all NaNs, repeated positive
infinity and signed-zero ties agree with `first_argmax`, except that `first_argmax` freezes at zero for a leading NaN.
Reading that first float fixes this algorithmic difference: `[NaN, 1, 2]` must select zero rather than two. A source
proof of these comparisons is not a proof of every compiled Metal floating-point behavior.

The [Apple Metal Shading Language specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
dated 2026-06-04, sections 1.6.3, 8.1, 8.4 and 8.5, permits flushing subnormal operands/results and does not promise
ordered NaN/Inf comparisons in fast mode. The pinned Metal CMake release flags use `-O3`; shader-debug flags disable
fast math. Runtime library compilation does not override math mode. These settings are not changed by this experiment.

A specification-permitted counterexample is `[0, denorm_min]`: host `first_argmax` selects one; flushing the second
value to zero permits device index zero. `[denorm_min, 2 * denorm_min]` and `[-denorm_min, 0]` also distinguish the
comparisons. No such mismatch has been observed here because execution is pending. Tests keep the original bits and
fail on a mismatch; they do not round inputs, loosen a threshold or treat failure as a pass. Nonfinite/subnormal
results are reported separately from normal audio. Universal equivalence is not claimed, and adopting the variant
would require addressing any remaining public-contract difference after establishing an actual benefit.

The existing decoding decision record rejects device argmax because ggml's CPU once chose the last equal maximum.
That reason differs from the pinned CPU/Metal source above. This reversible experiment records the changed evidence;
it does not change the permanent decision before evaluation.

## Checks prepared, not run

`fastconformer-tdt-argmax-check --raw <device> <threads>` separates 162 finite cases and 216 nonfinite/subnormal cases.
Token/blank widths are 1, 5, 31, 32, 33, 63, 64, 65 and 3,073; every case also has five duration entries. It checks both
ranges against the unchanged host function, including first/last/interior maxima, ties across lanes and SIMD groups,
signed zeros, adjacent F32 values, large finite values, leading/later NaNs, all NaNs, infinities and subnormals. The
opposite range contains a large sentinel. Repeated graph computations poison both output IDs first, and compare all
retained source bits afterward. It checks exactly two arithmetic nodes and rejects invalid row type/shape/ranges
before backend computation.

`fastconformer-tdt-argmax-check --synthetic <output-dir> <device> <threads>` generates a small F32 KDA fixture. Its
state/controlled-logit fixture and TDT boundary cases are taken from experiment 017's checks; no 017 runtime change is
included. It compares complete recurrent/convolution state, projected prediction and all joint logits bit for bit.
Decision IDs are also compared with the retained row from the very graph that reduced them. These direct step probes
walk projected frames; they are not a trace of the normal TDT token trajectory.

Complete TDT comparisons use the actual switch off and on and compare IDs, frames, durations, graph counts and every
progress/cancellation callback. Cases include empty and one-frame inputs, zero duration, the max-symbol limit, blank
chains, a token extending past the final frame, cancellation after blank chains, repeated requests and a request after
boosting. Boosting with positive and zero alpha uses the full-logits route. The same runtime predicate used by decoding
is checked for CPU/other backends, exact model identity and boosting, so an ignored selector cannot pass as a device
speedup. Non-Haas and LSTM metadata controls exercise the selector only; their tensors are not presented as an LSTM
inference fixture. The ordinary RNN-T control retains its existing path.

`fastconformer-tdt-argmax-check <model.gguf> <device> <threads> [joint_encoded.npy...]` runs differential checks with
decoder-only model weights and optional real recorded encoder projections. It includes direct projected-input probes
and complete TDT comparisons with and without boosting. These C++ comparisons do not replace official NeMo/FLA stage
checks or an original-binary comparison. The check mode deliberately exercises the reduction helper on CPU too;
production TDT on CPU still uses the ordinary route.

Pending under an exclusive compute lease:

1. Build and run raw and synthetic checks on CPU and Metal, saving ordinary and adversarial results separately.
2. Compare the fixed original binary, this binary with the switch off, and this binary with it on. Run the existing
   official stages with their unchanged bounds, then Q4/Q8/F16 real-input decision and full/public result comparisons.
3. Confirm CPU-four-thread and boosted requests remain controls, and measure Metal decoding separately with alternating
   cold/warm runs. Record decoder/whole-request time, peak process memory and graph allocation. Do not sum gains from
   independent variants or include extra diagnostic reads in performance runs.
4. Reject the idea if the reduction overhead and four gets do not produce a repeatable gain. Review residual numerical
   contract limits only if the normal-path comparison and measured benefit make it an adoption candidate.
