# 011b — Fuse the gate projection and GLU epilogue

**Status: source-only final candidate — UNBUILT / UNMEASURED.** Source review is complete with
no remaining blocking finding, and the kernel is committed. Nothing has been compiled, run or
measured: compilation, runtime, math and performance checks are all **not run**. No numerical
check is claimed to have passed, and shader compilation is not claimed to work.

## Identity and revisions

- Trial 011b, independent of 011a; no measured result is transferred from any other trial.
- Speech base commit `114c514430c8aa1455c077819f8e72e674a48b75`.
- ggml base commit `e1a9ac12edbb57402618fd6936598603f5ee6266` — the base, not the final pin.
- Kernel commit `18d581ee87ae2830488d466b054581e94fea4ead` — the final ggml revision; the ggml
  gitlink in the final parent speech commit pins exactly this revision.
- Diagnostic cases commit `de13b1be95800eabfbadbc4a71d287a02d6758be` (final).

## Candidate

The ordinary (non-tensor) simdgroup `kernel_mul_mm` gains a `MUL_MAT+UNARY+MUL` epilogue on
its STORE: the gate projection `W_gate * x` is sigmoid-ed as F32 and multiplied by the factor
`a` inside the gate mat-mul's store.

- The epilogue elides the materialization of the gate-MM intermediate and of the sigmoid
  intermediate, and elides the two standalone elementwise dispatches (SIGMOID and MUL). The
  final MUL output is still materialized — written by the fused store.
- The separate factor projection `a = Wa * x` and its materialization remain: this is **not**
  a two-projection or concatenated-weight kernel, and it is more than 011a's standalone
  sigmoid/MUL fusion.
- Speech source site: `Encoder::convolution` builds `mul(pw1_a(h), sigmoid(pw1_gate(h)))` —
  the factor projection first, then the consecutive gate MUL_MAT, SIGMOID and MUL nodes the
  pattern matches (the `twoproj` fixture asserts this node order).
- Kernel source: the non-tensor simdgroup branch of
  `ggml/src/ggml-metal/kernels/mul_mm.metal` (`FC_mul_mm_glu` / `FC_mul_mm_glu_swap` function
  constants, an extra `factor` argument). Pattern and gate:
  `ggml/src/ggml-metal/ggml-metal-fusion.cpp` (`GGML_METAL_FUSION_MUL_MAT_GLU`).
- The epilogue is **not implemented on the tensor API/M5 path**: the Metal kernel's tensor
  branch is untouched, and the pattern is skipped on devices that report the tensor API.
  M5 and Vulkan are untested by this trial.
- No packing, dtype conversion, bias, new GEMM shape or altered multiplication loop; both MUL
  operand orders (`mul(sigmoid, a)` and `mul(a, sigmoid)`) remain supported.

## Selector `GGML_METAL_MM_GLU`

| Value | Effect |
| --- | --- |
| unset or literal `0` | candidate disabled (ordinary path) |
| literal `1` | candidate enabled |
| any other string, including empty | abort |

Read once and cached on the first candidate pattern check; each original / candidate-OFF /
candidate-ON side must run in a fresh process.

## Device gate

Requires `has_simdgroup_mm` and no tensor API. Target device: Mac M2 Ultra.

## Eligibility

A backend gate on the shape/type signature — model identity is not a gate.

- Gate weight: contiguous rank-2, F32 / F16 / Q8_0 / Q4_K, K = M = 1024.
- x, factor, MUL_MAT output, SIGMOID output and MUL output: F32 and contiguous, equal output
  shape.
- T (activation columns) in 17..384; T <= 16 takes the few-row MMA or mat-vec dispatch
  instead, and T >= 385 stays on the ordinary path.
- src[1] precision hint UNDEFINED: the gate conservatively requires the UNDEFINED hint for
  each of its four eligible weight types (F32, F16, Q8_0, Q4_K); an explicit F32 hint does
  not fuse.
- The actual family requests F32 accumulation for both projections; the encoder helper checks
  dtype, source hint and accumulation fields (`op_params`) separately.
- This is a deliberately gated backend optimization, not a new family graph option.

## Safety contract

Implemented and source-reviewed; **unexecuted**. Generic public-output, view and
extra-consumer protections are retained (the pattern is a regular elision chain, so the
generic chain, subgraph elidable-use and memory-range checks still apply). All activations
(x, factor, gate MM, sigmoid, MUL) must share one Metal buffer, while weights may be separate.

- FULL matching rejects the final output overlapping W or x, and rejects MM/sigmoid overlap
  with the factor.
- It rejects either elided MM/sigmoid output overlapping either W or x: the ordinary sequence
  has already overwritten those bytes by the time a later consumer runs, while the fused
  execution would skip the intermediate stores and expose the original mat-mul input.
- Only the exact final-in-place factor is permitted; partial overlap is rejected.
- Both Metal-buffer intervals and the underlying host addresses are checked, so multiple
  wrappers over one host arena cannot hide an alias.
- Rejecting aliases is conservative; the pattern is skipped in the allocation-dependency
  pass, so it adds zero allocation dependencies and does not extend input lifetimes.
- Implemented same-buffer negatives in the final cases commit, one Metal buffer with a
  trailing `add(mul, ·)` that reads the aliased input back after the product, in both MUL
  operand orders — none may fuse:
  - `raw_sigmoid_x`: sigmoid storage placed exactly over the activation;
  - `raw_sigmoid_x_partial`: sigmoid storage 16 bytes into the activation;
  - `raw_sigmoid_w`: sigmoid storage over the head of the F32 weight, read back through a
    legitimate view of the weight (no in-place matmul is constructed).
  The runner binds the whole triple — the weight included, for `raw_sigmoid_w` — into a
  single buffer object at page-aligned slots, so plain tensor identities carry the alias and
  no `view_src` is forged.
- Mapped-wrapper negatives (`mapped_out_x`, `mapped_sigmoid_x`, `mapped_mm_a`) bind the same
  host arena through two distinct Metal mappings; they must also stay unfused.

## Arithmetic and real limits

- Full tiles retain the original K loop, operand staging, accumulators, workgroup geometry
  and stores; the factor tile is loaded at the same simdgroup tile origin and leading
  dimension, pairing each thread element with the matching gate accumulator element.
- Tail tiles keep the existing scratch and barriers and apply the epilogue only to valid
  float4/scalar stores; the factor rows of a partial tile stay inside the valid tail.
- Sigmoid is the ordinary expression `1/(1+exp(-gate))` evaluated with an intended F32
  intermediate before a separately evaluated product; the multiply uses safe math and
  contract-off semantics in the fused MUL's original operand order. The full-tile path
  evaluates the sigmoid per scalar, while the existing standalone unary path operates on
  float4 values.
- Compiler math and rounding therefore require an actual hardware bit comparison of the
  full-tile scalar sigmoid against the standalone float4 unary result; source equivalence is
  not evidence of exact bits, and no such comparison has been run.
- Staging on this pinned backend: with default source precision, this ordinary large-MM path
  stages operands in half, including F32-weight × F32-activation storage; an explicit F32
  src[1] hint selects F32 operand staging only for the F32-weight × F32-activation kernel.
  The gate's UNDEFINED src[1] requirement is a conservative eligibility check applied to all
  four weight types, separate from this staging selection. No precision threshold is relaxed.
- Theoretical saving: 2 elementwise dispatches per eligible layer — 48 across the 24 layers —
  and 16 bytes per output element of device traffic (gate-MM write/read plus sigmoid
  write/read). Both GEMMs' FLOPs, the factor read and the final write remain. These are
  arithmetic counts, not measured performance, memory-bandwidth or allocator-peak savings;
  extra epilogue registers and factor loads may offset the benefit.

## Candidate and diagnostic source (read-only inputs)

| Path | Role |
| --- | --- |
| `src/families/fastconformer/encoder.cpp` | pinned speech source; the GLU site in `Encoder::convolution` |
| `ggml/src/ggml-metal/kernels/mul_mm.metal` | committed kernel: the non-tensor simdgroup store epilogue |
| `ggml/src/ggml-metal/ggml-metal-fusion.cpp` | committed pattern table and the `MUL_MAT+UNARY+MUL` gate |
| `ggml/tests/test-mm-glu-cases.h` | shared case table, graph fixture, deterministic initialization |
| `ggml/tests/test-metal-mm-glu.cpp` | Metal-only runner: counts, bit-exactness, poison, mapped and raw controls |
| `experiments/haas-mac/011b-glu-check.cpp` | encoder helper `haas-mm-glu-check` |

## Required validation (all not run; needs an exclusive compute lease)

1. Build original and candidate identically; use the same diagnostic source and the same
   synthetic values in separate fresh processes for original, candidate OFF and candidate ON;
   original vs OFF must be exact.
2. Direct Metal runner `test-metal-mm-glu [expect-fusion: 0|1] [output.bin]`: expected `0`
   for the original/OFF builds and `1` for candidate ON on eligible hardware; the per-case
   expected count is 1 only when the actual graph is eligible and 0 for every negative. It
   checks final exact bytes and finite values, intended fusion counts, zero allocation
   dependencies, original/OFF/ON binary snapshots, existing public intermediate retention,
   raw and mapped alias negatives, and poison controls.
3. `test-backend-ops` MUL_MAT_GLU: a numerical comparison of the final output plus explicitly
   OUTPUT-protected intermediates at its pinned 5e-4 NMSE tolerance; never compare stale
   elided intermediates. (The expected 0/1 fusion-count CLI belongs to `test-metal-mm-glu`,
   not to `test-backend-ops`.)
4. Encoder helper `haas-mm-glu-check model.gguf expected-eligible-fusions [output.bin]`
   (0 or 24), compiled identically against all three sides: one actual GLU per layer, 24
   normal fusions only for eligible T, zero retained-output fusions, and exact normal/retained
   projected output. Its T list covers 1, 8, 16, 17, 31, 32, 33, 63, 64, 65, 171, 172, 375,
   376, 384, 385. If the ordinary allocator yields fewer than the required 24 target fusions
   at an eligible frame length in the ON run, record that the intended production path was
   not reached and stop before performance measurement, keeping the guard and the expected
   count fixed; this stop rule is distinct from the retained-output runs, which expect zero
   fusions. The count helper is diagnostic and must not substitute for ordinary worker
   timing.
5. Official encoder thresholds unchanged: CPU F32 90 dB (an unchanged control, not a
   GPU-fusion pass) and Metal F32/F16 25 dB.
6. Recognition comparison with the same model variants/inputs on each side over the same 650
   inputs: raw tensors/argmax, raw TDT ids/frames/durations, full public
   text/tokens/times, and literal plus repository-normalized CER.
7. Observe the unchanged selector OFF control and the invalid-selector rejection.
8. Only after the numerical/path checks: ordinary non-instrumented paired
   latency/RTFx/load/peak-memory measurements, enough alternating trials to distinguish round
   noise; report p50/p90/p99.
9. Vulkan and M5 remain untested and unapproved. No adoption decision is made here. Raw
   artifacts later belong under `outputs/haas-mac-campaign/011b/`; publication and evaluation
   import belong to the coordinator.

## Execution state

No code, build, tests or timings were run; this trial is unbuilt and unmeasured.
