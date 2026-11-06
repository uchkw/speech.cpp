# Trial 006a: Haas FFN-up SiLU epilogue

This is an unbuilt experimental source checkpoint for [issue 1](https://github.com/uchkw/speech.cpp/issues/1).
It does not establish correctness or a speed-up and is not part of the adopted runtime.

## Source and boundary

- Parent base: `5b1b24a2471524daf9447c2126bf0bb6a1783546`, with the original runtime.
- ggml base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Cases first: `d26db9300e9e0790ecd5cdb13f35f66611e34c56`.
- Kernel prototype: `5b8aac4feaf70b8acd45df6a8b92538ae878127d`.
- Check correction: `85e2b88f847fd46bf05b06fabdaaea139ccb7db9`.
- Allocation-dependency correction: `0984bab5fc2e38e284a289d481576db3d76d80f4`.
- Independent original kernel with corrected checks: `72a0c44f16914cba1f73d50096d968edf0a6752d`, on
  `codex/haas-mac-006a-original-cases`, from the case-only commit plus the test-only correction.
- Parent branch: `codex/exp-haas-006a`; ggml branch: `codex/haas-mac-006a-silu`.
- Both remotes belong to `uchkw`; the coordinator alone pushes them, ggml first.

`GGML_METAL_MM_SILU=0|1` is read once per process. Unset means `0`; another value is rejected when the pattern
is checked. A/B workers must start in separate processes with the value explicitly recorded. A future adopted
implementation would remove the switch and require a new comparison with the original runtime.

The prototype fuses `MUL_MAT -> SiLU` only for contiguous two-dimensional F16, Q8_0 or Q4_K weights with
K=1024, M=4096, F32 activations and N>16. These are the two bias-free FFN-up products in each of Haas's 24 encoder
layers. The new epilogue reads the existing F32 accumulator elements and applies the existing unary expression
`x / (1 + exp(-x))` before their existing store. It changes neither operands, dequantization, K accumulation order,
64-by-32 tile, shared memory size nor output layout. It adds no bias, residual, scale or approximation.

The optimizer and encoder consult the same capability gate: SIMD-group matrix multiplication is available and
the tensor API is not selected. The M5 tensor shader is unchanged and unmeasured. The few-row MMA and MV paths,
other matrix dimensions, batched or strided inputs, other types and other unary operations retain their paths.
Products marked as outputs or read by another consumer cannot be elided. A full match also rejects an output
that overlaps either external matrix operand. The generic fusion dependency and memory checks still apply.
This new pattern adds no allocation dependencies: the raw-pattern dependency pass skips it, so disabled experiments,
unrelated shapes or unary operations and M5 keep their original input lifetimes. Graph optimization unpacks every
temporarily grouped node after reordering. If allocation then makes an output overlap an input, the full matcher
rejects fusion and the encoder runs the original product and unary nodes separately.

The adopted fork's decision table remains unchanged: this branch is an experiment, not a measured addition to the
fork's `speech.cpp` branch. No earlier tile, prepacking, resident-state, graph or QKV trial is included.

## Checks prepared

The shared case table has 89 cases: 51 eligible cases and 38 controls. Eligible cases cover all three types with
N=17,31,32,33,38,63,64,65,75,125,171,172,250,375, offset contiguous views, an in-place SiLU and a following consumer.
Controls cover a second consumer of the product, a protected product output, strided and transposed activations,
batches, ReLU, N=1,8,16, M=4095, K=1280 and F32/BF16 weights. Three more controls place the unary output over an
expanded activation-input allocation: the separate operations can safely reuse it after the product finishes,
whereas the fused product must reject that placement after graph optimization.

`test-backend-ops -o MUL_MAT_SILU -b MTL0` compares the complete graph with the CPU using the existing
matrix-product NMSE bound, with the switch explicitly set to each value. It does not compare with official model
dumps. `test-metal-mm-silu` uses deterministic stored weights and activations, runs the same buffers with generic
Metal fusion disabled and enabled, and requires finite outputs, bitwise equality and the exact `MUL_MAT+UNARY`
fusion count. It poisons product, activation and final-output storage with NaN before each execution, rejecting
unexpected overlap with either source input and allowing output-only aliases. The explicit input-alias control
restores the same active input prefix each time, preserves it during poisoning and poisons the remaining output.
Two four-value buffer-write guards emulate a wrong destination and a missing final store: they require the stale
output to hide the defect without poisoning and a non-finite result to expose it with poisoning. These guards do
not dispatch a GPU kernel. The helper calls the real Metal graph optimizer before allocation and requires zero
added allocation dependencies for these graphs. It also checks the protected product itself for finite, identical
values. Use `GGML_METAL_MM_SILU=0 test-metal-mm-silu 0`, then
`GGML_METAL_MM_SILU=1 test-metal-mm-silu 1` on this M2 Ultra. The independent original-kernel branch has identical
check files and expects zero fusions; compare its emitted input/output hashes with the candidate's disabled and
enabled runs. M5 expects zero, not one.

The direct helper is a correctness check, not a performance benchmark. Repeating the same pair of graph pointers
would give the product extra uses and reject fusion; that construction cannot measure this epilogue. Product timings
must preserve a single consumer and prove the actual fusion count. The encoder diagnostic should observe 48 new
fusions for an eligible Haas request before attributing its timing to this kernel.

## Evaluation pending

No configure, compilation, backend execution or GPU measurement has run at this checkpoint. Wait until trials
004 and 005 finish. Build with four jobs and hold the shared performance lock for every build or GPU run.
Retain the original case-only and candidate binaries, source commits, environments and SHA-256 hashes separately.

After the backend checks pass, compare the same candidate binary with switch 0 and 1 on Q4_K, Q8_0 and F16,
using explicit CPU decoder threads=4, warm alternating rounds and the common nine-input screen. Record encoder
stage latency separately from the worker's complete request. A promising result needs the full 30-input latency
and 650-input quality comparison with the fixed anchor, followed by the separate six-prefix supplement where
useful. Use `/usr/bin/python3` and the common runner's recorded timer. Do not treat timing within round variation
as a gain. A no-switch production patch requires its own original-runtime comparison before adoption.

Official-reference stage tensors, Windows Vulkan and M5 execution remain unavailable or untested. Mac differential
checks do not replace those requirements. The experiment's result, including a rejection, must be preserved before
another epilogue variant is tried.
