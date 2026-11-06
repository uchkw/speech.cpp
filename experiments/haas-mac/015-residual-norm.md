# 015: Residual addition fused with affine layer normalization

Status: source-only prototype. No build, shader compilation, model execution, timing,
or numerical result is available. The expected 24 fusions in an ordinary Haas
encoder graph have not been observed. This branch is not an adoption candidate.

The parent starts at `114c514430c8aa1455c077819f8e72e674a48b75`, with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The ggml cases are committed before the
kernel: `3613336f` adds the cases and `977f41d9` adds the partial-residual alias
case. Kernel commit `4a409be2ec3dc2c23070d57247f48d43bdbda664` implements the
prototype. Review adds only positive residual-input alias cases in candidate
`eb97e21752e680fa92f402cc80a84a4700e60cea`. The original-kernel comparison source
`8f5f4e2932255663d46f3fedbc180464b2db4ea6` contains the same test-only addition
on the cases-only base. Their test files compare byte for byte. No other
campaign optimization is included.

## Scope and arithmetic

`Encoder::feed_forward()` ends in `ADD(x, SCALE(down))` and `Encoder::build()`
immediately applies `out_norm` after `ff2`. The normalization is already a Metal
`NORM+MUL+ADD` fusion. This experiment adds only the preceding residual ADD.
The residuals before attention, convolution and the second feed-forward network
also feed their skip branches, so the existing consumer and output checks must
keep those values. The 24 final block residuals are structural candidates, not a
measured fusion count.

`GGML_METAL_RESIDUAL_NORM=0|1` defaults to zero and rejects other values. The new
pattern requires an M2 Ultra without the tensor API, contiguous F32 activations
of shape `[1024,T,1,1]`, two identically shaped residual inputs, and contiguous
F32 weight and bias vectors of 1024 elements. The model layout stores both
`out_norm` vectors in F32 in Q4_K, Q8_0 and F16 files. No RMS_NORM kernel changes.

The old NORM template, its pipeline selection and its dispatch function remain
textually unchanged. A separate float4 kernel repeats its statements:

1. Accumulate each lane's four values, then `dot`, `simd_sum`, threadgroup
   scratch and another `simd_sum`; divide by 1024 for the mean.
2. Store the centered float4 in the destination; accumulate its squared values
   with the same `dot`, SIMD and threadgroup reductions; divide for the variance.
3. Compute the same `1/sqrt(variance+eps)`, then `(y*scale)*weight+bias`.

Only the source loads in steps 1 and 2 become F32 residual sums. Their small
helper uses `METAL fp math_mode(safe)` and `contract(off)` to keep the separate
ADD's rounding boundary. These pragmas are scoped to that helper. The remaining
arithmetic, including the existing affine expression's compiler contraction
policy, is unchanged. This is a source-level argument; emitted shader arithmetic
and bitwise equality still require execution. No tolerance will be enlarged to
make the prototype pass.

The original pipeline selects `_4` for width 1024. Its `ne00_t` is 256; the
dispatch loop doubles 32 to 64, 128 and 256, then caps by the pipeline maximum
and the 256-element row. The new kernel dispatches exactly 256 threads, or eight
SIMDgroups, only when both original and new pipeline limits are at least 256.
If either limit is lower, this pattern is not dispatched. Actual limits have
not been queried. Both kernels use 32 floats of threadgroup scratch.

This can remove one residual ADD dispatch per eligible block. It recomputes
the residual sum on both read passes; it does not change the normalization
reductions or claim a whole-request gain.

## Allocation and aliases

The raw allocation-dependency visitor always excludes the new pattern. Its
device/environment gate also applies in structural matching, so OFF and
unsupported devices retain the existing allocation dependencies and graph
ordering. The common overlap helper is unchanged.

FULL matching requires the two residual inputs and all four operation tensors
to occupy the same underlying Metal buffer. Each tensor is float4 aligned.
Residual and final outputs may exactly alias a residual input with the same
address and layout; partial overlaps are rejected. The final output also
rejects a partial overlap with the residual ADD output, which would disable
the original three-operation NORM fusion. Output flags and extra consumers of
any elided intermediate reject the new pattern.

The mean reduction and its barriers finish before any destination write. During
centering, each lane reads both operands of its own float4 before replacing that
same vector. Contiguous equal-shaped rows make each row independent, and the
last pass reads the centered destination rather than either residual input.
Therefore exact input aliases cannot destroy another lane's or row's pending
input. Weight and bias must not overlap any activation. The local check keeps
the existing Metal-buffer range test and additionally compares host ranges
only when both buffers are shared. Private buffers use synthetic addresses,
so those addresses are never compared as host memory.

## Prepared checks and pending execution

`test-backend-ops -o RESIDUAL_NORM` has 71 backend-reference cases. Shapes include
T=1,31,32,33,100,171,172,375, with asymmetric inputs, half-ULP residual addends,
large finite values and nearly constant rows. Rejection cases cover other
widths, row strides, broadcast residuals, per-row weights, F16 residuals,
retained or reused intermediates, partial aliases and separate shared wrappers.
Positive physical layouts include the residual ADD overwriting A or B, with
the final affine output either separate or sharing the same input, at T=1,33,375.

The serial Metal helper runs each case with two deterministic input generations
and OFF/ON: 284 planned executions. It poisons intermediate/output storage
before every execution and restores all four inputs afterward. Exact input
aliases necessarily restore finite output bytes as well, so those cases also
require the result to differ from the restored input. Each result must be finite,
bitwise identical to OFF, and have the expected new-fusion count. Physical alias
and wrapper layouts are applied in this direct helper; the ordinary CPU-reference
cases compare their logical graphs without those forced placements.

For the original cases-only executable, set `GGML_TEST_RESIDUAL_NORM_EXPECT=0`;
for the candidate set it to `1`. Set `GGML_TEST_RESIDUAL_NORM_DUMP` to distinct
new files and compare the complete original/candidate dumps byte for byte.
The original executable ignores the experimental switch and uses its unchanged
kernels in both helper passes. Existing ADD, NORM and affine-normalization
regression cases also remain required. Fusion statistics in the direct helper
change command-buffer policy, so those runs do not prove ordinary graph behavior
and must not be used as latency measurements.

`015-encoder-trace.cpp` is the ordinary-graph helper, preserved byte for byte
from the 007 diagnostic. Build it against each exact product build's libraries
and flags. Use `GGML_METAL_GRAPH_DEBUG=1`, without fusion debug, disabling generic
fusion or initializing fusion statistics. It marks only the final projected
encoder tensor, uses normal `Graph::compute()`, and saves its finite F32 bytes.
Compare original/OFF/ON outputs and dispatch logs, and verify the expected 24
new fusions at the normal command-buffer count. Retained intermediate checks
are separate diagnostics because their allocation and fusion behavior differ.

After these checks pass, measure Q4_K, Q8_0 and F16 with the frozen common runner,
CPU decoding with four threads, all-clip warmup, alternating rounds and the
fixed campaign anchor. Record load, first request and peak memory separately.
Start with the nine-clip screen; broader timing and quality runs need evidence
of a stable request benefit. Use the shared performance lock and build with
`-j4`. No such execution is authorized by this source checkpoint.

Official Haas F32 stage dumps and Vulkan validation are unavailable. M5 and
other devices are outside the experimental gate and unmeasured. These are not
passes. No ADR or production behavior is settled by this prototype.

Static checks performed: `git diff --check`; all new fusion call signatures and
call sites reviewed; unchanged original NORM template/pipeline/dispatch bodies
compared with e1a9; specialized reduction and affine statements compared with
the original F=3 body after substituting its two residual loads. No compilation
or runtime check is represented by these source comparisons.
