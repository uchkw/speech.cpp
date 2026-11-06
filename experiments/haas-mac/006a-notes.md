# Trial 006a: Haas FFN-up SiLU epilogue

This rejected experimental prototype belongs to [issue 1](https://github.com/uchkw/speech.cpp/issues/1).
Its local checks pass within the scopes below, but the screen does not establish a stable warm request latency gain.
It is not part of the adopted runtime. This result does not reject other epilogue or GEMM optimizations.

## Source and boundary

- Parent base: `5b1b24a2471524daf9447c2126bf0bb6a1783546`, with the original runtime.
- ggml base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Cases first: `d26db9300e9e0790ecd5cdb13f35f66611e34c56`.
- Kernel prototype: `5b8aac4feaf70b8acd45df6a8b92538ae878127d`.
- Check correction: `85e2b88f847fd46bf05b06fabdaaea139ccb7db9`.
- Allocation-dependency correction: `0984bab5fc2e38e284a289d481576db3d76d80f4`.
- Protected in-place output case correction: `84e511117191c4999aca6981da6d1f7fb57116fc`.
- Independent original kernel with corrected checks: `72a0c44f16914cba1f73d50096d968edf0a6752d`, on
  `codex/haas-mac-006a-original-cases`, from the case-only commit plus the test-only correction.
  The protected in-place output correction advances that branch to `a9337c01bf6d27ae864f5cfb970599a62646ad2a`.
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

The shared case table has 89 cases: 48 eligible cases and 41 controls. Eligible cases cover all three types with
N=17,31,32,33,38,63,64,65,75,125,171,172,250,375, offset contiguous views and a following consumer.
Controls cover a second consumer of the product, a protected product output, strided and transposed activations,
batches, ReLU, N=1,8,16, M=4095, K=1280 and F32/BF16 weights. Three more controls place the unary output over an
expanded activation-input allocation: the separate operations can safely reuse it after the product finishes,
whereas the fused product must reject that placement after graph optimization.
The in-place SiLU cases are controls because `ggml_set_output` also marks their `view_src` product as an output.
The generic subgraph matcher preserves that product and rejects its elision. Haas uses an out-of-place SiLU.

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

## First direct-check result

The original kernel and the candidate with the switch disabled passed 89 cases and both poison guards. The candidate
with the switch enabled failed only the three in-place cases: each expected one fusion but correctly performed zero
because the product was marked as an output. All 89 input and output hashes matched the original kernel and both
candidate modes, all outputs were finite, and all allocation-dependency counts were zero. The initial failure logs
and binaries remain in `outputs/haas-mac-campaign/006a/`; they are not replaced by the corrected run.

The test-only correction classifies those three existing cases as controls and records each product's output flag.
It changes no runtime or numeric threshold and is applied identically to the original-kernel comparison branch.

## Completed checks

The corrected original, candidate-disabled and candidate-enabled direct checks each pass 89 cases and two poison
guards. All input and output hashes match across all three conditions. All outputs are finite, all added allocation
dependencies are zero, and only the 48 eligible cases fuse in the enabled condition. The 41 controls do not fuse.
Each in-place control confirms that the product is marked as an output.

The first CPU-reference run stops at the transpose control in the original CPU implementation: its matrix product
asserts that the activation's element stride equals its type size and does not support that transposed operand.
This is not a numerical failure of the new kernel. The original abort log remains. With the explicit filter
`-p 'variant=(?!transpose)'`, all 86 supported cases pass against the CPU reference in all three conditions:
48 eligible cases and 38 controls. The names of the three excluded transpose cases are recorded; they still pass
the original/disabled/enabled Metal differential comparison. The numeric bound remains unchanged.

The candidate also passes all 168 existing `MUL_MAT_ADD` cases against the CPU and the allocator topology check
on both CPU and Metal. The original sandbox Metal-initialization failure and the successful same-binary execution
outside that sandbox are kept separately.

Ordinary encoder graphs on all three types and the three stage inputs show 48 additional two-operation fusions and
48 fewer unary dispatches, with all other operation and fusion counts unchanged. Projected output bytes match
between disabled and enabled runs. This trace uses `GGML_METAL_GRAPH_DEBUG=1` alone, does not initialize fusion
statistics, and preserves the normal command-buffer policy. It is separate from latency measurement. The initial
parser looked for labels emitted only by fusion debug and therefore reported zero; its result remains beside the
corrected analysis of the existing graph-debug dispatch records. No runtime rerun or numeric-bound change is used
to correct that parser.

## Screen result

Measured on 2026-11-07 JST, Apple M2 Ultra with 76 GPU cores and 192 GB memory, Metal encoder and CPU decoder with
four threads. Builds use Release, native code and four build jobs. Every build or GPU run holds the shared performance
lock. The frozen runner is campaign commit `1dbc333010f5a544651cdef4637fc9b3ce39bf2e`, executed through
`/usr/bin/python3` (3.9.6). Its manifest records the resolved Python executable. The measured parent is
`e0b0bf2c0e25003013d3c26f01d0823e4f7f0d0e`, with ggml `84e511117191c4999aca6981da6d1f7fb57116fc`.

The same speech binary, SHA-256 `28f59b31b6599e63ad8b2deb12560b451e72c852d538527010fbc0d16ffc4aea`, runs with
the switch explicitly set to 0 or 1 in separate processes. The fixed anchor uses the original ggml kernel. The screen
contains nine duration-stratified inputs, three alternating rounds and nine timestamped quality requests per type.
Every process warms the input set before its measured requests.

| Type | Paired median request reduction | p50 ms, off to on | p90 ms, off to on | p99 ms, off to on | Encoder reduction on three inputs |
|---|---:|---:|---:|---:|---:|
| Q4_K | 0.53% | 80.07 to 77.79 | 125.78 to 125.98 | 146.05 to 146.80 | 1.05%, 0.75%, 1.18% |
| Q8_0 | 0.30% | 75.20 to 74.49 | 120.65 to 122.10 | 141.03 to 142.07 | 1.43%, 0.94%, 1.24% |
| F16 | 0.10% | 73.38 to 73.54 | 120.68 to 119.37 | 140.58 to 140.45 | 0.80%, 0.92%, 1.00% |

The three-round range of the nine-request time sum divided by its median is 0.62%/1.21% for Q4_K,
1.52%/1.43% for Q8_0 and 1.35%/0.50% for F16, disabled/enabled respectively. The fixed anchor's variation is
0.83%, 1.12% and 1.79%. The request differences are within this spread. Encoder reductions are about 0.18 to 0.59 ms;
they do not establish a stable reduction of the complete request. Nine-input p99 estimates do not describe a
production tail distribution. Load, first request and peak memory are recorded without claiming a gain.

For each type, all nine quality texts, public token text/times and segments match, and all 27 ordinary latency texts
match, both relative to the disabled path and to the fixed anchor. These public results do not expose raw TDT token
IDs or blank-duration decisions. The full 30-input/650-input comparison and short-prefix supplement are not run:
the screen is sufficient to reject this prototype for the current warm single-request workload.

Evidence is in `outputs/haas-mac-campaign/006a/`: `trial.json`, `checks.json`, `screen/summary.json`,
`corrected-checks/direct-comparison.json`, `corrected-checks/backend-reference-summary.json`, and
`encoder-trace/graph-dispatch-analysis.json`. Original failures, corrected runs, source snapshots and each build's
binaries and manifest are retained separately. `evidence-manifest.json` identifies the frozen evidence files.

Official-reference stage checks, Windows Vulkan and M5 execution are not performed for this prototype. Mac
differential checks do not replace them. A future no-switch production patch would require a new comparison with
the original runtime and the remaining quality and platform checks.
