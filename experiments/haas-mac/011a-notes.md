# Trial 011a: fuse Conformer's sigmoid and multiplication

This GLU-only prototype was rejected after the Mac screen for
[campaign issue 1](https://github.com/uchkw/speech.cpp/issues/1). The correctness checks passed, but the latency
gain did not exceed round variation. [The evaluation](011a-results.md) records the measurements and limits.
The broader pointwise-projection-plus-GLU measure remains unfinished.
It starts from speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The initial ggml case-first commit is
`57af121df58bf5de8e8fa4987f507d3dd6a53d8f`, followed by kernel commit
`d219664b7accaedc8adc2c6c22ccc1742a42459f`. Additional mapped-buffer controls are in case-first commit
`c1f0bfedf63fc4c577d99fc9cdc8da36dc47cce9`, followed by the single-buffer gate in
`232e507d0c35312b77a57720a4c7095b3c77ff5e`, the current candidate pin. The original-kernel comparison must add the
test-only `c1f0bfed` change to `57af121d` so its fixture includes the same mapped-buffer controls. That control is
ggml `caa352ff64190598e6661a383c4768110dc47fd9`, pinned by speech.cpp
`d8971602369b9fe4478d52e797bde48d718729cb`. Both repository origins are the `uchkw` forks; the coordinator has
pushed the candidate and control branches. Evaluation used speech.cpp `54dd7a5df9b3646b03581c9b4cd27cb3a3b55600`
without a runtime-source change.

## Operation and scope

The pinned Metal fusion table and unary dispatcher have no sigmoid-plus-multiplication fusion. Existing
`test_unary_mul` cases also exercise other backends; their presence does not establish a Metal fusion.

Each Haas Conformer layer computes `a * sigmoid(gate)` after two independent pointwise projections. This is GLU,
not SiLU (`gate * sigmoid(gate)`) or SwiGLU. Trial 011a leaves the two projections and all speech.cpp inference source
unchanged. Projection packing belongs to a separate trial. One eligible fusion removes one dispatch per layer;
24 per normal encoder was verified for each of seven lengths and all three model types, separately from timing.

`GGML_METAL_GLU=1` enables the experiment. An unset variable or `0` keeps it disabled; any other value aborts with the
accepted values. The process reads the switch once at its first matching pattern, so each setting uses a fresh
process. The new matcher requires a contiguous F32 sigmoid and same-shape F32 factor and result, `[1024,T,1,1]`,
with positive `T` and at most `INT32_MAX` elements. Broadcasting, other widths, types, ranks and unary operations are
excluded. The existing generic matcher protects a public sigmoid result and extra consumers.

The kernel evaluates the same float4 expression as the separate F32 sigmoid: `1 / (1 + exp(-gate))`. A small helper
with safe math and contraction disabled multiplies that F32 result in the original MUL operand order, so the
compiler must not fold the two operations into `a / (1 + exp(-gate))`. The sigmoid expression keeps its existing
math mode. Direct original/off/on Metal output files were byte-identical on the tested M2 Ultra and compiler.
Each thread loads four original gate and factor values before writing their four result indices; there is no shared
memory or barrier. The integer guard bounds the four-value group and the fixed width makes every group complete.

## Allocation and aliasing

The new raw pattern is excluded from allocation-dependency insertion. Disabled runs and unrelated unary/multiply
graphs retain the prior allocation dependencies. The optimizer may group then unpack the new sequence; a late FULL
rejection still executes both ordinary operations.

The FULL matcher requires the gate, factor, sigmoid result and product to share one underlying MTLBuffer. It uses
the existing identity helper for three comparisons; common overlap logic and the structural matcher are unchanged.
Distinct Metal wrappers can map overlapping host memory, which the common same-buffer range check does not see.
Such wrappers are therefore outside this experiment even when their tensor ranges do not overlap. Twelve mapped
controls verified zero fusions and original/off/on output identity. The pre-gate kernel was not run to reproduce
its missing guard. The normal Haas graph arena remains the intended target; the separate normal-graph check
verified 24 fusions.

Within that buffer, same-shape, same-stride output aliases with the exact starting address of either original
operand are allowed. Every element reads its two inputs before storing that element. Partial overlap is rejected.
The matcher also rejects a sigmoid destination overlapping the other factor, because the separate sigmoid would overwrite the
factor before MUL. A partial overlap of the sigmoid result with its own gate is rejected as well. A legitimate
`a == gate` graph has its own case. No allocator lifetime extension is used to force fusion.

## Cases prepared before the kernel

`test-backend-ops -o CONFORMER_GLU` uses whole graphs and the CPU reference. The shared 64-case fixture has 38 expected
fusion cases and 26 controls, with both MUL operand orders. It covers `T=1,31,32,33,171,172,375`, distinct asymmetric
inputs, sigmoid saturation, signed zero, large finite factors, offsets, a following consumer, identical operands,
exact output aliases, a partial gate overlap, a sigmoid overwriting the other factor, a public intermediate,
multiple consumers, F16, neighboring widths, strided inputs, broadcasting, a batch dimension and SiLU controls.
The existing CPU-reference NMSE bounds of the unary/multiply checks are retained: F32 `1e-7`, F16 `5e-5`.
They are not the acceptance bound for the original/off/on Metal comparison, which requires identical bytes.

The direct Metal tool adds twelve controls, separate from those 64 CPU-reference cases, for 76 total comparisons
(38 expected fusion cases and 38 controls). A page-aligned host arena has two distinct Metal mappings. In each MUL
operand order, four controls bind only the gate, factor, sigmoid result or product to the second wrapper, with
otherwise disjoint tensor ranges. Two more controls bind the sigmoid to the factor's exact host address or the
product 16 bytes into the gate, through the other wrapper. They assert the wrapper and host-address arrangements,
require zero fusions and contribute their output bytes to original/off/on comparisons. Both wrappers outlive execution and are released
before the host arena. These controls exercise Metal mapping APIs and are not Vulkan or CPU-reference coverage.

`test-metal-conformer-glu` uses the same input buffers for its fusion-disabled and fusion-enabled executions. It
calls the real optimizer and requires zero new allocation dependencies for these minimal graphs. Before each run
it writes deterministic inputs and NaN-poisons output regions outside the active input ranges. It verifies that
poisoning does not change either input. Exact aliases cannot be poisoned without changing their inputs, so those
bytes remain the restored original values; the log reports their count. The partial alias retains its poisoned tail.
Independent emulated wrong-destination and missing-last-store cases show that stale output would pass without
poisoning and fail with it. Neither guard is a GPU fault injection.

The tool checks strict output bytes, finiteness, protected sigmoid output and the actual `UNARY+MUL` fusion count.
An optional binary records every final result, including lengths, for direct original/candidate-off/candidate-on
file comparisons. A matching hash or CPU-reference result does not substitute for those comparisons.

## Normal encoder graph evidence

`011a-glu-check.cpp` builds the actual unretained `encoder.build()` followed by `joint.project_encoder()`, with only
the final projection retained, as `Recognizer::encode()` does. It locates each GLU by the two original projection
weight operands and counts fusions after computing that normal graph. A separate instrumented run retains each
sigmoid and GLU output and must report zero such fusions. Their final projected outputs must match exactly.
The seven lengths above exercise both paths. The optional binary contains the normal outputs and the retained
intermediate values for independent original/off/on comparison. Instrumented counts are never evidence of the
normal graph's dispatch reduction; fusion debug modes that change command-buffer partitioning are not used.
The helper rejects fusion-debug and generic-fusion-disable environment overrides. It initializes counters only after
the sole backend exists and makes no later backend or command-buffer setter call. In the pinned backend, `n_cb=1`
encodes the main-thread prefix before its one remaining worker, so counters are not incremented concurrently and
the ordinary two-command-buffer boundary is retained. Counter initialization before backend construction would
force one command buffer and is deliberately avoided. A graph-debug trace without counters remains an independent
way to confirm the dispatches; counter collection is not a performance measurement.

## Execution and remaining limits

The clean original and candidate builds, CPU-reference Metal cases, direct Metal comparisons, normal encoder checks,
official frontend/encoder checks and worker screen completed under `/private/tmp/speech-haas-mac-perf.lock`, with
builds bounded to `-j4`. The result files and original logs are under `outputs/haas-mac-campaign/011a/`.

The original comparison includes both test-only changes above. The intermediate `c1f0bfed` source retains the
pre-gate kernel; no failure on that intermediate source is claimed. `CONFORMER_GLU` passed 64 CPU-reference cases
on original, candidate-off and candidate-on Metal. Related regressions passed 158 supported cases per side, with
the same 111 unsupported cases per side excluded from the pass count. Direct comparisons used:

```sh
GGML_METAL_GLU=0 original/bin/test-metal-conformer-glu 0 outputs/original.bin
GGML_METAL_GLU=0 candidate/bin/test-metal-conformer-glu 0 outputs/off.bin
GGML_METAL_GLU=1 candidate/bin/test-metal-conformer-glu 1 outputs/on.bin
cmp outputs/original.bin outputs/off.bin
cmp outputs/original.bin outputs/on.bin
```

The normal helper was compiled against each exact build's static libraries, preserving source, compile command
and binary hashes. The command is the campaign's standalone-check
link procedure with `src/common`, `src/families` and `ggml/include` on the include path, `libfastconformer.a`,
`libspeechcommon.a`, ggml's static libraries and the Metal/Accelerate/Foundation frameworks. For each of the existing
Q4_K, Q8_0 and F16 model files, the original and candidate-off helper reported `0`, candidate-on reported `24`,
and retained graphs reported `0`. Their binary outputs matched across original/off/on for all seven lengths.

An independent graph-debug capture initially failed to capture node logs because speech.cpp's default sink filters
Debug messages. The failed capture is preserved. A separate diagnostic-only launcher enabled that log level for
the unchanged stage tool and confirmed 24 fusions per normal encoder. Its times, flags and binary were excluded
from the worker screen; `GGML_METAL_FUSION_DEBUG` was never used.

The nine-input worker screen kept public text, token text/times and segments exact, including text from every
latency round. Full 650-input and sub-6-second follow-up were not run after the screen rejection. Windows Vulkan
execution and official KDA/full-TDT validation are unavailable. No tolerance, release, model file or public
interface is changed by this prototype.
