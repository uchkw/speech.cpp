# Trial 011a: fuse Conformer's sigmoid and multiplication

This is an unbuilt, unmeasured source checkpoint for [campaign issue 1](https://github.com/uchkw/speech.cpp/issues/1).
It starts from speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The ggml case-first commit is
`57af121df58bf5de8e8fa4987f507d3dd6a53d8f`; it contains no kernel changes and is also the original-kernel comparison
source. The kernel commit is `d219664b7accaedc8adc2c6c22ccc1742a42459f`. Both repository origins are the `uchkw` forks.
Neither repository has been pushed.

## Operation and scope

The pinned Metal fusion table and unary dispatcher have no sigmoid-plus-multiplication fusion. Existing
`test_unary_mul` cases also exercise other backends; their presence does not establish a Metal fusion.

Each Haas Conformer layer computes `a * sigmoid(gate)` after two independent pointwise projections. This is GLU,
not SiLU (`gate * sigmoid(gate)`) or SwiGLU. Trial 011a leaves the two projections and all speech.cpp inference source
unchanged. Projection packing belongs to a separate trial. One eligible fusion removes one dispatch per layer;
24 per encoder is a target to verify, not a measured fact.

`GGML_METAL_GLU=1` enables the experiment. An unset variable or `0` keeps it disabled; any other value aborts with the
accepted values. The process reads the switch once at its first matching pattern, so each setting uses a fresh
process. The new matcher requires a contiguous F32 sigmoid and same-shape F32 factor and result, `[1024,T,1,1]`,
with positive `T` and at most `INT32_MAX` elements. Broadcasting, other widths, types, ranks and unary operations are
excluded. The existing generic matcher protects a public sigmoid result and extra consumers.

The kernel evaluates the same float4 expression as the separate F32 sigmoid: `1 / (1 + exp(-gate))`. A small helper
with safe math and contraction disabled multiplies that F32 result in the original MUL operand order, so the
compiler must not fold the two operations into `a / (1 + exp(-gate))`. The sigmoid expression keeps its existing
math mode. Whether the compiler preserves exact bits on the current Metal compiler remains a required direct check.
Each thread loads four original gate and factor values before writing their four result indices; there is no shared
memory or barrier. The integer guard bounds the four-value group and the fixed width makes every group complete.

## Allocation and aliasing

The new raw pattern is excluded from allocation-dependency insertion. Disabled runs and unrelated unary/multiply
graphs retain the prior allocation dependencies. The optimizer may group then unpack the new sequence; a late FULL
rejection still executes both ordinary operations.

Same-shape, same-stride output aliases with the exact starting address of either original operand are allowed.
Every element reads its two inputs before storing that element. Partial overlap is rejected. The matcher also
rejects a sigmoid destination overlapping the other factor, because the separate sigmoid would overwrite the
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

## Pending execution

No build, CPU reference, Metal dispatch, performance run or Vulkan check has occurred. Source inspection and
`git diff --check` are the only completed implementation checks. Serialize every later build and execution with
`/private/tmp/speech-haas-mac-perf.lock`, and bound builds to `-j4`.

After the coordinator opens the slot, build the original case-first ggml and the candidate separately with Metal and
tests enabled. Run `test-backend-ops -o CONFORMER_GLU -b MTL0` on both, with candidate env `0` and `1`. Related
regressions include `SIGMOID_MUL`, `SILU_MUL`, `SIGMOID`, `MUL`, `SWIGLU` and existing unary/multiply layout controls.
Keep CPU F32 reference results apart from Metal strict-byte results. Then run, under the shared lock:

```sh
GGML_METAL_GLU=0 original/bin/test-metal-conformer-glu 0 outputs/original.bin
GGML_METAL_GLU=0 candidate/bin/test-metal-conformer-glu 0 outputs/off.bin
GGML_METAL_GLU=1 candidate/bin/test-metal-conformer-glu 1 outputs/on.bin
cmp outputs/original.bin outputs/off.bin
cmp outputs/original.bin outputs/on.bin
```

Build speech.cpp against each pinned ggml and compile `011a-glu-check.cpp` against each exact build's static
libraries, preserving source, compile command and binary hashes. The command is the campaign's standalone-check
link procedure with `src/common`, `src/families` and `ggml/include` on the include path, `libfastconformer.a`,
`libspeechcommon.a`, ggml's static libraries and the Metal/Accelerate/Foundation frameworks. For each of the existing
Q4_K, Q8_0 and F16 model files, run the original and candidate-off helper with expected count `0`, and candidate-on
with `24`, and compare their binary outputs. A count other than 24 is a finding about the normal path, not a reason
to replace it with an instrumented or fusion-debug count.

Full worker/ASR output equality, official stage-reference checks, alternating latency and memory measurements, and
Windows Vulkan validation remain separate pending work. An unchanged Vulkan kernel does not constitute a Vulkan
pass. The new checks must also be run there when a build and machine are available. No tolerance, adoption decision,
release, model file or public interface is changed by this prototype.
