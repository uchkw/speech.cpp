# Trial 007a: relative-position ADD and softmax fusion

This is an unbuilt source checkpoint for [issue 1](https://github.com/uchkw/speech.cpp/issues/1).
It establishes neither numerical equivalence nor a speed-up. It is not an adopted runtime change.

## Source

- speech.cpp base: `114c514430c8aa1455c077819f8e72e674a48b75`, with the original campaign runtime.
- ggml base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Cases first: `c0118325707483b9a7a3b7935e981e7eac01e59b`.
- Deterministic original-kernel comparison: `15c05bd816a904b0cf001921e4ca147643ed1a41`.
- Kernel prototype: `c17f151a0915630c95b70ce3ced39cf2b842741d`.
- Alias cases and aligned controls: `4b448b7d` and `2925ae93`, before the alias runtime change.
- Exact-ac alias support: `f4832b21`.
- Parent branch: `codex/exp-haas-007a`; ggml branch: `codex/exp-haas-007a-metal`.
- Original-kernel branch: `codex/exp-haas-007a-original-cases`, at `0cea543c`, with identical check sources and no
  runtime changes. The original stricter prototype remains preserved by its earlier commit and parent `cbf59a9`.

No earlier runtime trial is included. The adopted fork's decision table is unchanged because this is an
unmeasured experiment. The coordinator owns publication, with ggml published before its parent gitlink.

## Operation and bounds

FastConformer's global attention computes two separate products, `ac` and `bd`. The relative-position view is
`[T,T,heads]` F32, with `nb1=(2*T-2)*4`, `nb2=T*(2*T-1)*4` and byte offset `(T-1)*4`. The trial preserves both
products and fuses only their addition and the following unmasked, sink-free softmax. It does not use flash
attention or change attention scope, positions, scores, precision or decoder behavior.

`GGML_METAL_ADD_SOFTMAX=0|1` is an experimental switch; unset means `0` and another value aborts. It is read
when matching the pattern. Worker processes keep their recorded value fixed. The direct correctness check changes
it only after the parallel test workers have joined and each device execution has completed.

The new pattern requires M2 Ultra with the tensor API disabled. It rejects other devices, tensor-API devices,
non-F32 operands, broadcast or nonsquare shapes, batches, a view as the first addend, another relative-view
stride or offset, a mask, sinks and an in-place softmax. The generic matcher also rejects a protected ADD output
or a second consumer. Graph grouping and dispatch use the same environment and device gate. The raw allocation
dependency pass always skips this pattern, so it adds no input lifetimes; the full matcher rejects a softmax
output that partially overlaps `ac` or overlaps `bd` after allocation. Exact `ac` reuse is permitted only when
address, type, shape and all strides are identical. No Vulkan or matrix-product kernel is changed.

The graph allocator can reuse `ac` for ADD and then for softmax, since both operations permit in-place allocation.
The fused kernel supports that exact reuse: the maximum pass reads the complete row before `simd_max`, and any
additional SIMD groups synchronize through the existing threadgroup barriers. The exponential/sum pass assigns
each scalar or float4 element to one lane; that lane reads its input before replacing the same element, and no
other lane needs it. Normalization reads only the destination. Distinct rows occupy disjoint intervals because
`ac` and the output share the same contiguous layout. This argument does not permit shifted addresses or `bd`
aliasing, where a write could affect another row's pending input.

The fused load adds two F32 values under local `#pragma METAL fp math_mode(safe)` and `contract(off)` before
the existing scale expression. The rest of the softmax retains its compiler mode, exponent, scalar or float4
selection, thread count, maximum and sum reductions, barriers and normalization. The float4 path reads four
scalar values from `bd`, whose offset and row stride need only four-byte alignment. The ordinary specialization
has no added arithmetic, but equivalence to the independently built original kernel still requires measurement.

The local pragma follows [Apple's Metal Shading Language Specification, section 1.6.3](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
dated 2026-06-04. It prevents reassociation of the new sum without changing the softmax's math-function mode.
The downloaded source has SHA-256 `41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5`.
Compiler acceptance and emitted arithmetic have not yet been checked. Non-finite inputs are not an equivalence
claim: the existing fast-math path does not define their general behavior.

## Prepared checks

`test-backend-ops -o ADD_SOFT_MAX_RELPOS` contains 44 cases. Fourteen use `T=1,31,32,33,171,172,375`, eight heads
and two asymmetric input sets, including logits near positive and negative 10000. Fourteen controls cover
contiguous or broadcast `bd`, another row stride, padded or viewed `ac`, exchanged operands, rectangular scores,
F16 addition followed by an F32 cast, masks, sinks, in-place softmax, output marking, a second consumer and a
following consumer. The following consumer remains eligible; the other controls must record no new fusion.
Fourteen additional cases place the output exactly over `ac`, at all seven sizes with both input ranges. Two
negative controls place it sixteen bytes into `ac` or over the start of `bd`; both retain valid float4 output
alignment and require zero new fusions. These allocation aliases are installed by the direct Metal check.

The normal whole-graph check compares with the CPU using the existing softmax NMSE bound of `1e-6`. The M2 Ultra
direct check runs two deterministic generations, env 0 then 1, for 176 executions. In separate allocations it
poisons every owned computed tensor with NaN before each execution. For aliased allocations it restores the same
input bytes after poisoning, so the input portion of the output cannot remain poisoned; it additionally requires
the final output to differ from the restored pre-execution bytes. The two roles are distinct. Every case checks
finite final and protected outputs, byte equality between paths and the exact `ADD+SOFT_MAX` count. The model
comparison must separately confirm that the normal allocator actually permits
the fusion, with 24 new fusions expected for an eligible 24-layer Haas encoder.

The same test-only sources are retained with the original ggml kernel. After build authorization, run the original
with `GGML_TEST_ADD_SOFTMAX_EXPECT=0`, and the candidate with `GGML_TEST_ADD_SOFTMAX_EXPECT=1`. Set
`GGML_TEST_ADD_SOFTMAX_DUMP` to a new file under `outputs/` for each run. The dump contains framed case names,
input bytes and output bytes for both switch values. Require the two complete files to match byte for byte and
save their SHA-256 hashes. An existing dump is refused. Leave the generic Metal-fusion-disable and tensor-enable
settings unset. CPU-only and Vulkan runs do not execute this Metal-specific direct comparison.

## Pending evaluation

Only source inspection and `git diff --check` have run. No configure, build, backend execution or timing has run.
Wait for the coordinator after trials 005 and 006. Every build and device execution holds
`/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock`; builds use four jobs.

First build the original and candidate ggml checks and compare CPU, Metal, actual fusion counts and the original
output dump. Verify the prototype's disabled path against the fixed original-runtime anchor. If those checks pass,
use the frozen common runner and `/usr/bin/python3` for Q4_K, Q8_0 and F16 with CPU decoder threads=4, the same
nine-input screen, complete-input warmup and three alternating rounds. Keep load, first request, peak memory,
encoder timing and warm complete-request latency separate. A promising screen alone cannot support adoption;
continue to 30 inputs, exact 650-input quality, the anchor and the separate short-input supplement only if warranted.

Official stage tensors, Windows Vulkan and M5 execution remain unavailable or untested. A bitwise differential
check cannot substitute for an official reference check. No result may be recorded as passed before it is run.
