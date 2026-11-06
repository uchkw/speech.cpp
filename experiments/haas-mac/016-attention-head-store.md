# Trial 016a: write attention products in merged-head order

Status: source preparation only. The prototype and checks are committed, but no build, GPU execution or timing has
run. This is an independent experiment, not an adopted speed-up. Tracking issue:
https://github.com/uchkw/speech.cpp/issues/1.

## Source boundary

- Parent runtime base: `114c514430c8aa1455c077819f8e72e674a48b75`.
- ggml runtime base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Cases-first checkpoint: `4eded218dd5ba641e3b9cf6ad61ac97222deb427`.
- Test-only backing-storage order correction: `617548f5b9307105b556d112d6b4c51fb0f29d85`.
- Kernel-only implementation: `acb259164e3404a8ab39fa3255fe9991897d8297`.
- Candidate with correctness-mode check registration: `9066e6f2471eabed8632ded32d4514b6429d9790`.
- Original kernel with identical checks: `40830de1f9b83695cce372a387763ef3600b9951`, branch
  `codex/haas-mac-016-original-checks`, worktree `outputs/worktrees/haas-mac-016-original-ggml`.
- Parent branch: `codex/exp-haas-016`; ggml branch: `codex/haas-mac-016-head-store`.
- Worktree: `outputs/worktrees/haas-mac-016`.
- ggml origin fetch: `https://github.com/uchkw/ggml.git`; push: `git@github.com:uchkw/ggml.git`.
- Only the coordinator pushes. Earlier campaign runtime changes are excluded.

Q/K reshapes and head permutations are views already; there is no copy to remove there. This trial targets only the
copy after the last attention matrix product. The separate copy that prepares V remains unchanged. Direct V stores
would be a separate trial because they meet the packed-QKV projection's layout. No GGUF, model weights, C API, CPU
or Vulkan code changes are proposed.

## Existing product and proposed store

`src/families/fastconformer/encoder.cpp:213` prepares contiguous V with shape `[T,128,8]`. The next product multiplies
it by contiguous attention weights `[T,T,8]`, producing `[128,T,8]`. Line 215 permutes that result to `[128,8,T]`,
copies it with CONT and reshapes the copy to `[1024,T]`. Haas has 24 such global-attention blocks.

The graph's matrix-product wrapper requests F32 accumulation but does not request F32 operands
(`src/common/graph.h:15`). On the current M2 path, these F32/F32 attention products stage both operands in half and
accumulate in float. Q4_K, Q8_0 and F16 model files all use those F32 intermediate tensors, so all three model types
would be affected. This does not specialize a quantized weight product.

The initial path is intentionally narrow: both operands and the output are F32 and contiguous, `M=128`,
`K=N=T>=64`, eight equal heads, one outer batch, no broadcast and default source precision. The ordinary SIMD MM
dispatch requires `K>=64` (`ggml/src/ggml-metal/ggml-metal-common.cpp:35`). Inputs with `T<64` remain controls, as do
few-row/MV products, explicit F32 source precision and the tensor-API path on M5-class devices. This trial does not
replace their kernels.

In `ggml/src/ggml-metal/kernels/mul_mm.metal:348`, the normal destination offset is

```
d + 128*t + 128*T*h
```

The fused destination is the existing CONT tensor, with offset

```
d + 128*h + 1024*t
```

Only the full-tile and tail-store addresses change: the row stride becomes `128*8` and the head offset becomes
`h*128`. Input loads, half conversion, K iteration, all eight float accumulators, 64-by-32 tiles, threadgroups and
shared-memory size stay unchanged. A function constant selects the new store, with a distinct pipeline cache key;
the disabled pipeline retains its old key. The tensor-API implementation remains unchanged. Compiled byte equality
must still be measured; source-level arithmetic identity alone does not prove it.

## Fusion and allocation boundary

Match three consecutive raw nodes, `MUL_MAT -> PERMUTE -> CONT`, and consume two nonempty nodes. Verify the exact
permutation, dimensions, contiguous layouts and view origin/offset. The final reshape is an unchanged view of CONT;
it need not be part of the matcher or a new operation. `ggml_can_fuse_subgraph_ext()` protects the omitted MM and
permutation against OUTPUT flags or additional consumers. CONT is the real output, so marking it or its final
reshape as OUTPUT, or reading CONT from additional downstream consumers, remains valid.

The matcher requires SIMD MM without tensor support in both STRUCTURAL and FULL modes. Device properties pass
through the existing fusion queries and graph optimizer; this does not import trial 006's fusion or kernel.
`GGML_METAL_HEAD_STORE=0|1` defaults to zero, is read once per process and rejects other values when the pattern is
examined. The runner owner has received the fixed name and value range; the execution snapshot is not yet selected.

Do not add allocator dependencies for this pattern. Full validation requires both MM inputs and CONT's destination
to belong to the same underlying Metal buffer, then rejects any destination overlap with either input. These are
all activations in the ordinary graph, so a same-buffer gate is simpler than extending the common overlap helper.
It also rejects distinct wrappers that map overlapping host addresses. Disjoint buffers are conservatively
unfused. Existing general fusion and memory checks remain unchanged.

This restriction may prevent useful fusion in the actual model. CONT is not an in-place operation, but after MM the
allocator can free V and attention weights, then reuse those ranges for CONT. V and the destination have the same
byte count. `ggml/src/ggml-alloc.c:632` allocates a new node from free ranges and lines 785 onward free parents after
their last use. Ordinary `Graph::compute()` allocates the raw graph without backend graph optimization
(`src/common/graph.cpp:96`). Consequently, adding an allocation dependency would change the experiment rather than
demonstrate that the store change works with the existing graph.

The first authorized device investigation must allocate the real encoder graph and record all 24 candidate input
and CONT ranges, buffer identities and overlap decisions. It must preserve the normal graph's OUTPUT flags and
order. Marking intermediate tensors as outputs changes lifetimes and can conceal the very alias under examination.
Report the actual eligible count per input; 24 is an upper bound, not a result. If all normal candidates overlap,
stop and report it before timing or changing lifetime policy. A different allocation policy would require a separate
proposal and measurement of its memory cost.

## Checks prepared before the kernel

One fixture is shared between `test-backend-ops` and the direct Metal comparison. It builds the real product, permutation,
CONT and final reshape. Declare only the true final output, except in explicitly protected-intermediate controls.

- Eligible time lengths: 64, 65, 95, 96, 127, 128, 129, 171 and 375. These cover complete and incomplete K steps and
  32-column stores. Add nonzero aligned source/destination offsets and a downstream consumer of CONT.
- Rejections: T=1,8,9,16,17,31,32,33,63; another head count or head width; `K!=N`; outer batch/broadcast;
  transposed/noncontiguous operands; another permutation; explicit F32 source precision; F16/Q8_0/Q4_K weight
  products; protected MM/permutation; another consumer of either omitted result; missing CONT; noncontiguous CONT.
- Allocation controls: exact and partial destination aliases with each input in one Metal buffer; distinct mapped
  wrappers over the same host range; and disjoint wrappers, which deliberately remain unfused. Map only cases whose
  ordinary unfused graph remains valid. Direct Metal checks own mapping cases; CPU graph copies do not preserve
  physical aliases across wrappers.
- Preserve every input before each direct run and poison every output region that does not overlap an active input.
  Restore aliasing input bytes, verify inputs, and require all result elements to differ from their previous contents
  before execution. Artificial wrong-destination and skipped-store controls check the sentinel mechanism.

The direct test compares complete canonical F32 output bytes for original-kernel, candidate-off and candidate-on
processes, with identical check sources and inputs. Same-binary off/on uses `memcmp`; complete binary result dumps
support the separate-build comparison. There are 52 direct cases: 14 eligible cases and 38 controls. The correctness
mode of `test-backend-ops -o ATTENTION_HEAD_STORE` registers 46 CPU-reference cases with the existing MUL_MAT NMSE
bound of 5e-4; five mapped cases and the transposed first operand are direct-only. The CPU implementation requires
that operand's reduction dimension to be contiguous. Two host-store poison guards are also prepared. A test-only
follow-up moved the registration from the performance list to the correctness list; no run preceded this correction.
Offset and strided output storage precedes its view in context creation order. Protected outputs and fusion counts
are checked separately from numerical equality. Zero newly inserted allocation dependencies and disabled/M5 gates
remain explicit validation requirements; M5 exclusion is presently source-reviewed only. No bound was relaxed after
a run, and none of the prepared checks has executed.

`016-encoder-layout.cpp` is a separate diagnostic source, linked against each recorded product build using that
build's timing-target flags and libraries. It marks only the real final joint-projected output, allocates once with
the ordinary allocator, and reports all 24 input/destination ranges. Its `allocate` mode skips encoder execution,
though building the graph can populate the model's existing position cache on the GPU. Range eligibility is not a
fusion count. Its separate `compute` mode preserves normal command-buffer policy and writes complete final projected
F32 bytes; it refuses to execute a T>=64 graph whose range-eligible count is zero. No new runtime diagnostics or
intermediate retention are introduced into speech.cpp.

## Deferred evaluation

1. Freeze source and runner snapshots before execution, retaining the original-kernel branch with identical check
   bytes. After execution permission, build with `-j4` and hold the shared performance lock for builds and device runs.
2. Run the ordinary allocator diagnostic first, with no added retention. If all normal candidates overlap or belong
   to different buffers, preserve that result and stop without loosening the gate or proceeding to timing.
3. If allocation permits fusion, check direct original/off/on bytes, fusion expectations, CPU references and related
   MUL_MAT/CONT/allocator cases. Observe actual normal encoder dispatches and compare final projected output bytes.
   Use graph-debug only in a separate diagnostic process; fusion statistics change command-buffer partitioning.
   Do not claim normal-command-buffer fusion counts from a statistics-only run.
4. Only if normal graphs use the fusion and correctness passes, run the warm alternating 9-input, 3-round screen
   with explicit CPU decode and four threads for Q4_K/Q8_0/F16. Use `/usr/bin/python3` and a frozen strict runner that
   records every `GGML_*`/`SPEECH_*` variable, binary SHA, source and model hashes. Include stage time, request latency,
   memory, output identity and the fixed anchor; no speedup estimate is asserted here.
5. Report the screen before requesting the full 30-input/650-quality and short6 supplement. A useful prototype needs
   a separate no-switch production implementation and a fresh original-runtime comparison. Official F32 reference,
   Windows/Vulkan and other Apple generations remain separate, unverified requirements.
