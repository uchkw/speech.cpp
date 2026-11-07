# Trial 016a: write attention products in merged-head order

Status: complete and rejected; the default stays off (`GGML_METAL_HEAD_STORE=0`, unset=0) and nothing is adopted.
The corrected checks, related regressions, normal encoder execution, official F32 stages and the worker runs all
completed. The Q4_K screen (+2.786% paired median, 9/9 inputs faster, all three rounds improving) justified only the
Q4_K full run, which measured +0.316% on 17/30 inputs with ON p90/p99 above OFF: insufficient full whole-request
benefit, so the prototype is rejected. Q8_0 and F16 remain screen-only and short6 was not run. Final numbers and
decision: `experiments/haas-mac/016-attention-head-store-evaluation.md` with machine-readable companion
`experiments/haas-mac/016-attention-head-store-evaluation.json`. This is an independent experiment, not an adopted
speed-up, and its results are never combined with the separate quantization study. Tracking issue:
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
- Initial test parent runtime: `efa6a4ea491a576198a1f0dc2a7d8159f3646a48` tested original ggml
  `40830de1f9b83695cce372a387763ef3600b9951` against the initial candidate
  `9066e6f2471eabed8632ded32d4514b6429d9790`, with frozen runner `bc9198776ba66074d6fca55fa2c0a79be7690876`; the
  pins above remain historical.
- Corrected original-kernel cases-only ggml: `ccadee2edb3db66f37e840e921f0d2b2c5940932`; corrected candidate ggml:
`f9d1f765dcc119403206775a137d5a03448c7e13`; corrected check parent `d930bbf5185c41c85b8175834666a2532c4a157b`.
The measured normal results used each side's retained product: the original ordinary-encoder helper was linked to
the retained original product — speech parent `114c514430c8aa1455c077819f8e72e674a48b75` with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266` — while the candidate normal helpers were linked to the retained
candidate product — speech parent `efa6a4ea491a576198a1f0dc2a7d8159f3646a48` with ggml
`9066e6f2471eabed8632ded32d4514b6429d9790`. Each ordinary helper is a distinct executable linked to the relevant
product libraries, not one shared binary; only the primary OFF/ON worker timings used the same retained candidate
executable. The corrected commits above are test-only — shared synthetic test header, linked ggml revision and
experiment note — and served only the corrected standalone checks. Only the original and candidate product builds
have matched 74/74 compile/link flags; the fixed anchor `f976ebd8e2fca1f15ddc2e6fc9f99e7090b0f9ea` used its own
respective executable with implicit architecture/SDK flags and is context only.
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
the disabled pipeline retains its old key. The tensor-API implementation remains unchanged. Source-level arithmetic
identity alone does not prove compiled byte equality; the recorded direct checks compare complete result bytes, not
compiled kernel bytes.

## Fusion and allocation boundary

Match three consecutive raw nodes, `MUL_MAT -> PERMUTE -> CONT`, and consume two nonempty nodes. Verify the exact
permutation, dimensions, contiguous layouts and view origin/offset. The final reshape is an unchanged view of CONT;
it need not be part of the matcher or a new operation. `ggml_can_fuse_subgraph_ext()` protects the omitted MM and
permutation against OUTPUT flags or additional consumers. CONT is the real output, so marking it or its final
reshape as OUTPUT, or reading CONT from additional downstream consumers, remains valid.

The matcher requires SIMD MM without tensor support in both STRUCTURAL and FULL modes. Device properties pass
through the existing fusion queries and graph optimizer; this does not import trial 006's fusion or kernel.
`GGML_METAL_HEAD_STORE=0|1` defaults to zero, is read once per process and rejects other values when the pattern is
examined. The runner owner received the fixed name and value range; the initial runs used frozen runner
`bc9198776ba66074d6fca55fa2c0a79be7690876`.

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

The first authorized device investigation has run, after the product and helper builds completed: the real encoder
graph was allocated with the ordinary allocator, only the final joint output declared, no intermediate retention,
and the normal graph's OUTPUT flags and order preserved. Across Q4_K, Q8_0 and F16 with 14 inputs each, candidate
ON covered 42 model/input pairs and 1,008 observed slot ranges: the 36 T>=64 pairs had 864/864 eligible ranges, all
in the same underlying buffer and overlapping neither input, while the six T<64 pairs had 0/144 eligible. Original,
OFF and ON matched across all 3,024 range records. Full data:
`outputs/haas-mac-campaign/016/allocator/result.json`. This was allocation only, and range eligibility is not a
fusion count; the later normal encoder execution, actual fusion/dispatch counts and worker timing are recorded in
the evaluation. A different allocation policy would require a separate proposal and measurement of its memory cost.

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
support the separate-build comparison. The correctness mode of `test-backend-ops -o ATTENTION_HEAD_STORE` uses the
existing MUL_MAT NMSE bound of 5e-4; a test-only follow-up moved that registration from the performance list to the
correctness list before any run. The CPU implementation requires that operand's reduction dimension to be
contiguous. Two distinct host poison guards are prepared.

The initial original and candidate CPU-reference comparisons on Metal ran and passed 46/46 each under that bound.
The subsequent original direct test produced 46 case rows, then aborted at graph construction of case 47: public
`ggml_mul_mat` rejects a transposed first operand for every backend, so the old transpose_a case could not be built
and cannot execute on any backend, Metal included. That abort did not test the kernel, no full direct gate
completed, and the candidate direct run did not start. The original failed record and partial dump were retained:
`outputs/haas-mac-campaign/016/checks/head-store/initial-failure-classification.json`.

The corrected original-kernel cases-only ggml `ccadee2edb3db66f37e840e921f0d2b2c5940932` and candidate ggml
`f9d1f765dcc119403206775a137d5a03448c7e13` change the same shared test header and no runtime. Case 47 now uses a
legal F32 head-stride-padding A negative at T=171, with rows and reduction contiguous and a gap of four floats
between heads, separate from the existing row-padding strided_a; it is eligible for the CPU reference. The
corrected inventory stays 52 direct cases (14 eligible, 38 controls), is now 47 CPU-reference, and only five mapped
controls are Metal-direct-only. Corrected fixture bounds and unchanged guards/runtimes were independently source
reviewed; the selector, allocator and threshold did not change. The corrected runs then completed: 47 CPU-reference
logical cases per original and candidate build, 52 direct cases plus two poison guards per original/OFF/ON process,
14 actual ON synthetic fusions with none in original or OFF, identical complete dumps in all three processes, and
2,755 related cases per build. Related regressions, normal encoder execution, official F32 stages and the worker
runs also completed; see the evaluation record for numbers.

Offset and strided output storage precedes its view in context creation order. Protected outputs and fusion counts
are checked separately from numerical equality. Zero allocation dependencies were introduced, and M5 exclusion
remains source-reviewed only. No bound was relaxed after a run.

`016-encoder-layout.cpp` is a separate diagnostic source, linked against each recorded product build using that
build's timing-target flags and libraries. It marks only the real final joint-projected output, allocates once with
the ordinary allocator, and reports all 24 input/destination ranges. Its `allocate` mode skips encoder execution,
though building the graph can populate the model's existing position cache on the GPU. Range eligibility is not a
fusion count. Its separate `compute` mode preserves normal command-buffer policy and writes complete final projected
F32 bytes; it refuses to execute a T>=64 graph whose range-eligible count is zero. No new runtime diagnostics or
intermediate retention are introduced into speech.cpp.

## Outcome

The plan below ran to completion and the prototype was rejected; the full record is
`experiments/haas-mac/016-attention-head-store-evaluation.md` with machine-readable companion
`experiments/haas-mac/016-attention-head-store-evaluation.json`.

1. The ordinary allocator diagnostic confirmed eligibility with no loosened gate: every T>=64 candidate range shared
   one buffer and overlapped neither input — 864/864 eligible ranges over 36 pairs — while T<64 inputs remained
   ineligible controls (`outputs/haas-mac-campaign/016/allocator/result.json`).
2. The corrected direct gate completed in all three processes: identical complete dumps, 14 actual ON synthetic
   fusions with none in original or OFF, zero introduced allocation dependencies, 2,755 related cases per build, and
   47/47 CPU references per build. Normal graphs used the fusion — 288 per model type, 1,152 total — with equal
   complete final projected bytes across original/OFF/ON, and two local encoder index sequences matched every input;
   runtime fusion counters were never initialized, so the command-buffer count is source-inferred only.
3. The official F32 stages passed their SNR bounds through encoded output — CPU bound 90 dB with lowest printed
   104.4 dB, GPU bound 25 dB with lowest printed 56.4 dB; full KDA/TDT was not run.
4. The warm alternating 9-input, 3-round screen measured Q4_K +2.786% paired median with 9/9 inputs faster and all
   three rounds improving, which justified only the Q4_K full run; Q8_0 (+0.678%, 5 faster/4 slower) and F16
   (+0.779%, 6 faster/3 slower) stayed screen-only.
5. The full Q4_K run measured +0.316% paired median on 17/30 inputs with ON p90/p99 above OFF and essentially
   unchanged RTFx, so the prototype is rejected for insufficient full whole-request benefit; short6 was not run. A
   useful prototype would still need a separate no-switch production implementation and a fresh original-runtime
   comparison. Official F32 reference beyond encoded output, Windows/Vulkan and other Apple generations remain
   separate, unverified requirements. No production integration or permanent ADR change was made.
