# 032: distinguish command submission from tensor batching

Status: read-only design at parent `114c514430c8aa1455c077819f8e72e674a48b75`
and ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`. No 032 runtime code,
build, GPU execution or performance measurement is included in 031.

## 032a: two independent subgraphs in one submission

An experimental family-level harness can call `Encoder::build` twice in one
`Graph`, with two independently normalized feature arrays and separate final
joint encoder projections. Mark both final outputs live, allocate once and
compute once. Shared weights and positional projections stay immutable. Warm
the lazy position cache before constructing the combined graph, rather than
starting another graph on the same backend during execution.

This retains each matrix's M/N/K, tensor shapes, reduction order and operations.
It tests host graph construction, allocation and command submission overhead.
It is not tensor batching and should never be labeled a length-batched encoder.
Both outputs become available after the combined graph completes, so the first
request can wait longer. Compare the two existing separate graphs, combined
subgraphs and each isolated request. Check the graph capacity and memory; do not
silently truncate either graph or retain diagnostic outputs in performance runs.

The minimal harness owns one backend and one allocator and computes them
sequentially. `Graph::compute_again` is valid only until that allocator is used
by another graph (`src/common/graph.h:83`). Concurrent calls on the same CPU
backend also share its work area (`ggml/src/ggml-cpu/ggml-cpu.cpp:170`), so merely
giving graphs different allocators would not establish thread safety.

## 032b: a true, equal-length B=2 encoder

This is feasible as a separate family-graph experiment using existing ggml ops.
It does not inherently require a new kernel, GGUF layout or packed weight file.
The initial scope is Haas global relative attention, exactly two independently
normalized inputs with the same **frontend frame count**, no length padding,
and sequential existing TDT decoders. Equal post-subsampling length alone is
insufficient: different frontend lengths retain different convolution boundaries.

Use batch dimension `ne[3]` throughout. The proposed shape flow is:

| Stage | Batch-preserving shape |
| --- | --- |
| Separately normalized frontend inputs | `[mels, F, 1, B]` |
| First convolution im2col | `[9, W, H, B]` |
| First/pointwise matrix input | `[K, W*H, 1, B]` |
| Subsampling depthwise activation | `[C, W, H, B]` |
| Subsampling output and conformer residual | `[1024, T, 1, B]` |
| Q/K/V before head permutation | `[128, 8, T, B]` |
| Q/K and relative-position matrix input | `[128, T, 8, B]`, positions with batch 1 |
| Content/shifted-position scores | `[T, T, 8, B]` |
| Transposed V | `[T, 128, 8, B]` |
| Attention output before head merge | `[128, T, 8, B]` |
| Kernel-9 depthwise activation | `[1024, T, 1, B]` |
| Final joint encoder projection | `[joint_hidden, T, 1, B]` |

Subsampling currently flattens to 2D and returns 3D after each convolution
(`src/families/fastconformer/encoder.cpp:99-120`). Replace those reshapes with
their batch-preserving 4D forms. Do not flatten B into the matrix column count:
keeping `[K, W*H, 1, B]` preserves each utterance's M/N/K rather than changing the
matrix tile regime as an additional experiment. The final channel/mel permutation
keeps B last, then reshapes to `[mel_width*C, T, 1, B]` before `sub.out`.

`ggml_im2col` already preserves `b->ne[3]` in its output
(`ggml/src/ggml.c:4570-4614`), and its Metal path passes `N`, per-batch source
offsets and the corresponding dispatch (`ggml/src/ggml-metal/ggml-metal-ops.cpp:4683-4756`).
Depthwise direct convolution preserves the same batch dimension
(`ggml/src/ggml.c:4931-4965`); the Metal dispatch includes N and its byte stride
(`ggml/src/ggml-metal/ggml-metal-ops.cpp:4881-4934`). The current depthwise helper's
permutations leave axis 3 intact, so its arithmetic and tap packing need not change.

Attention (`encoder.cpp:183-216`) needs four explicit updates:

1. Q/K/V reshapes carry B. Head permutation keeps it last. The model's `[128,8]`
   positional biases broadcast across T and B without copying or changing values.
2. Projected relative positions remain `[1024,2*T-1]`, shared read-only across
   the equal-length utterances. Their head view has batch 1; ggml MUL_MAT's
   existing `ne[2]`/`ne[3]` broadcasting handles B=2 (`ggml/src/ggml.c:3356-3386`,
   `ggml/src/ggml-metal/ggml-metal-ops.cpp:2631-2661`).
3. The unshifted BD result is `[2*T-1,T,8,B]`. Its relative shift becomes a 4D
   view `[T,T,8,B]`, with row stride `(2*T-2)*sizeof(float)`, the original head
   and batch strides `bd->nb[2]`/`bd->nb[3]`, and offset `(T-1)*sizeof(float)`.
   No row can read another utterance. T=1 must be checked explicitly because
   the row stride becomes zero even though there is only one row.
4. V permutation/CONT and final head-merge PERMUTE/CONT retain B, then the latter
   reshapes to `[1024,T,1,B]`. Neither 005's packing nor 016's direct stores is
   included in this independent variant.

Feed-forward, layer norm, residual additions, GLU and linear projections already
operate over the channel axis and broadcast weights across other dimensions.
The kernel-9 convolution's 3D/2D reshapes must retain B
(`encoder.cpp:304-312`). Batch norm stays folded into the same F32 weights/bias.
The output joint projection uses the same existing linear operation, after
which each batch slice is copied to its own projected vector and decoded in
sequence with an independent existing request state. Keep all model tensor
bytes, quantization blocks, F32 accumulation requests and activation types.

The smallest source boundary is an internal equal-batch entry point with a shared
batch-aware encoder implementation, while ordinary `build` calls it with B=1.
Reject B other than 1/2, unequal F, a mismatched feature element count and local
attention for B=2 before ggml can assert. Existing B=1 local attention remains
on its present path. Do not expose batching through the C API, server or worker
in this first trial. In particular, each frontend still performs its own
ordered double mean/variance over only its valid frames
(`src/common/nemo-frontend.cpp:104-120`).

Changing rank/dispatch can alter allocator reuse, fusion eligibility or kernel
selection even with the same per-utterance M/N/K. Preserving the intended
arithmetic therefore does not establish bit equality without execution.

## Checks and decision boundary

Before an implementation trial, add batch-shape cases for im2col, depthwise,
norm, position-broadcast MUL_MAT, relative-shift view/ADD/softmax and V/head merge.
Compare two different deterministic utterances, swap their order, duplicate one,
change only one, and verify the other never changes. Exercise T=1, short padded
convolutions, 31/32/33, 63/64/65, 171 and 375. Reject unequal frontend lengths
even when their subsampled T agrees. Cover F32 CPU, F16/Q4_K/Q8_0 CPU/Metal where
supported, tails and quantization blocks without relaxing existing tolerances.

Compare B=1 through the proposed implementation with the original graph first.
Then compare each B=2 slice, subsampling output, all 24 layer outputs, final
encoder and joint encoder projection against two independent B=1 runs. A check
may retain those intermediate outputs; timing uses the ordinary output-only
allocator. Exact final public TDT results are a separate check. Record numerical
differences at their first stage rather than inferring encoder equality from text.
Official dumps currently exercise one utterance; an official batch reference
and Windows/Vulkan checks remain additional requirements, not implied passes.

The primary workload still has one short request available. Batch formation
does nothing for it. Any performance screen must separate pair throughput from
the first completion, per-request queue wait and peak memory. With two ready
equal-length requests, no fill delay is necessary; allowing a wait or reordering
unlike lengths would be a different scheduling policy. Variable-length masks,
padding, packed decode and concurrent cancellation inside one batch are outside
this bounded variant and need another design. Keep both inputs/results alive
until the shared encoder completes; cancellation of one member cannot free its
storage or cancel the other member's work.

## Separate 020b idea: constant first prediction

The first blank/zero-state prediction and `joint.pred` have no audio dependency
(`src/families/fastconformer/decoder-step.cpp:10-18`, `transducer.cpp:93-112`).
They are constant for the same weights, backend, types and execution settings.
A cache of that first prediction and its resulting state may be simpler than
starting a CPU thread during encoding. Existing 020 only caches KDA's `-exp(a_log)`
constants; it does not cache this prediction. This is a separate proposed 020b,
not part of 031 or 032. Request-local state copying, initial blank/duration rules,
keywords, cancellation and backend identity would need their own checks.
