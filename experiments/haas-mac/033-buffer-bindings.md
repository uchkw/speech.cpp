# Trial 033 - per-encoder Metal buffer binding cache (source-only prototype)

Independent small-trial prototype. Rejected-or-adopted-later: nothing here establishes
equivalence, speed or adoption. No persistent ADR is intended for this trial.

## Source base and cases

- Runtime base commit: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Bundled synthetic cases: `f80257234bf0e7bcd38283147302090d69e82775`, with the test CMake
  fix `65f8904d195195b91f69d26e83c19826a888d5f0`.
- Changed here: `ggml/src/ggml-metal/ggml-metal-device.m` (encoder wrapper only), the
  compile-time trace option wiring in `ggml/CMakeLists.txt` and
  `ggml/src/ggml-metal/CMakeLists.txt`, and this note. Shaders, fusion and context
  partitioning, other APIs/headers, the tests and their CMake target, the allocator and
  graph topology are untouched.

## Runtime flag

- `GGML_METAL_BIND_CACHE`: unset and `0` = OFF (default), `1` = ON.
- Read strictly at most once per process, before the first encoder wrapper is created
  (`pthread_once` in `ggml_metal_encoder_init`). Any other string, including empty,
  aborts with `GGML_ABORT`. The resolved bool is snapshotted onto each fresh encoder;
  there is no `getenv` in the setter hot path.

## Cache scope, semantics and lifetimes

The cache lives inside the newly allocated `struct ggml_metal_encoder` and is zeroed by
the allocation, so it is empty for every fresh wrapper. It is never global and never
shared across encoders, command buffer partitions or threads. Slots follow the Apple
Metal ordinary compute buffer argument tables: indices 0..30 (31 entries). With the
cache on, a buffer or bytes index outside 0..30 aborts instead of indexing outside the
table; with the cache off the setters are forwarded exactly as before.

Per slot the last known non-null `MTLBuffer` object identity and byte offset are
remembered. The cache is non-owning and changes none of the existing resource ownership
or lifetimes, and nothing retains graph tensors: the production command buffers are
created with `commandBufferWithUnretainedReferences` (`ggml-metal-context.m`), so the
existing owners keep the bound buffers alive through GPU completion, not merely for the
host-side encoding, exactly as the direct setters already require. The cache itself dies
with its fresh encoder as before:

- same object and same offset: the redundant `setBuffer` call is omitted;
- same object at a different offset: `setBufferOffset:atIndex:` runs and the offset is updated;
- different or unknown object: the original `setBuffer:offset:atIndex:` runs and the slot is remembered;
- nil binding: the original `setBuffer` always runs, even when repeated, and the slot is invalidated;
- `setBytes`: the original `setBytes` always runs, even from the same stack address, and
  invalidates the same slot; bytes are never cached.

Comparison is exact resolved non-null `MTLBuffer` object identity - the object pointer
handed to `setBuffer` - never the underlying host pointer/contents and never the tensor
identity. Two distinct `MTLBuffer` wrappers mapped over one host page compare as
different objects and re-bind in full.

Pipeline changes do not invalidate buffer bindings and the pipeline setter is still
called every time. Threadgroup memory lengths, barriers, dispatch dimensions and counts,
command ordering, `endEncoding`/`free`, the existing abort boundaries, `n_cb` and fusion
are unchanged. The only new aborts are the strict flag parse and the on-cache index
check.

## 31-slot contract and inspected indices

In the bundled production callsites in `ggml-metal-ops.cpp` the largest `setBuffer`
argument is 9 (it appears at exactly one call). `setBytes` uses the same buffer argument
slots: the RWKV7 op binds buffers 0..7 and then passes B, T, C and H through `setBytes`
slots 8..11, so production buffer-argument slots including `setBytes` reach 11. The
direct wrapper cases exercise slots 0, 1, 2 and 28, 29, 30. The SDK contract and the
cache table remain the 31 slots 0..30, with no table expansion or fallback.

## Diagnostic compile-time trace (never in primary builds)

`GGML_METAL_BIND_CACHE_TRACE` is a CMake option, default OFF; a default build contains
no trace counters, logging, atomics or IDs. When compiled in, each encoder keeps local
counters and `ggml_metal_encoder_end_encoding` emits one parseable row through the
existing logger:

`cache req full offs skip nil bytes inval pipe disp bar`

- `req` counts requested buffer setter calls and satisfies `req == full + offs + skip`;
  `nil` is the reported subset of `full` that bound nil.
- With the runtime flag OFF the row counts the original full native setter calls
  (`full == req`, `offs == skip == 0`); with the flag ON it reports the actual optimized
  calls. That count difference is how two distinct mapped wrappers over one storage are
  shown to re-bind in full - output equivalence alone cannot prove it.
- Counts only: no GPU timing, no global counters, no `n_cb` change, and the row text
  makes no speed claim. A real API test/build of the trace is deferred.

## Declared cases and unrun status

- 48 whole-graph cases: width x frames x steps {3, 32} x offset views x alternating weights.
- 24 direct native wrapper cases: 10 transitions x serial/concurrent, plus fresh-encoder
  x 2 and parallel x 2.

All are UNRUN: no compute lease exists for 033 and this trial is source-only, so no
build or execution happened and no measured correctness or speed is claimed. The future
comparison is cases-only: run the identical cases under original, flag OFF and flag ON
builds and compare outputs exactly, using trace builds for the binding-count checks.

## Graph integration boundaries

Each graph compute creates one fresh encoder per command buffer partition under the
normal `n_cb` policy, and each encoder starts with an empty cache; nothing is shared
across partitions or threads. Allocator growth and remapping only produce new
`MTLBuffer` objects, which simply bind in full. All primary dispatch, barrier and
pipeline boundaries are unchanged.

## Limits

Metal-only code: Windows and Vulkan paths are unrun and out of scope here. No
equivalence, performance or adoption statement is established by this prototype.
