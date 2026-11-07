# 032b: a true B=2 equal-length batched FastConformer encoder

This folder holds the source of the 032b experiment and nothing else: **source only, never built, never run, nothing
measured**. It was authored against speech.cpp parent `114c514430c8aa1455c077819f8e72e674a48b75` with the ggml
submodule at `e1a9ac12edbb57402618fd6936598603f5ee6266`. No compiler, check, model, audio or GPU has touched it; every
claim below about what the code does is a claim about what it says, and every claim about behavior waits for the
builds and runs that the complete worktree will make.

This experiment's design supersedes the proposal in `../031/032-equal-batch-design.md` that one shared batch-aware
encoder implementation serve both B=1 and B=2 through the same internal entry point. The builder here is **B=2 only**
and builds nothing for one utterance: the differential baseline is the existing `fastconformer::Encoder`, unchanged,
run on each utterance separately. The 031 note itself stays as it was written, an immutable checkpoint of its time;
this folder is what replaces its implementation proposal.

## Scope

- `batch-encoder.h/.cpp`: `fastconformer::BatchEncoder`, the experiment-only builder. It carries `ne[3] = 2` through
  every stage of the global relative-attention encoder — one im2col, one graph path per layer, the shapes the design
  names: normalized input `[mels,F,1,2]`, im2col `[9,W,H,2]`, subsample matrix input `[K,W*H,1,2]`, depthwise
  activation `[C,W,H,2]`, conformer residual `[D,T,1,2]`, Q/K/V `[dk,heads,T,2]` with head views `[dk,T,heads,2]`, a
  shared positional projection `[D,2*T-1]` of batch 1 that MUL_MAT broadcasts, AC and shifted BD `[T,T,heads,2]`, V
  CONT `[T,dk,heads,2]`, attention output `[dk,T,heads,2]`, head-merge CONT to `[D,T,1,2]`, kernel-9 activation
  `[D,T,1,2]`, and the joint projection `[joint_hidden,T,1,2]` through the existing `Joint::project_encoder`.
- The BD shift is the 4-D view of the raw `[2*T-1,T,heads,2]` product with row stride `(2*T-2)*sizeof(float)`, the
  raw tensor's own `nb[2]` and `nb[3]`, and offset `(T-1)*sizeof(float)`: no row of one utterance reads another's, and
  at T=1 the single row stays valid although the row stride is zero.
- Weights, tensor types, quantization blocks, arithmetic expressions, accumulation order (F32-accumulation `mul_mat`)
  and each utterance's M/N/K are the production encoder's own; B is never folded into N, and no kernel, packing, cache
  layout or precision change is in this experiment. The frontend is not batched: each utterance's features arrive
  independently normalized, as `NemoFrontend` makes them.
- The position cache mirrors `Encoder::Positions` privately: the same 32 MiB bound, the same position formula and the
  same projected weights, so a cached baseline is never compared with an uncached candidate — both encoders take the
  same branch for the same T, and both are warm before any timing. Inputs longer than the span keep the production
  behavior of computing their positions in the request graph. The cache lives in its own backend buffer and is never
  a per-request mutable input.
- Refusals, all before ggml can assert: nonpositive frame counts and unequal frontend frame counts (including
  different F that give the same subsampled T, since the convolution boundaries differ), feature arrays that do not
  hold their frame count of mel bins, and any attention other than global `rel_pos` (a file error on construction).
  No length padding, no masks, no cropping, no temporal concatenation; each utterance keeps the convolution's own
  zero padding.
- Computed results are split for their consumers by `split_batches`, which reads the whole `[.,.,1,2]` tensor and
  splits it at its **own** `nb[3]` after checking F32, batch 2 and contiguity, so one half cannot hold any of the
  other's values. The existing TDT decoders then run separately and sequentially on the two projected vectors, each
  with fresh request state; there is no batched decode, no concurrency, no queue and no new public API, and the
  production encoder, the C API, the worker and the server are untouched.

## Files

- `batch-encoder.h/.cpp` — the B=2 builder with optional `BatchStages` outputs for the checks, and `split_batches`.
- `fixture.h/.cpp` — the deterministic synthetic fixture writer and the check's own `Layout`: D=256, heads=2 (dk=128),
  FFN=512, channels=256, mels=80, two conformer layers, three stride-2 subsampling layers, kernel 9, and the minimal
  joint tensors. Every matrix axis is a whole 256-value block, so a Q4_K file keeps its matrices in Q4_K. Weights are
  finite, nonzero, distinct per tensor, made at run time only; nothing is committed.
- `check-support.h` — the poisoned compute (`ggml_gallocr_alloc_graph` + NaN-filled outputs + `Graph::compute_again`),
  bit-strict comparison and stage reporting shared by the two executables.
- `batch-check.cpp` — the synthetic differential check.
- `batch-model.cpp` — the real-model differential check and timing entry point.
- `CMakeLists.txt` — the standalone build, which adds the complete parent repository as a subdirectory and links
  `fastconformer` and `speechcommon` as they are.

## The synthetic differential check

Compares the two halves of one batched graph against the production encoder run on each utterance alone, at the
subsampling, every conformer layer, the encoder output and the joint projection, in the order data flows. **Pass is
bit equality of both halves' every stage, with every value finite**; largest error and SNR are printed as
measurements and relax nothing. This says original-versus-batch only — how close either stands to NeMo's numbers is
the family checks' question, unanswered here. Graph evidence is printed and enforced: the intermediate shapes are
printed in detail — im2col, depthwise, score softmax, subsampled, every axis of each, with `ne[3] = 2` — and the
node count, the split of each result at its own `nb[3]` and the bit equality above are each enforced, so a departure
fails the check. Scenarios: different utterances at every length; swap; duplicate; change-one-member-alone, whose
unchanged partner must keep its bits and whose changed one must move (so a constant-zero fixture cannot pass);
rejections for unequal F of one subsampled T (249 against 250 frames), short and long feature arrays, zero and
negative frame counts, and a local-attention model. Lengths: T = 1, 2, 3, 5, 9 (the kernel-9 boundary), 31/32/33,
63/64/65, 171 and 375, each from F = 8(T−1)+1. Each graph gets its own allocator with its results read before any
other graph touches that memory, and its outputs poisoned with NaNs before compute so an unwritten value cannot hold
an earlier graph's result. Every one of these lengths sits inside the cached position span — the fixture's own span
is 8191 positions on either side of zero and the longest length asks 374 — so the uncached branch, where an input
longer than the span computes its positions in the request graph, is untested here.

The build and the GPU runs take the campaign's lease, which keeps one configure, build or GPU run on the machine at
a time; the build directory is this experiment's own, so a main build beside it is not overwritten. The Metal runs
name their device outright — `MTL0` on the campaign's Macs — where the CPU runs name `cpu`, and the tensor type is
always the check's last argument.

    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake -B build-032b -S experiments/haas-mac/032b
    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake --build build-032b -j4 --target fastconformer-batch2-check
    ./build-032b/fastconformer-batch2-check <work dir> cpu f32        # the CPU F32 differential
    ./build-032b/fastconformer-batch2-check <work dir> cpu f16        # same-type CPU differential, F16 weights
    ./build-032b/fastconformer-batch2-check <work dir> cpu q4_k       # same-type CPU differential, Q4_K weights
    ./build-032b/fastconformer-batch2-check <work dir> cpu q8_0
    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock ./build-032b/fastconformer-batch2-check <work dir> MTL0 f32
    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock ./build-032b/fastconformer-batch2-check <work dir> MTL0 q4_k

CPU F32 and same-type CPU/Metal are distinct runs; a quantized file is never called an F32 differential because the
activations are F32 — the file's own matrix type names the run, and the check prints it. The work dir receives the
F32 fixture and its quantized copy, quantized by the existing `quantize_model_file`, and both are removed on a pass.

## The real-model check and timing

For a later lease on a machine with a model file and two recordings of one length; nothing of the kind is asked for
or provided here. The lease, the build directory and the named device are as the synthetic check's above.

    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake -B build-032b -S experiments/haas-mac/032b
    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake --build build-032b -j4 --target fastconformer-batch2-model
    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock ./build-032b/fastconformer-batch2-model <model.gguf> <a.wav> <b.wav> MTL0 4 10

The model loads on the requested device (`MTL0` on the campaign's Macs, `gpu`, `cpu` or a device name, `auto` by
default); decoding is explicit on the CPU through the recognizer's own CPU copy and thread pool, four threads by
default and any positive count for a check. Each recording must be at the model's own sample rate — the error names
each path, its rate and the model's — and nothing is resampled, whatever sample count or frontend frame count the
two share. The frontend normalizes each recording alone; unequal frontend frame counts stop with an error, and
nothing is padded or cropped. The differential runs first: each half's joint projection must be its baseline's own
bits (`Recognizer::encode` of that recording), and each half, decoded by the existing decoder with fresh state, must
give the baseline's ids, frames, durations, graph count and text. **A differential failure stops the timing; nothing
falls back to B=1 and no tolerance is relaxed.**

The timing then runs one untimed warm round of both methods — the two production encodes back-to-back, and the one
B=2 graph — and after it every round runs both methods, their order alternating with the round index, each timed
between its own wall-clock bounds whichever order it ran in, with both inputs' features ready at its start. Per
method it reports medians of the pair makespan; the first result's availability, the moment the caller owns the
projected vector, at the end of the completed first encode for the sequential method and of the completed pair for
the batched; and the second request's queue wait, the start of its own encoding minus the pair start, which the
sequential method's second utterance spends waiting for the first and the batched pair's second does not spend at
all, its encoding beginning with the pair's. The build / allocate / compute / readback phases are said apart, the
readback covering the transfer into the returned caller-owned vectors. Every figure covers the encoder and the joint
projection alone, without the frontend and the decoding, so none of them is an utterance's final latency. All four
outputs of a round are compared with the differential's baselines after both timed spans have ended. The graphs
differ in what they retain: the synthetic check keeps every stage and poisons it, this strict differential retains
the final joint projection and poisons it, and the timing graphs retain the final joint projection alone, without
poisoning. No benefit is claimed for an isolated request: both halves of the batched pair arrive together, so its
first result waits for the whole pair.

## Not supported here

Windows and Vulkan are unverified: nothing in this folder has run anywhere, and the `windows-check` skill's machine
has not seen it. No official reference comparison is claimed — NeMo's dumps exercise one utterance, so no batched
official reference exists — and no CER, no speech-bench figure, no Metal or Vulkan pass and no adoption is claimed
before the checks above actually run. The checks' compared conditions are exactly those they print; an unavailable
check is unverified, not passed.

## Source pins

- speech.cpp parent `114c514430c8aa1455c077819f8e72e674a48b75`
- ggml submodule `e1a9ac12edbb57402618fd6936598603f5ee6266` (branch `speech.cpp` of github.com/uchkw/ggml)

This experiment imports no 001–031 kernel or runtime prototype.
