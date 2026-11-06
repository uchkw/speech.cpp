# Encoder QKV packing trial

`SPEECH_HAAS_ENCODER_QKV=1` packs each Haas KDA model's encoder Q, K and V
projection matrices into one additional matrix when the encoder is constructed.
The default and `=0` keep the original graph. Values other than `0` and `1`
are refused. The switch does not select other FastConformer prediction families.
Within a layer, all three source tensors must have the same type, shape and
contiguous row layout. The enabled prototype refuses mixed types explicitly;
it neither converts them nor quietly runs a different path.

The source bytes are concatenated in Q, K, V row order. No GGUF tensor, stored
precision, attention equation, ggml kernel or decoder changes. The packed output
is `[3072, T]` for Haas v1.0. Q, K and V are views `[128, 8, T]`, with byte strides
`[4, 512, 12288]` and byte offsets `0`, `4096`, `8192`. The K view is permuted
directly rather than reshaped as a contiguous tensor. Q's positional biases and
the contiguous copy of V keep their original operations. A model with linear
biases adds each original bias to its corresponding head view.

The original weights remain allocated. The additional retained matrices occupy
144 MiB for F16, 76.5 MiB for Q8_0 and 40.5 MiB for Q4_K across 24 layers, plus
tensor metadata and buffer alignment. Packing temporarily holds one source
matrix's bytes on the host, at most 2 MiB for these F16 files. Loading and the
peak process footprint must be measured separately from warm request latency.
Two matrix dispatches per encoder layer are removed; the number of multiply-adds
and source weight bytes is unchanged. The combined projection can also increase
the live activation allocation, so the final peak must be measured.
The three campaign GGUF headers were inspected on November 7, 2026: each has
72 matrices of shape `[1024, 1024]`, all Q4_K, Q8_0 or F16 respectively, with
no mixed Q/K/V layer. This header inspection is not an execution check.

`005-encoder-qkv-check.cpp` checks the prototype against the same model with the
switch disabled. It is a paired implementation check, not an official reference
check. It checks:

- All bytes of all 72 source matrices against the uploaded 24 packed matrices,
  including the weight types, shapes and buffer usage.
- The actual encoder graph's selected operands: 72 separate products become 24,
  every unrelated named matrix still uses its original tensor, and the backend
  supports every node of the packed graph.
- Every production head view's offset and strides, its connection to Q's bias,
  K's score product and V's contiguous copy.
- Invalid environment values and mixed tensor types before any incompatible
  tensor data are read. The mixed case changes a borrowed descriptor temporarily
  and restores it; it is a guard check, not a new mixed-type GGUF fixture.
- Synthetic encoded lengths 1, 2, 31, 32, 33, 170, 171, 172 and 375, including
  tile boundaries, the position-cache boundary and a 30-second equivalent.
- Q/K/V outputs, attention results before their output projections, subsampling,
  every encoder layer, the encoder output and its joint projection.
- The graph with only its ordinary final output against the graph retaining
  intermediate outputs, on each side independently. Retaining intermediates
  can change allocation or fusion and is not the ordinary request path. The
  check retains each projection itself, then separates the packed Q/K/V rows
  on the host without adding graph operations. Comparisons follow data-flow
  order and report the first differing retained tensor.
- For supplied benchmark audio, emitted token IDs, frames, durations, text and
  graph counts from the ordinary final projections. It does not observe every
  blank-duration decision.

Every paired tensor must be finite and bit-identical. The check reports NMSE
and the largest absolute difference even when this strict requirement fails;
it does not borrow or loosen an official-reference bound. A rounding difference
requires diagnosis before any quality or adoption claim. Full worker comparison
with the fixed corpus and timestamps remains separate from this check.

Builds and inference must wait for the coordinator's campaign slot. Once the
prototype is committed and built clean, compile the check against that exact
build's static libraries, retaining its compile command and all source, library
and executable hashes. Set `source_root`, `build_root` and `out_root` to that
source, verified build and a new output directory. The command takes a backend
explicitly:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  /usr/bin/c++ -std=c++17 -O3 -DNDEBUG -arch arm64 \
  -I "$source_root/src/common" -I "$source_root/src/families" \
  -I "$source_root/ggml/include" -I "$source_root/checks" \
  "$source_root/experiments/haas-mac/005-encoder-qkv-check.cpp" \
  -o "$out_root/encoder-qkv-check" \
  "$build_root/libfastconformer.a" "$build_root/libspeechcommon.a" \
  "$build_root/ggml/src/libggml.a" "$build_root/ggml/src/libggml-cpu.a" \
  "$build_root/ggml/src/ggml-blas/libggml-blas.a" -framework Accelerate \
  "$build_root/ggml/src/ggml-metal/libggml-metal.a" \
  "$build_root/ggml/src/libggml-base.a" -lm \
  -framework Foundation -framework Metal -framework MetalKit
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$out_root/encoder-qkv-check" "$model" cpu "$audio_dump_dir" \
  > "$out_root/qkv-cpu.jsonl"
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$out_root/encoder-qkv-check" "$model" gpu "$audio_dump_dir" \
  > "$out_root/qkv-metal.jsonl"
```

Each audio directory contains `audio.npy`, mono float32 samples at 16 kHz.
Q4_K, Q8_0 and F16 each need their own check. Official Haas CPU F32/Metal dumps
and Windows Vulkan checks remain required for acceptance and are not supplied
by these paired comparisons.

## Validation on November 7, 2026

The runtime remains the clean build of `d3796ef`, with the pinned ggml unchanged.
The corrected check links those same retained libraries; its source, compile
command and executable hashes are recorded separately in the campaign artifacts.
Q4_K, Q8_0 and F16 each pass all 12 inputs on CPU and Metal with exact bits,
including the ordinary final output against the instrumented output on each
side. All nine real-audio/type pairs have identical emitted IDs, frames,
durations, text and graph counts on each backend. The CPU uses the actual
quantized or F16 weights with F32 activations and KDA states; these are not
official F32-weight reference checks.

The original strict Metal failure is preserved. Its candidate check alone
added 72 `CONT` nodes to read the head views. On Q4_K with one encoded frame,
this increased the candidate graph from 1695 to 1767 nodes. Metal assigns the
first 10 percent of the nodes to the first command buffer: its boundary moved
from 169 to 176. The trace shows `blk.2.ff1_norm`'s `NORM` at index 175 in the
first buffer and its `MUL` and `ADD` at 176 and 177 in the second. It dispatches
`kernel_norm_f32_4` and separate arithmetic instead of the three-operation
fusion. The first retained tensor that differs in data-flow order is layer
2's Q projection; layers 0 and 1 remain exact.

Collecting all original failures found the ordinary final projections exact
for all 36 input/type pairs, while the asymmetric candidate instrumentation
differed on 32 pairs. Disabling fusion made every original comparison exact.
Fusion debug mode also made the one-frame case exact, but that mode forces
single-command-buffer encoding and is not evidence of the ordinary path's
fusion pattern. The boundary trace therefore uses graph debug alone.

Retaining the packed product and separating its rows on the host restores the
1695-node instrumented graph and the three-operation fusion. It changes no
runtime tensor, projection dimensions, head stride or precision. All raw Q/K/V
projections and attention results are exact with this check, so neither the
larger projection nor its strided heads causes an observed numerical change
over the checked workload. No tolerance is relaxed and the failed original
logs are not relabeled as passes. The original checks, both fusion diagnostics,
the boundary traces and the corrected CPU/Metal results have separate paths
under external campaign artifacts `005/`.

## Performance on November 7, 2026

On the M2 Ultra with four CPU decoder threads, the full comparison uses the
same `d3796ef` binary with the switch explicitly disabled or enabled, plus a
directly measured fixed campaign anchor. The runner and Python 3.9.6 are fixed.
Each type has 30 latency inputs over three alternating rounds and 650 separate
timestamped quality requests. The paired reduction uses each input's median
over rounds; it is not the percentage change of the aggregate p50.

| Type | Paired latency reduction | Against fixed anchor | p50, off to on | Peak footprint, off to on | Load, off to on |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K | 2.48% | 1.99% | 81.48 to 77.81 ms | 456.99 to 493.17 MiB | 140.57 to 145.97 ms |
| Q8_0 | 1.62% | 1.25% | 77.48 to 76.67 ms | 750.02 to 825.11 MiB | 189.14 to 197.03 ms |

Q4_K improves 26 of the 30 input medians against the disabled switch and 28
against the anchor. Its per-round reductions are 1.19%, 6.01% and 1.85%; the
direct anchor comparison is steadier at 1.27%, 2.73% and 1.91%. Q8_0 improves
all 30 input medians against both references, with per-round reductions of
1.46%, 1.89% and 1.81% against the disabled switch. Every type's 650 quality
outputs and 90 latency texts match both references exactly. Literal and
repository CER are unchanged: 7.4668% and 4.6557% for Q4_K, and 7.5097% and
4.5761% for Q8_0. The separate three-input stage timer reduces encoder compute
by 2.70-4.57% for Q4_K and 2.74-5.29% for Q8_0; these percentages are not added
to worker request gains.

The independent six-prefix supplement reuses two- and four-second prefixes of
three existing recordings and the primary run's binaries, timer and thread
setting. All 18 text comparisons match for each type, and all six input medians
improve. Q4_K's paired reduction is 2.63% overall, 3.17% for two seconds and
1.87% for four seconds; Q8_0's is 4.19%, 3.51% and 4.21%, respectively. These
are truncated prefixes without reference labels, not naturally completed
utterances. Both CER fields remain null, and the supplement is separate from
the primary evaluation.

F16 remains a nine-input screen only: its paired reduction is 0.62%, with
per-round changes from a 4.98% regression to a 3.40% reduction, so it does not
receive a full evaluation. Its paired CPU/Metal correctness checks pass.

This is a promising, unadopted prototype with a memory and load-time cost.
Adoption still requires a switch-free implementation, another performance
comparison of that implementation, official CPU F32/Metal reference checks,
and Windows Vulkan validation. No model default changes here. The frozen full
and prefix evidence is under `005/full/` and `005/short/`; `005/evaluation.json`
records the independent per-round and duration-stratum aggregation.
