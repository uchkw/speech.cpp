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
  can change allocation or fusion and is not the ordinary request path.
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
by these paired comparisons. No build, execution or measurement result is
claimed by the preparation of these sources.
