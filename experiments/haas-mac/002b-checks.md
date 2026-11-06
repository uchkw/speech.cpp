# Load-time FFN expansion checks

These tools check trial 002b against the same GGUF values with the experiment
disabled. They do not replace the unavailable official Haas F32 reference
checks. The prototype adds an F16 copy of quantized encoder FFN matrices; it
does not recover the checkpoint's values before quantization.

`002b-ffn-f16-check.cpp` builds the actual encoder graph and inspects its matrix
operands. Every original operand remains selected with the switch off. With the
switch on, exactly the quantized FFN matrices use the additional F16 buffers;
every other named operand retains its original pointer. Every uploaded F16 value
must match scalar conversion of the original GGUF value. The backend must
support the actual FFN matrix products. Haas v1.0 expects 96 expanded matrices
for Q4_K and Q8_0, and zero for native F16.

`002b-encoder-compare.cpp` compares frontend features, subsampling, all encoder
layers, the encoder output and its joint projection. It records the numerical
differences without applying an official-reference tolerance. Features and
subsampling must match bit for bit, as must every tensor in the native F16
control. It checks emitted token IDs, emission frames, emitted token durations,
text and graph counts. It does not observe every blank's duration decision.

Build the prototype with the shared campaign build command first. Set
`source_root`, `build_root`, and `out_root` to its source, verified build, and a
new output directory. Compile each tool against that build's libraries:

```sh
build_check() {
  /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
    /usr/bin/c++ -std=c++17 -O3 -DNDEBUG -arch arm64 \
    -I "$source_root/src/common" -I "$source_root/src/families" \
    -I "$source_root/ggml/include" -I "$source_root/checks" \
    "$source_root/experiments/haas-mac/$1.cpp" -o "$out_root/$2" \
    "$build_root/libfastconformer.a" "$build_root/libspeechcommon.a" \
    "$build_root/ggml/src/libggml.a" "$build_root/ggml/src/libggml-cpu.a" \
    "$build_root/ggml/src/ggml-blas/libggml-blas.a" -framework Accelerate \
    "$build_root/ggml/src/ggml-metal/libggml-metal.a" \
    "$build_root/ggml/src/libggml-base.a" -lm \
    -framework Foundation -framework Metal -framework MetalKit
}
build_check 002b-ffn-f16-check ffn-f16-check
build_check 002b-encoder-compare encoder-compare
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$out_root/ffn-f16-check" "$model" 96 > "$out_root/weights.jsonl"
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$out_root/encoder-compare" "$model" "$audio_dump_dir" \
  > "$out_root/paired-stages.jsonl"
```

Use zero instead of 96 for the native F16 model. Each audio directory contains
`audio.npy`, mono float32 samples at 16 kHz. The tool accepts several directories
in one invocation. Benchmark audio is not an official stage dump.

The measured implementation and first check are in
`f7b82dab076296188ded2423d20431eff8d620df`, built clean with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The exact compile commands, linked
library hashes and executable hashes are retained in the external campaign
artifacts `002b/checks/build.json` and `002b/paired-stages/build.json`. The second
check links the same retained libraries; it makes no inference source change.

The external `002b/stage30/` diagnostic uses the existing timing executables on
all 30 latency inputs in three alternating rounds. Each input is warmed directly
before its measured stage run. The request harness instead warms all inputs
before measuring requests, so subtracting the two separate timings does not
measure IPC directly. The diagnostics retain that distinction.
