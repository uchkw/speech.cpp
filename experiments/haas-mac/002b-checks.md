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
The check marks intermediate tensors as graph outputs, which can change fusion
and allocation relative to an ordinary request. The full worker comparison is
separate evidence from the ordinary request path.

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

## Evaluation on November 7, 2026

The prototype remains unadopted. On the M2 Ultra with four CPU decoder threads,
the Q4_K median paired request reduction over 30 inputs and three rounds is
1.76%, versus 1.85% for the unchanged native F16 control. Q4_K adds 768 MiB of
retained FFN weights and raises median load time from 166.15 to 500.96 ms.
All 90 latency texts and 650 timestamped public outputs match for each type.
Q8_0 remains an exploratory nine-input result, with no full extension.

The 30-input stage diagnostic reduces Q4_K encoder compute by 9.99% and core
time by 7.56%; every input improves. A separate instrumented worker build on
three inputs confirms reductions of 9.30% in encoder compute, 7.37% in the
engine span including destruction, and 3.94% in the request. Native F16 encoder
compute is unchanged. The three-input request figure is supplementary and does
not establish a stable gain for the primary 30-input evaluation.

The same-request diagnostic's outside residual is about 42-50% of request time.
It includes input preparation, transport, parsing, queuing and terminal receipt
outside the contained worker interval. It does not identify IPC, standard I/O,
JSON parsing or any individual operation as the cause. The measured audio
copies and object cleanup are too small to account for the encoder savings.
Median subintervals are not additive; paired arithmetic mean differences obey
the request/contained/outside decomposition. No further inference change is
included in the diagnostic branch.

External evidence is retained in `002b/` and `002b-worker-probe/` under the
campaign artifact directory. The latter has an independent source/build,
frozen runner, three-input warmup, alternating order and instrumentation
overhead comparison. Its Python environment differs from the primary run and
is recorded in its manifest. Official CPU F32/Metal stage references and
Windows Vulkan validation remain unavailable. These limitations and the memory
trade prevent an adoption claim.
