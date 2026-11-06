# CPU decoder repack check

`002a-cpu-repack-check.cpp` checks the experiment's allocation and row-one products
against ordinary CPU buffers containing the same GGUF values. It is a prototype
check, not a comparison against official reference tensors. It takes a model and
the expected count of repacked matrices. For Haas v1.0 on the native M2 Ultra
build, the expected count is 17 for Q4_K and Q8_0, and 0 for F16.

Build the prototype branch with the shared campaign build command first. Set
`source_root`, `build_root`, and `out_root` to the source, verified build, and a new
output directory. Compile the check against that build's libraries:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  /usr/bin/c++ -std=c++17 -O3 -DNDEBUG -arch arm64 \
  -I "$source_root/src/common" -I "$source_root/src/families" \
  -I "$source_root/ggml/include" \
  "$source_root/experiments/haas-mac/002a-cpu-repack-check.cpp" \
  -o "$out_root/cpu-repack-check" \
  "$build_root/libfastconformer.a" "$build_root/libspeechcommon.a" \
  "$build_root/ggml/src/libggml.a" "$build_root/ggml/src/libggml-cpu.a" \
  "$build_root/ggml/src/ggml-blas/libggml-blas.a" -framework Accelerate \
  "$build_root/ggml/src/ggml-metal/libggml-metal.a" \
  "$build_root/ggml/src/libggml-base.a" -lm \
  -framework Foundation -framework Metal -framework MetalKit
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  "$out_root/cpu-repack-check" "$model" 17 > "$out_root/result.jsonl"
```

The check compares every ordinary copy byte for byte, and every unselected
tensor after repacking. Selected matrices must have a CPU_REPACK buffer, its
repack trait, and support for the actual row-one multiplication. Each matrix
runs seven deterministic inputs, including zero and scales 0.001, 1, and 100.
The product bounds are normalized mean square error at most `1e-10` and largest
absolute difference at most `2e-5 * max(1, largest reference magnitude)`.
The JSON Lines output records each shape, buffer, dispatch checks, and errors.

For trial 002a, the retained check source has SHA-256
`c0ca8b2e3af02b95381b3f0b0bf054caa40cb732540f5338476b603f37c10cd0`.
It links the clean candidate libraries built at
`85c8f39e6bfee9f54bb8c668a7ca90f6b42e9245`. The check was compiled and run before
being added to the prototype branch; its source is retained byte for byte.
The precise compile command and binary hash are in the external campaign
artifact `002a/checks/build.json`.
