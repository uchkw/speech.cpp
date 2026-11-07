# 020b — CPU-only KDA constant preparation (default off)

This separate, independent, default-off trial derives from the published general prototype runtime `980269a` and
keeps `ggml` pinned at `e1a9ac12`. It is not yet built, run or measured, and the general trial's record is not
touched by this note.

## Change

The `KdaLayers` constructor is now the one owner of the constant-cache decision, stored once for its prediction
network's life: an enabled request with a null backend still throws the existing `logic_error`, and only
`ggml_backend_is_cpu()` — the null-safe declaration of the pinned `ggml-cpu.h`, still included explicitly — may
enable the cache. The `SPEECH_HAAS_KDA_CONSTANTS` flag/model gate at the Recognizer is unchanged and still
defaults to off.

The cache therefore selects actual CPU prediction only. On Metal or any other non-CPU prediction backend, every
prediction graph keeps the original per-layer EXP then NEG, and the constant cache's own allocation stays at
zero bytes because no setup graph is ever built for it there. On CPU, the existing preparation expression,
first-build timing, ABORT/retry, per-model ownership and lazy CPU threadpool lifecycle are untouched. By source
design a CPU-loaded run and a Metal-encoder/lazy-CPU decode follow those existing routes; this note does not
claim either has executed successfully. The standard warmup difference stands: quantized-default files
additionally warm the CPU decoder and can therefore prepare once during that warmup, while F16-default files
warm only the device, so the first requested CPU prediction can prepare later.

## Checks

`fastconformer-constants-check` keeps the full original trajectory comparison on every backend and now branches
by backend: CPU runs keep the positive contract (one preparation, the expected F32 `[1,heads]` values and
allocation, no per-layer EXP/NEG, the same derivation/retry and per-model ownership), while non-CPU runs are a
negative control (one EXP and one NEG per layer like the original, exactly zero preparations, a zero-byte
constant-cache allocation, a null snapshot backend and empty tensors after work). The real-model lifecycle
decodes in the order Device, CPU, Device, CPU and expects 0/1/0/1 preparations on the selected owners of a
device-loaded model and 1/1/1/1 on a CPU-loaded one, with plain/default-off owners always zero; the unset-switch
control now decodes both choices.

## Plan

- Strict synthetic check on CPU with 1 and 4 threads, then on Metal.
- Real Haas-ja-v1.0 models in Q4_K, Q8_0, F16 and F32.
- Public API, workers, keyword and cancellation gates.
- Distinct first and warm measurement runs, only after correctness passes.

No outcome is predicted here; nothing is added to CMake or other families.
