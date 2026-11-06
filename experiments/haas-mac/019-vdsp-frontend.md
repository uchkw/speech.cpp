# Use a double-precision vDSP transform in the Haas frontend

This is a source-only prototype on `codex/exp-haas-019`, from speech.cpp
`114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. No build, FFT execution, inference, performance measurement or adoption
decision is completed by this checkpoint. Trial 018's frame parallelism is not included.

## Selection and dependencies

`SPEECH_HAAS_FRONTEND_VDSP` accepts exactly `0` or `1`; unset and `0` select the ordinary frontend. The FastConformer
factory reads it only when `general.name` is `Haas-ja-v1.0` and `fastconformer.decoder.prediction` is `kda`. Other models
do not read it, including an invalid value. The recognizer and its official frontend stage check use the same factory.
Sortformer and Nemotron keep their three-argument `NemoFrontend` construction; the shared frontend does not read the
environment. Explicit selection on a non-Apple platform is an error, not a fallback.

The implementation is C++17. CMake links `speechcommon` explicitly to the system Accelerate framework on Apple
platforms; it does not rely on ggml's transitive framework dependencies. No package or downloaded library is added.
Non-Apple builds include the refusal path and need no Accelerate headers or library.

The existing `RealFft` implementation and its positive-even-size contract are unchanged. The vDSP helper accepts the
documented subset `N = f * 2^n`, where `f` is 1, 3, 5 or 15 and `n >= 4`. Haas's 512 is in this subset; 400 and 42 are
examples supported by `RealFft` but refused by an explicitly selected vDSP helper. A setup allocation failure is an
error. Neither an unsupported transform nor an allocation failure silently selects another implementation.

## Transform layout and arithmetic

The helper uses `vDSP_DFT_zrop_CreateSetupD` and `vDSP_DFT_ExecuteD`, both double-precision APIs. The input is packed as
`real[j] = input[2*j]` and `imag[j] = input[2*j+1]`, each with `N/2` values. Execution uses the supported in-place form,
with exactly identical input/output real pointers and input/output imaginary pointers, and disjoint real/imaginary
halves. The helper then writes every element of the ordinary `N/2+1` complex spectrum:

```text
spectrum[0]   = (real[0] / 2, 0)
spectrum[N/2] = (imag[0] / 2, 0)
spectrum[k]   = (real[k] / 2, imag[k] / 2), 1 <= k < N/2
```

The division by two corrects vDSP's real-forward scale, including DC and Nyquist. Multiplication by the binary-exact
constant `0.5` occurs before the existing `std::norm` power calculation. There is no extra one-sided-spectrum factor.
Pre-emphasis remains float32. Window multiplication, power, mel accumulation, the additive log guard, log and
per-feature normalization remain double precision, in the existing loop order. The FFT's internal rounding order
changes, so identical features or recognition output cannot be asserted from source inspection.

Apple's primary documentation establishes the packing, scaling and supported lengths:

- [Data packing and scaling](https://developer.apple.com/documentation/accelerate/understanding-data-packing-for-fourier-transforms)
  describes DC in the real slot, Nyquist in the imaginary slot, and the real-forward factor of two.
- [Double real DFT setup](https://developer.apple.com/documentation/accelerate/vdsp_dft_zrop_createsetupd)
  describes the supported lengths, forward direction, allocation failure and shared-setup alignment requirement.
- [Double DFT execution](https://developer.apple.com/documentation/accelerate/vdsp_dft_executed)
  identifies the double-pointer API. Some prose labels on that page say single precision; the declaration and the
  Apple SDK header specify `double`.

## Ownership and setup lifetime

Each selected frontend constructs one vDSP setup with `Previous=nullptr`, owns it through RAII, and destroys it when
the frontend is destroyed after its calls finish. There is no global setup cache or sharing with another frontend.
The original `RealFft` member remains initialized on both paths; this prototype adds the vDSP setup without changing
ordinary setup construction. Its additional retained setup memory and load time must be measured.

Each `log_mel` call owns its scratch array of `N` doubles, reused across that call's frames. At `N=512` this adds
4,096 bytes of working storage, beside the existing windowed input, spectrum and power vectors. No scratch is mutable
frontend state, and no allocation or setup creation is added inside the frame loop. The prototype starts no threads
and changes no process-wide thread settings. It uses a private setup, so the 64-byte alignment rule for setups that
share storage through `Previous` does not apply; ordinary vector storage is valid, though alignment can affect speed.

[Apple's DFT guide](https://developer.apple.com/library/archive/documentation/Performance/Conceptual/vDSP_Programming_Guide/USingDFTFunctions/USingDFTFunctions.html)
places setup creation outside real-time processing and permits repeated execution.
[Double setup destruction](https://developer.apple.com/documentation/accelerate/vdsp_dft_destroysetupd)
must not overlap an operation using that setup or one sharing its storage. The installed Apple SDK's `vDSP.h`, under
“How to use the Discrete Fourier Transform (DFT) and Discrete Cosine Transform (DCT) interfaces”, additionally states
that execution reads setup memory only and can run concurrently, while creation/destruction must not overlap users
of shared storage. This was checked in the Command Line Tools SDK header on 2026-11-07; concurrent execution is not
exercised by this trial. Frame-parallel interaction remains a separate future variant.

Metadata-only `speech_model_info_open` goes through `read_file_info` and the metadata-only `ModelFile` constructor;
it constructs neither a frontend nor an FFT setup. A normal recognizer constructs the frontend after its model has
loaded. The CPU-only decoder copy has no frontend. None of these ownership boundaries is moved.

## Prepared checks

These checks have been written but not compiled or run:

- `fft-check` remains unchanged, including its arbitrary-even lengths 2, 6, 10, 14, 16, 42, 50, 400 and 512.
- `fft-vdsp-check` compares 964 transform cases at sizes 16, 32, 48, 80, 240 and 512 against both the ordinary FFT and
  a compensated direct DFT or an analytic impulse. It covers every impulse position, constant/DC, alternating/Nyquist,
  bin-one and mixed tones, very quiet data and zeros. Output and scratch are poisoned before reuse, input and boundary
  guards are checked, and the existing `6 * epsilon * floor(log2(N))` relative-norm bound is retained. Unsupported
  lengths are explicitly refused. On non-Apple platforms it checks refusal and reports numerical cases as not run.
- `fastconformer-frontend-vdsp-check` compares whole log-mel and normalized outputs, finite values and shapes, and
  keeps a 90 dB floor for differential comparisons. It covers silence, DC, Nyquist, mixed tones, quiet audio, centered
  zero padding, the two-frame minimum, repeated calls, long-to-short calls and split-frame calls using a nonzero input
  origin. Repeated and split calls must be bit-identical within one implementation. Unset and zero must be exact, and
  the shared constructor must ignore the flag. Running it with a non-Haas FastConformer model requires exact off/on
  equality and verifies that an invalid selector is ignored there.
- An optional reference directory runs both selected and ordinary outputs against each official frontend dump with
  the unchanged `SNR > 90 dB` criterion. The existing `fastconformer-frontend-check` also keeps that exact criterion
  and now selects the same path as the recognizer. A differential pass is not an official-reference pass.

Examples for the coordinator's later serialized compute slot, not commands executed at this checkpoint:

```sh
./build/fft-check
./build/fft-vdsp-check
./build/fastconformer-frontend-vdsp-check "$MODEL" "$REFERENCE_OUT"
SPEECH_HAAS_FRONTEND_VDSP=0 ./build/fastconformer-frontend-check "$MODEL" "$REFERENCE_OUT"
SPEECH_HAAS_FRONTEND_VDSP=1 ./build/fastconformer-frontend-check "$MODEL" "$REFERENCE_OUT"
```

## Pending evaluation

Build and execution must hold `/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock`, with builds bounded to `-j4`.
The runner owner has the final selector name for a new allowlist checkpoint. Preserve every existing run's frozen
runner and provenance.

First verify candidate-off against the fixed base, then off/on with the same weights, audio and options. Run the
official Haas frontend dumps without loosening 90 dB; the same frontend weights occur in Q4_K, Q8_0 and F16, but all
end-to-end variants still need their own output checks. Check text, token IDs, frames and timestamps, and use the
campaign's CER/WER evaluation if any recognition decision changes. Non-Haas FastConformer and Sortformer stage
controls, Linux/Windows builds and explicit non-Apple refusal remain unverified.

Measure pure load, ready warmup, first request, frontend stage time, whole-request RTFx and peak memory separately.
Alternate off/on under the same lease and require a gain beyond round variation. Arithmetic in another FFT and fewer
host instructions are hypotheses, not measured improvements. This checkpoint contains no result HTML or acceptance
claim.
