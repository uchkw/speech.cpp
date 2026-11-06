# 029: reuse verified raw mel prefixes

This is an unbuilt, unmeasured source checkpoint from 114c514430c8aa1455c077819f8e72e674a48b75,
with ggml e1a9ac12edbb57402618fd6936598603f5ee6266. It targets repeated readings of growing live audio,
not the intrinsic latency of an independent, cold, one-off input. No C++ check or worker has been run.

SPEECH_HAAS_RAW_MEL_PREFIX accepts only 0 or 1, defaults to off, and is read once when a KDA
FastConformer engine loads. Non-KDA models do not read it, including invalid values. With it off, the engine
uses the original NemoFrontend::features() path. With it on, one engine owns one completed PCM/raw-mel
entry. The C API, worker protocol, request options, and shared NemoFrontend/Sortformer code do not change.
The existing model request mutex serializes access to the entry.

The cache compares the actual F32 PCM delivered after the library's whole-input resampler. A complete
memcmp is the fast path; on a mismatch, a forward bit comparison finds the first changed sample.
Signed zero is distinguished. A sample pointer, utterance ID, source waveform, duration, or text is not
evidence that samples match. A different utterance with the same actual prefix can safely reuse this pure
function's result. A shorter input starts fresh. An equal-length edit can reuse only its unchanged beginning.

The model instance fixes the sample rate, FFT/window/filterbank, mel count, guards, preemphasis, frame
origin zero, and arithmetic. The frontend has no dither. Its centered STFT uses constant zero padding,
not reflection. Only old frames whose exclusive window_end(frame) is within the identical PCM prefix
are copied. A frame that previously read right-hand zeros is recomputed when new audio can reach it.
The whole current PCM is preemphasized again in the original F32 order, so the sample before an append
boundary is handled normally. The full current raw-mel array is normalized in the original two-pass,
per-feature order every time. No normalized feature, encoder frame, or attention result is cached.

Haas uses a 512-point FFT, a 160-sample hop and 128 mels at 16 kHz. The conservative immutable frame
boundary is frame * 160 + 256 <= unchanged_samples; the number of output frames remains samples / 160.
Changing the resampler's former right-edge zeros can change old output PCM; comparing that actual PCM
excludes the affected raw-mel frames. Enhancement can change much more of the prefix and is treated the same
way. Live provisional reads currently use raw audio and the final read alone can use whole-utterance
enhancement, so the final read must not inherit a cache merely because its original recording matches.

Each prepared reading owns a private candidate entry. It is published only after the complete plain or
keyword recognition reaches its existing end and cancellation is not observed. Exceptions, rejected inputs,
and cancelled readings discard their candidate and retain the preceding completed entry, which is still
subject to full PCM comparison. An input longer than 30 seconds is processed whole on the original path;
it retains no candidate PCM/raw mel, and successful publication clears the older entry. If that long request
is cancelled or errors, the preceding completed entry remains. No valid audio is truncated or rejected by
the retention limit. Model destruction drops the cache.

At 30 seconds the retained payload is at most 480000 F32 samples plus 3000 by 128 F64 raw mels:
4,992,000 bytes, about 4.76 MiB. A prepared candidate can coexist with that entry; normal frontend scratch,
normalized features, the library's input audio, and encoder allocations are additional. The PCM comparison
and candidate PCM/raw-mel copies are part of the experiment's cost. A repeated identical file may hit the
cache too; report that separately from an independent one-off request.

## Checks prepared, not executed

fastconformer-frontend-prefix-check MODEL [REFERENCE_ROOT] compares raw F64 mel and final F32 features
bit for bit with fresh whole-input computation. It covers silence, boundary impulses, quiet alternating
samples and deterministic noise; 2/4/30 second inputs; exact window-end minus one/equal/plus one; shorter
inputs; first/middle/last sample changes at the same length; a changed append predecessor; signed zero;
discarded and failed candidates; the 30-second-plus-one limit; and whole-prefix 8/44.1/48 kHz resampling.
Optional official dumps keep the existing 90 dB frontend bound and compare their prefixes with fresh C++.
The check loads only frontend tensors but validates the model's full layout. Missing official references
are explicitly unverified.

Future validation requires original/off/on builds under the shared lock, CPU checks, API cancellation at
each progress boundary, late cancellation and errors, all plain/keyword paths, both recognition families,
Sortformer controls, Haas Q4_K/Q8_0/F16 CPU4 and Metal exact text/tokens/times, and Windows/Vulkan checks.
Use a separate growing-prefix diagnostic to record cache hits and frontend cost, then the unmodified worker
runner for ordinary single-input controls. Live end-to-end timings must include request copying, resampling,
normalization, full encoding, decoding, memory, and cancellation; fewer repeated FFTs alone are not a
measured user-visible gain. No frontend stage tool or campaign runner is silently changed by this branch.
