# 030: last-one KDA prediction prefix cache for plain TDT requests

Status: source-only checkpoint of an unmeasured prototype on the branch `codex/exp-haas-030`. The four findings of
the completed gpt-6.1-sol source review are fixed in source alone — the exact byte-for-byte comparison, the
mandatory controlled synthetic cases, the projected-input contract and the explicit threads — and nothing was
built, run or compared when this was written; every validation below marked unrun is unverified. No decision is
settled and no ADR is recorded.

## Scope

`SPEECH_HAAS_KDA_PREFIX_CACHE` makes the FastConformer engine keep the KDA prediction prefix of the last completed
plain request and reuse it where a later request's fed labels still match, skipping the prediction network and the
joint's prediction projection on a hit and computing the joint alone. The reuse is family-local
(`src/families/fastconformer/tdt-prefix.h/.cpp`); the decoder runs it (`tdt.cpp`), the recognizer carries the
overload (`recognizer.h/.cpp`, `decoder.h/.cpp`), and the engine owns the cache and its lifetime
(`src/engines/fastconformer-engine.cpp`).

Unchanged: the frontend, the encoder, global attention, whole normalization and the host encoder readback; the
public C API, the worker protocol, the model layouts, the resampler and every other family; the plain and boosted
vocabulary flow of a keywords request; warm-up, which never populates the cache. Nothing of 029 (raw-Mel reuse),
017 (buffer reuse), 003 (resident state), 024 (device argmax) or 028 (request-local plain-to-boosted sharing) is
part of this. A request that explicitly selects a weight type keeps it.

## The switch

`SPEECH_HAAS_KDA_PREFIX_CACHE` is read once, in the engine's constructor, after the model loads, and only for a
model with `fastconformer.decoder.kind` `tdt` and `fastconformer.decoder.prediction` `kda`: unset and `0` leave the
engine as it was, `1` turns the cache on, and any other value fails the load with an `InvalidArgument` naming the
variable. A model of any other kind never parses the value, so an invalid one goes unnoticed there. The name is
fixed and has been sent to the runner owner; it is not a request option and appears in no option table.

## Prefix induction and the off-by-one rule

A TDT decoding starts at frame 0 from the prediction network's zero state and feeds the blank first; the first
cached entry is that initial blank step, which every request shares. Each entry holds the fed label, the complete
KDA recurrent and convolution state after that label, the projected `joint.pred` vector as float32, and the next
label the joint actually selected on that request's audio.

Reuse is sequential induction. At step `k` a lookup succeeds only if the published entry `k` holds the same fed
label the request is about to feed; after the request's joint has chosen, the recorded choice is compared with the
published entry's next label, and the first mismatch ends reuse for the rest of the request even where later labels
would agree again. So a hit at step `k` certifies: identical model, decoder, backend and request configuration;
identical fed-label prefix `0..k` (by the lookups) and identical joint choices at steps `0..k-1` (by the
comparisons); and a fresh zero-state start. The prediction network reads its labels and the zero state alone — the
audio enters only through the joint — so the entry's prediction and state are the ones this request would compute,
which the check verifies exactly rather than assuming.

The off-by-one rule: the last emitted token can end the old audio before it is ever fed. It is recorded as the last
entry's next label, and no entry exists for the step that would feed it; a longer request reaching that point finds
no entry, computes the prediction and continues. The blank that ends an utterance is likewise a next label, never a
fed label. A request with more than 64 predictions keeps the first 64 entries and computes the rest normally;
candidates are capped at the same 64.

## Lifetime, reset and cancellation

One model serves one request at a time, and the cache holds only the last successful plain request's prefix. Each
request's candidate is request-local: it is published, replacing the old prefix, only after the whole recognition
and its result have succeeded and the run has not been cancelled. A candidate that is incomplete — cancelled by the
progress callback, before the first prediction or after some — or that belongs to another cache, or was never
bound, is refused by `publish()` with a `logic_error`.

The engine clears the cache conservatively: on a request with keywords (which bypasses it and takes the existing
plain+boosted flow untouched), on a changed decoding, language, `decode_on_cpu` or `timestamps` combination, on
shorter input audio, on cancellation and on any error from the request. A request rejected before the engine is
entered — a bad option value or a refused request as a whole — is outside the engine's visibility and leaves the
cache as it was; the next request to reach the engine still validates the whole prefix before any hit, so a stale
prefix cannot be misread, it can only fail to match.

## Memory and copies

With Haas's fixed dimensions each entry holds 46,720 state floats (186,880 bytes) and 640 projected prediction
floats (2,560 bytes): 189,440 bytes, 12,124,160 bytes (11.5625 MiB) for a 64-entry generation, and 23.125 MiB while
a candidate is under way beside the published one, since two generations coexist. These figures are this
configuration's, not a bound over arbitrary KDA models. They exclude the entry metadata, the current graph, the
decoding's prediction scratch and the graph allocator's storage. Every copy is a real cost: a hit copies the entry's
state and prediction into the decoding, every step copies them into the candidate, and publication moves the
candidate in place of the old prefix. What is not yet measured: the retained payload against the peak RSS of the
process, the allocator's share, and the time a hit saves against the copies it pays — all left for the runs below.

## Correctness matrix and what "on" must preserve

Unmeasured. For acceptance, with the switch off, unset and on, the same requests must return equal text, tokens and
times (`speech-api-check transcribe`, the worker recognition smoke, and the stage check below), and the campaign's
exact text comparison over the latency inputs and the timestamp quality run must not change. A hit computes the
joint through a joint-only graph where the plain path computes prediction and joint in one combined graph; the two
are mathematically the same computation, and the check holds every array of the two to byte-for-byte equality rather
than to any tolerance. No rounding between the two graphs has been measured, so no tolerance exists to borrow: the
owner of a real one records it where the check can name it. Source inspection is not proof of the runtime output:
any difference the check or the exact text comparison finds is a finding, not noise to absorb.

## The check

`fastconformer-tdt-prefix-check` (built by CMake, run by `checks/run.py` with the fastconformer family) takes a
model, the reference out dir, optional `.npy` inputs of projected rows, a device — `cpu`, `gpu` or a device's own
name — and `threads=N`, 4 by default, which the CPU backend and the recognizer's CPU copy compute with; the device
and the count are settled before any backend is created, and the run reports the backend and the count it used. On
each dump and place it compares the plain, cold and warm runs for equal tokens, frames, durations, graph counts and
progress sequences, and replays every recorded entry freshly through the prediction network. Every array it
compares — the full KDA state, the projected prediction, and all the token, blank and duration logits of the
joint-only graph of a hit against the combined graph of the same step — must be the same number of finite float32
values with every byte equal. The SNR, the largest difference and the argmaxes are printed only as diagnostics when
a comparison fails, and a difference the argmax agrees with fails all the same; no 40 dB floor and no other check's
tolerance is borrowed, for no rounding between the two graphs has been measured.

The inputs are the joint's projected encoder rows `[T, joint.hidden]` — 640 for Haas — which the TDT decoding takes
directly: a dump's `joint_encoded.npy`, the official `joint.enc` output, is read as it lies, and a dump that holds
none fails the check rather than having its `encoded.npy` projected; an optional `.npy` input of another width or
dtype is refused before anything reads it.

With `--synthetic <work dir>` the check needs no real weights: it writes deterministic synthetic decoder-only
weights, marked synthetic, into that work directory and runs the controlled cases on them through the plain and the
caching decoder paths alike, asserting expected ids, frames, durations, graphs, progress, hits and recorded
entries — the initial blank; blank chains with the blank-duration-0 correction; first-argmax ties; the max-symbols
frame advance on duration 0; the last emitted token ending audio before it is fed, then a longer input computing
its prediction; a first divergent next token followed by later matching ids that never resynchronize; the cap
boundary at 63, 64 and 65+ predictions; changed current frames altering the fresh joint decisions; cancellation
before the first prediction and after some; and empty input and the reset of a new request. The cases are a fixed
table; one that throws counts as failed, and the summary names how many ran, so a mandatory case cannot go missing
in silence. The synthetic weights are the check's own data, chosen so projected frames select labels and durations;
they are not a model, and nothing measured on them is evidence of NeMo's or flash-linear-attention's output. The
model run keeps the real-weight differential — the forged next label, the shorter-input cut, the cap tiling where
the audio reaches it, the cancelled decodings, shorter and zero input — and the tiny-state publication rules, which
run in both modes.

Its own limits, stated so no later reader mistakes them: the CPU copy of a GPU model's decoding tensors is not
reachable from outside the recognizer, so on that place the check verifies outputs, hits and graph counts, while
the exact entry replay and the joint-only-versus-combined logits comparison run where the weights are reachable —
the model's device, which for a CPU model is the whole check. Where a dump's own audio makes too few tokens for the
cap tiling to reach the cap, that is printed as not reached rather than passed, and the cap's boundary is the
synthetic cases' to hold, which cannot be short of it. Engine behavior — the flag's load-time parsing and
refusal, the keywords clear and bypass, publication after error and cancellation, and off/unset/on text, tokens and
times through the C API — is not proven by these family-level checks and stays for the runtime validations below.

## Commands for the later runs

None of these has been run. Every build and every inference waits for a compute lease from the campaign's root
first, and holds the shared perf lock `/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock` while it runs. The
model files, the dumps and the work directories are placeholders.

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock sh -c 'cmake -B build && cmake --build build -j4'
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock python3 checks/run.py fastconformer --devices cpu,gpu --types F32,Q4_K,Q8_0,F16
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock build/fastconformer-tdt-prefix-check <haas.gguf> <reference/fastconformer/out> [projected-rows.npy...] gpu threads=4
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock build/fastconformer-tdt-prefix-check --synthetic <check-work-dir> cpu threads=4
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock sh -c 'build/speech-api-check transcribe && python3 checks/smoke/worker_recognition_smoke.py'   # once, for the engine gaps
```

The runner's screen comes later, on the original instrumentation-free `compare.py`, with three alternating rounds,
only after a compute lease; this change adds no performance instrumentation and no timing of its own.

## Validations that did not run

- No build, no C++ check, no model inference and no GPU run: everything above is source inspection alone, on the
  base commit `114c514430c8aa1455c077819f8e72e674a48b75` plus this checkpoint. The submodule's pinned commit and the
  binaries' hashes are unrecorded because no binary exists; the run that measures records them. The synthetic
  weights exist as code alone: only the check itself, run with `--synthetic` and a work directory, writes them.
- The stage check has not been compiled or executed in either mode: the model-and-dumps differential, the
  `--synthetic` controlled cases, the exact byte-for-byte gates, the `joint_encoded.npy` reading and the threads
  parsing are all unverified by execution, and source inspection does not prove their runtime behavior.
- The graph and runtime comparisons of the original against the switch off and on — Q4, Q8_0 and F16 on four CPU
  threads and on Metal, the exact ASR tokens, texts and times — are unrun.
- The official Mac dumps cover the frontend, the encoder and `joint.enc` only. There is no official KDA or TDT
  reference for Haas, so the check's reference is the same weights' plain run, and no agreement with the official
  implementation is claimed anywhere; the synthetic cases are the check's own data and no official evidence either.
- Windows and Vulkan are unavailable in this run; a change to decoding code that runs on the GPU is unverified
  there until the windows-check pass.
- Engine and API behavior — flag parsing at load, keywords clear and bypass, publication after cancellation and
  error, off/unset/on outputs through the C API — is unrun.

## Measurement separation for the screen that follows

Live utterance extension is where reuse can pay: a request whose audio grows past a plain request just served hits
the whole shared prefix. Independent first requests do not inherently benefit from the history — the campaign's
benchmark repeats one identical input, where every request after the first reuses the whole prefix, and every
request begins with the same initial blank step; those identical-warm-repeat numbers must be reported apart from
any live-extension numbers, and neither is a gain over a cold first request. The screen must therefore show, per
side: the first (cold) request of each input apart from the later ones, and the alternating rounds as they come.
Gains are direct paired measurements against the fixed anchor, never compounded with other experiments'.
