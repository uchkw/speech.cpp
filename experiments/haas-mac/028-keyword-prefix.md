# Trial 028: the keyword prediction prefix of Haas-ASR's custom vocabulary

Trial 028 lets one recognition's boosted decoding borrow the plain decoding's predictor work: `SPEECH_HAAS_KEYWORD_PREFIX=1`
makes a Haas-ja-v1.0 recognition with a nonempty custom vocabulary (KDA prediction network, TDT decoder, the only eligible
model) record the first prediction steps its plain decoding runs and reuse them in its boosted decoding while every fed
token matches, every joint decision, blank decision, duration, tree transition, max_symbols step and progress call computed
again. The runtime is settled and unchanged by this note's files: runtime source 24cfaed8e81cdbbf5dd17bdba8809552b992f669
on base 114c514430c8aa1455c077819f8e72e674a48b75, ggml e1a9ac12edbb57402618fd6936598603f5ee6266. Everything below is
**source only**: nothing has been compiled, run, or measured, and no numerical or performance evidence exists yet. An
independent Sol review of the runtime (state-after ownership, prefix divergence and continuation, error and cancel order,
default-off) stands; its six blocking findings against the differential check were corrected in the sources named below and
are themselves unverified by any execution.

## Scope, ownership, cap and memory

The prefix (`src/families/fastconformer/keyword-prefix.h`) is owned by one request: it is recorded by the plain decoding and
borrowed only by the boosted decoding of the same `WordsModel::recognize` call, on the same decoder and backend; a prefix of
another decoder or backend is refused before any graph computes, and nothing of it survives the request. At most 64 fed
labels are kept, each with a full F32 `state_after` and a full F32 `joint.pred` copy:

- logical float payload per entry: 46,720 state floats + 640 joint.pred floats = 47,360 floats = **189,440 bytes**;
- 64 entries: 12,124,160 bytes = **11.5625 MiB**.

Four different figures must be recorded separately, and none is measured yet: `KeywordPrefix::payload_bytes()` (the
allocated vector-capacity payload, at least the logical one), `entries().size()` (the retained entry count), the transient
copies (each recorded step copies its state and prediction into the entry; each borrowed step copies them back into the
live decoding), and the measured process peak of the whole request.

## The files of this trial

- `checks/fastconformer/fastconformer-keyword-prefix-check.cpp` (candidate-only): the focused differential check — plain,
  recorded, ordinary-boosted and prefix-reused decodings of the same input on a watched backend; strict byte equality of the
  consumed joint logits rows (no approved floating-point tolerance), argmax agreement, graph accounting from the observed
  kinds, replay of every recorded entry (the terminal one included) through an independent `DecoderStep` from zero, a
  crafted same-final-label-after-two-pasts sensitivity case, cancellations that must reach their callback and leave a
  partial run, owner refusals, and the 63/64/65 capacity boundary. Its `--fixture` mode writes the F32 fixture of
  `003-state-fixture.h` (kept byte-identical to `reference-fixtures/003-state-fixture.h`) into a fresh directory the caller
  gives; its model mode runs on the real Haas-ja-v1.0 file's own weights with synthetic frames. Every backend it makes
  computes on four CPU threads.
- `checks/fastconformer/fastconformer-keyword-prefix-normal-check.cpp` (prototype, pre-028 APIs only): the normal
  production path. It drives `WordsModel::recognize` on a real model and a non-silent speech WAVE with an ordinary
  positive-alpha `Vocabulary` and a chosen guard case, sets or records `SPEECH_HAAS_KEYWORD_PREFIX` before any model loads,
  runs the same recognition unobserved and then observed in sequential scopes (never two full models at once), projects the
  audio through the real frontend and encoder, clears the observer after the encoder so only the decoding is counted, and
  writes one canonical artifact to an output path that does not exist, refusing one that does — an input file or a former
  run's artifact among them — before anything runs. It refuses `--decode cpu` on a non-CPU device: the lazy CPU decoding
  backend is made inside the library and cannot be watched without changing production code.
- `checks/decoding-observer.h` (shared, check-side): wraps a backend's `graph_compute` alone and reads, after the backend's
  own compute and a synchronize, only tensors production marked `GGML_TENSOR_FLAG_OUTPUT` — one contiguous F32 row of the
  joint's width, counted, so a graph of a decoding kind without exactly one row fails. No output flag, tensor, precision,
  allocator or fusion behavior is changed, and there is no timer or counter in the runtime.
- `experiments/haas-mac/028_compare_keyword_prefix.py` (stdlib-only): the offline comparator of the original/off/on
  artifacts, below.
- `CMakeLists.txt`: the normal tool's target block (checks are built but never installed; released binaries are untouched).

## Original, off and on: comparison boundaries

Three processes run the **same** model file, WAVE, keywords, alpha, guard, device and decode arguments, differing only in
their build and selector:

- **original** — a build of the base 114c5144 with the normal tool copied in (recipe below); the runtime has no prefix
  code, so `--prefix` changes nothing;
- **off** — the candidate runtime (24cfaed8) with `SPEECH_HAAS_KEYWORD_PREFIX` unset or `0`;
- **on** — the candidate runtime with `SPEECH_HAAS_KEYWORD_PREFIX=1`.

The comparator enforces: original and off agree on everything (texts, progress bits, logits bytes, the whole graph-kind
sequence); on agrees with them on texts, progress bits and logits bytes; the plain pass's graphs (up to the progress
boundary, the first progress call at a half or more) are identical in all three; after the boundary the on run has fewer
prediction graphs and as many more joint-only graphs, at least one of each, with the totals holding — each cache hit still
computes one joint graph, so total graph counts alone prove nothing and are never used as evidence. Text, progress and
strict F32 byte equality are required because replacing the combined prediction-and-joint `DecoderStep` graph with the
standalone joint graph can change a backend's arithmetic or allocation; the raw token-plus-blank and duration argmax values
(which sit before the tree's boost and are independent of it) are checked separately and must not move.

## Exact commands (for the later measured runs)

With `<model>` the converted Haas-ja-v1.0 GGUF (KDA + TDT), `<wav>` a non-silent speech utterance at the model's rate,
`keywords.txt` one `written form=reading` a line, and the two build trees `original/` (built by the recipe below) and
`candidate/`, a worktree of the trial's own sources at 24cfaed8, built with:

```
cmake -S candidate -B candidate/build
cmake --build candidate/build -j4 --target fastconformer-keyword-prefix-check fastconformer-keyword-prefix-normal-check
```

```
original/build/fastconformer-keyword-prefix-normal-check <model> <wav> /tmp/028/original.kpw cpu \
    --keywords-file keywords.txt --alpha 1.0 --guard 0 --decode device --prefix 0
candidate/build/fastconformer-keyword-prefix-normal-check <model> <wav> /tmp/028/off.kpw cpu \
    --keywords-file keywords.txt --alpha 1.0 --guard 0 --decode device --prefix 0
candidate/build/fastconformer-keyword-prefix-normal-check <model> <wav> /tmp/028/on.kpw cpu \
    --keywords-file keywords.txt --alpha 1.0 --guard 0 --decode device --prefix 1
python3 experiments/haas-mac/028_compare_keyword_prefix.py /tmp/028/original.kpw /tmp/028/off.kpw /tmp/028/on.kpw
```

Each of the three runs writes its own artifact path, which must not exist when the run starts: the tool refuses an
existing output path before anything runs, so a re-measurement takes a fresh path rather than an overwritten artifact.
The artifact's metadata names the model, its weights and the wav's name and sample count, but it does not identify those
files' bytes: record a manifest of the exact model, WAVE and keywords file once, and recheck it before each
original/off/on run, so a replaced input cannot pass for the one that was measured:

```
shasum -a 256 <model> <wav> keywords.txt > /tmp/028/inputs.sha256
shasum -a 256 -c /tmp/028/inputs.sha256
```

Direct CPU-loaded decoding uses `cpu` with `--decode device` or `--decode cpu` (one backend on the CPU); a selected Metal
device uses `gpu --decode device`. The differential check runs as:

```
candidate/build/fastconformer-keyword-prefix-check --fixture /tmp/028/fixture-a cpu
candidate/build/fastconformer-keyword-prefix-check --fixture /tmp/028/fixture-b gpu
candidate/build/fastconformer-keyword-prefix-check <model> cpu
```

Each `--fixture` directory must not exist (the tool refuses one that does); the fixture file stays in it afterwards.

## Unobserved vs observed control, and the direct check

Each normal run first performs the recognition **unobserved** (no observer in the path) and then **observed** (the same
backend wrapped), in sequential scopes, and fails unless the words, the speech bit and the exact progress doubles agree;
the artifact stores both. The differential check additionally holds the direct full-state and decision coverage the normal
path cannot see: every recorded entry's `state_after` and `joint.pred` bytes replayed from zero through an ordinary
`DecoderStep`, and the reused pass's every decision compared with the ordinary boosted one's.

## Building the original-base comparison tool

In a clean worktree of the base, at `original/`:

```
git -C original checkout 114c514430c8aa1455c077819f8e72e674a48b75
git -C original submodule update --init ggml
cp <bundle>/checks/decoding-observer.h original/checks/
cp <bundle>/checks/fastconformer/fastconformer-keyword-prefix-normal-check.cpp original/checks/fastconformer/
mkdir -p original/experiments/haas-mac
cp <bundle>/experiments/haas-mac/028_compare_keyword_prefix.py original/experiments/haas-mac/
```

and append to `original/CMakeLists.txt` the block:

```cmake
add_executable(fastconformer-keyword-prefix-normal-check checks/fastconformer/fastconformer-keyword-prefix-normal-check.cpp)
target_include_directories(fastconformer-keyword-prefix-normal-check PRIVATE checks ggml/src)
target_link_libraries(fastconformer-keyword-prefix-normal-check PRIVATE fastconformer)
```

then build the old worktree's target, its source and build directories spelled out:

```
cmake -S original -B original/build
cmake --build original/build -j4 --target fastconformer-keyword-prefix-normal-check
```

Before trusting the build, verify the old runtime is exactly the base and the copies are exact — every check below fails
by its exit status, and none of them has been run yet:

```
git -C original diff --exit-code 114c514430c8aa1455c077819f8e72e674a48b75 -- src include   # the base's tracked library code is unchanged
git -C original diff --exit-code 114c514430c8aa1455c077819f8e72e674a48b75 -- ggml          # the gitlink still pins the base's submodule
test "$(git -C original/ggml rev-parse HEAD)" = e1a9ac12edbb57402618fd6936598603f5ee6266   # the submodule is checked out at that pin
git -C original/ggml diff --exit-code -- src include                                       # the submodule's tracked code is unchanged too
cmp <bundle>/checks/decoding-observer.h original/checks/decoding-observer.h
cmp <bundle>/checks/fastconformer/fastconformer-keyword-prefix-normal-check.cpp original/checks/fastconformer/fastconformer-keyword-prefix-normal-check.cpp
grep -n 'GGML_TENSOR_FLAG_OUTPUT' original/ggml/include/ggml.h   # the flag the observer reads must exist at that pin
```

The tool targets the pre-028 APIs by inspection: it touches only `Recognizer`, `WordsModel::recognize`,
`Vocabulary`, `word_entries`, `read_wav`, `read_identity` and the backend helpers, whose signatures the bundle's
`reference-base/` copies confirm unchanged; the differential check, which needs the 028 APIs, is deliberately not copied.
It has not been compiled or run anywhere: whether the original-base build in fact succeeds is settled only when the
commands above are carried out, and this note claims no such result.

## Coverage status

| Coverage | Status |
| --- | --- |
| CPU F32 (synthetic fixture) | designed: fixture mode of the differential check; not built, not run |
| Real quantized CPU decoder (same-weight Q4) | designed: model mode on the real file, `cpu`; not built, not run |
| Metal device decoding | designed: normal tool `gpu --decode device`; not built, not run |
| Metal encoder + lazy CPU-decoder worker | not covered: unobservable without changing production; left to a separate entry-point differential run that does not exist yet |
| Official NeMo/FLA reference dump | not covered: the synthetic F32 fixture and the Q4 same-weight differential check supply no official reference evidence |
| Physical Windows Vulkan | unverified: no build or run on any Windows machine |

## Measurement protocol (later, none of it done)

Cold first requests are separated from warm ones; short utterances (a few seconds) and long ones (around 30 s) are separate
cases, and keyword cases where the boosted prefix matches the plain recording are separated from cases whose boost diverges
early. The recording and copy costs (the plain pass's entry copies and the borrowed steps' copies back) are measured apart
from the decoding they save, and the per-request process peak is recorded. Fixed four CPU threads, the same model and audio,
isolated serialized alternating runs of at least three rounds, off and on interleaved, cumulative gains read against one
fixed anchor rather than multiplied from per-experiment numbers.

Keywords disallow timestamps, so the campaign's normal 650-run no-keyword timestamp measurement is the regression control:
a keyword-only improvement is not added to any primary-code speedup figure, nor to trials 027 and 030, whose cache and
cross-request reuse stay out of this branch entirely. No ADR or README records a decision: the experiment is reversible and
off by default, and nothing is settled until the evidence above exists and the maintainer decides.
