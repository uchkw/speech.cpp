# 022: native settings, LTO and PGO

This is a source-only experiment checkpoint, based on speech.cpp
`114c514430c8aa1455c077819f8e72e674a48b75` and unmodified ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It changes no runtime, kernel, public option or release default.
No configure, C/C++ compilation, large-file hashing, download, model execution or performance measurement has run.
The coordinator schedules each later build or run. Only the coordinator pushes.

## Separate variants

| Variant | Host compiler change | Comparison |
| --- | --- | --- |
| baseline | Release `-O3`, native ggml CPU, no LTO or PGO | Fresh control from the same source and pinned toolchain |
| 022a | `GGML_LTO=ON` | baseline versus ggml-subtree ThinLTO |
| 022b | Top-level `CMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` | baseline versus all host targets using ThinLTO |
| 022c-train | Instrumentation, with an explicit `none`, `ggml` or `host` LTO mode | Training only; its latency is not performance evidence |
| 022c | A frozen profile, with the training build's same LTO mode | Non-PGO control with that same LTO mode versus profile use |

The existing campaign anchor already has `GGML_NATIVE=ON`, Release `-O3 -DNDEBUG` and
`-mcpu=native+dotprod+i8mm+nosve+nosme` in ggml-cpu. Those settings are not new speed-ups.
The portable macOS release currently sets `GGML_NATIVE=OFF`; this experiment does not change release policy.

`GGML_LTO` sets IPO within `ggml/src`, so it does not enable IPO for the parent's common code, families, API or CLI.
The installed CMake AppleClang module selects `-flto=thin`. Here “all host targets” describes scope, not LLVM FullLTO.
The CPU feature detector retains its existing `-fno-lto` exception. The whole-target variant includes the separate
CLI and shared-library links. No new flags enable fast math, reassociation or unrestricted FP contraction; existing
per-source `-ffp-contract=off` options remain in effect.

Metal shader source, embedded shader bytes, shader compile options and dispatch remain unchanged. Host-side Metal
code is part of the host compiler experiment. The driver compares generated embedded `.metal` hashes, rather than
calling a changed host executable proof that the GPU shader is unchanged.

## Toolchain and build provenance

Read-only inspection on 2026-11-07 found Apple clang 21.0.0 (`clang-2100.3.34.2`), Apple LLVM profdata 21.0.0,
Xcode 27.1 (`27A9275`), macOS SDK 27.0 and CMake 4.4.4. These are observed versions, not a completed pin.
`build.py pin` later records compiler/tool paths and hashes, version output, SDK path/version/settings hashes,
host OS, architecture and Python. The compiler invocation retains the `clang++` name even when it is a symlink.
Every build verifies the same pin before configuration. It records the empty deployment-target setting used by the
anchor and compares the resulting Mach-O SDK/minimum-OS load commands across the actual binaries.

All variants use fresh directories, the same source commit, native/O3 settings, explicit SDK, arm64 and bounded `-j4`.
The driver refuses inherited compiler, ggml and speech experiment settings. It disables ccache equally for all builds.
The IPO hook fails if the selected compiler cannot link an IPO probe; the probe excludes PGO flags because its
synthetic `main` has no trained profile. Actual profile compilation is separately required to succeed.

Each build saves configure/build commands, the CMake cache hash, compile commands, link commands, generated shader
hashes and binary hashes/sizes. It checks the actual ThinLTO scope in generated compile and final link commands and
the shared library's exported C API. The IPO option alone is not accepted as evidence that IPO took effect.
`comparison-conditions.json` preserves normalized configure, build, actual compile and target link commands. Pairing
requires exact equality after removing only source/build paths and the intended ThinLTO/profile-use flags. Native,
optimization level, all other compiler/linker flags and per-source FP controls remain in the comparison. Compiler IPO
probe links are kept outside this target comparison. The conditions file is hashed and must agree with the manifest's
raw configure/build commands and its reported build flags.
`build.json` is compatible with the campaign runner. Instrumented builds use `instrumented-build.json`, carry
`performance_evaluation_eligible=false`, and are never passed to the performance runner.

The following commands are for a later authorized execution slot. Every pin/build/train/merge invocation takes
`/private/tmp/speech-haas-mac-perf.lock`; each training invocation runs one worker and releases that lock afterward.
Do not wrap these commands in the same lock again.

```sh
SRC=/Users/gotz/local/voimem/speech.cpp/outputs/worktrees/haas-mac-022
OUT=/Users/gotz/local/voimem/speech.cpp/outputs/haas-mac-campaign/022
mkdir -p "$OUT"
/usr/bin/python3 -B "$SRC/experiments/haas-mac/022/build.py" pin --out "$OUT/toolchain.json"
/usr/bin/python3 -B "$SRC/experiments/haas-mac/022/build.py" build --source "$SRC" --toolchain "$OUT/toolchain.json" --variant baseline --out "$OUT/baseline"
/usr/bin/python3 -B "$SRC/experiments/haas-mac/022/build.py" build --source "$SRC" --toolchain "$OUT/toolchain.json" --variant 022a --out "$OUT/022a"
/usr/bin/python3 -B "$SRC/experiments/haas-mac/022/build.py" pair --baseline "$OUT/baseline/build.json" --candidate "$OUT/022a/build.json"
```

Use a separate `022b` directory for `--variant 022b`. The `pair` command checks manifest conditions; the common
runner still verifies the actual binaries and inputs before timing. Freeze the common runner at its then-current
reviewed checkpoint, including the instrumented-build rejection guard. Do not use the older runner inherited in this
worktree. Compare fresh baseline with the fixed campaign anchor before interpreting a candidate difference.

## PGO training data and procedure

Read-only inventory found FLEURS test assets only under `/Users/gotz/local/voimem/outputs/samples` and no FLEURS
train/validation cache under `/Users/gotz/.cache/huggingface`. No matching train/validation manifest was found under
the two projects' `outputs` directories. This is an inventory of those locations, not a claim about every disk.
The existing test metadata pins `google/fleurs`, `ja_jp`, revision
`70bb2e84b976b7e960aa89f1c648e09c59f894dd`.

At a later execution slot, select a small official train or validation subset at that revision and record the
archive and metadata URLs and SHA-256. Start with nine complete recordings, three each of 3–10, 10–20 and 20–30
seconds. Do not crop, relabel or reuse an evaluation recording. Audio, models, archive files, PCM and profiles stay
under `outputs/` and are not committed. Downloading and PCM preparation are not part of this source checkpoint.

The training JSON contains:

- `dataset`: `repository`, `revision`, `config`, `split` (`train` or `validation`), archive/metadata URLs and
  `archive_sha256`/`metadata_sha256`.
- `sample_rate: 16000`, `format: "mono PCM16 little endian"` and `records`.
- Each record: `file`, `source_recording_id` (the original WAV filename), official utterance `id`, `pcm_path`,
  `pcm_sha256`, `samples`, `source_samples` (the whole recording's sample count), `start_sample: 0`, `seconds`,
  `stratum`, and `reference`.

`pgo.py audit` rejects overlap with all 650 quality recordings, the 30 latency subset, the three stage recordings,
the six short prefixes and their original recordings, using recording names/IDs and PCM hashes. It also excludes
shared official utterance IDs, which is stricter than excluding the same recording. Training verifies actual PCM
hashes and lengths again before inference. Selection must be frozen before any PGO evaluation result is inspected.

Build `022c-train` with an explicit LTO mode. For example, `--lto host` uses 022b as its non-PGO control, even if 022b
alone showed no gain. Training uses `-fprofile-instr-generate -fprofile-update=atomic`; profile-use builds use
`-fprofile-instr-use` and reject out-of-date profile diagnostics. Counters do not belong in the evaluated binary.
The translation units under `checks/` and `measure/` omit profile use because their separate `main` functions are not
the trained worker's entry point. Their linked production libraries still use the profile, and the same ThinLTO
scope remains in effect. The generated compile-command audit requires this exclusion only for those diagnostic
sources. The actual compiler behavior of this source-only setup remains to be checked.

Run `pgo.py train` separately for Q4_K, Q8_0 and F16 with the same subset and a frozen current `worker.py` path.
It calls the existing worker harness with GPU encoder, CPU decoder, four threads and `phase=pgo_train`.
Each process performs the training records once as warmup and once again; both passes train the compiler and neither
is evaluated for latency. `LLVM_PROFILE_FILE` uses separate module/process names and is recorded outside the worker's
runtime experiment overrides. Normal worker exit must produce nonempty profiles. Each run preserves its worker
output, explicit profile environment, model, Python, subset and binary identities.

Merge exactly those three `train.json` files with the same pinned `llvm-profdata`. The resulting `profile.json`
binds the raw profile hashes, source commit, LTO mode, toolchain, training and evaluation manifests. Build `022c` with
that profile and the same LTO mode, then use `build.py pair` with the corresponding non-PGO control and the actual
evaluation `--assets` and `--short` manifests. Their hashes must match the sets excluded during training. No source or
compiler change is allowed between training and profile use. Untrained application paths can remain unprofiled;
their performance is outside the Haas workload claim.

## Validation and measurement boundary

`check.py` exercises recipe, comparison and disjointness guards with synthetic metadata only. It does not prove that
the compiler accepts these options, that LTO/PGO improves anything, or that model outputs remain unchanged.

After build authorization, first compare old/new CPU F32 stage outputs and state/logits with the available official
references and differential checks, then the Metal encoder/final projected outputs and complete TDT output. Preserve
existing arithmetic thresholds; do not relax them to accommodate a compiler change. The compiler experiment includes
common host code, so passing Haas alone does not validate every other family or the portable shared-library release.

Initial timing is the same nine recordings in three alternating warm rounds for Q4_K/Q8_0/F16, GPU encoder and CPU
decoder four threads, with `/usr/bin/python3` 3.9.6 and the frozen strict runner. Record stage times, warm request
p50/p90/p99, RTFx, first request/load time, process peak memory and binary size. Small samples do not establish stable
population tail latency. Expand a promising variant to the unchanged 30-recording/full-650 comparison and short-six
supplement only after coordinator review. A compiler change that alters output is investigated before further timing.

No candidate is adopted at this checkpoint. Official-reference, Vulkan/Windows and other-platform verification are
not replaced by metadata guards or a Mac-only comparison. A successful experiment still needs a separate production
change and a fresh direct comparison before adoption.

Primary mechanism references: [CMake IPO initialization](https://cmake.org/cmake/help/latest/variable/CMAKE_INTERPROCEDURAL_OPTIMIZATION.html),
[IPO support checks](https://cmake.org/cmake/help/latest/module/CheckIPOSupported.html),
and [Clang profile-guided optimization](https://clang.llvm.org/docs/UsersManual.html#profile-guided-optimization).
