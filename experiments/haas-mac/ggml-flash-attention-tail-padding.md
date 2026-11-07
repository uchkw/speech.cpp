# ggml Metal flash-attention tail padding — correctness-only validation

Date: 2026-11-07. Status: **CORRECTNESS-ONLY METAL VALIDATION COMPLETE.** The
commands below are the procedure as executed on an Apple M2 Ultra on 2026-11-07, in
the order shown: each configure and build command ran once, and each of the four
measured hardware test runs ran once; the measured outcomes beside them met the
pre-run predictions. The validation inputs are synthetic arrays only, and nothing
here claims a parent commit SHA, a whole-model result or release acceptance.

## Identities

- Measured parent context: `114c514430c8aa1455c077819f8e72e674a48b75`; its preserved
  evidence parent is `6afe21a2fc87bea75ead59b492eb6ab9f69517b8`.
- This clean parent change is based on `03555100287c118b41d1b475044e5ab1e07baa8c`
  and keeps the same production runtime source and ggml pin
  `d27e2a6bb32cf7e7e188375e2f58e7f6e1a5e524` as the preserved evidence parent; no
  configure, build or test was rerun for this PR-scope/metadata correction.
- Public ggml runtime base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Old source, cases only: `082fda79273680640bc74803f0c4561bb3dfc1fa`, already checked
  out detached at `outputs/worktrees/ggml-fa-tail-pad-cases`.
- New source, fixed: `d27e2a6bb32cf7e7e188375e2f58e7f6e1a5e524`, pushed to the
  public branch `codex/metal-fa-tail-pad` of `uchkw/ggml` (public `nyosegawa/ggml`
  upstream), checked out at `outputs/worktrees/ggml-fa-tail-pad/ggml`.
- Both ggml trees were clean at execution, and the test files are identical
  between the two checkpoints
  (`git diff --exit-code 082fda79273680640bc74803f0c4561bb3dfc1fa d27e2a6bb32cf7e7e188375e2f58e7f6e1a5e524 -- tests`
  was checked), so the test definitions are held constant between the two builds.
  The standalone pad check takes deterministic inputs, while the normal FA regression
  randomizes its inputs independently in each process and compares them against that
  run's CPU reference; test-source identity alone therefore supports no cross-binary
  numerical attribution.

## Change under validation

Source-only review passed, after scope and style corrections. The defect contract is
ordinary tensor padding: copy the logical row bytes, not `nb11`/`nb21` stride bytes,
and zero the remaining destination pitch. Two host fields carry `ggml_row_size` of the
actual F16-converted or original K/V types. Source strides, destination pitches,
scratch offsets, masks, attention arithmetic and quantized conversion are unchanged.

## Validation inputs and claims

The standalone pad check uses only valid allocated deterministic synthetic arrays
with spare source storage, read through a padding-byte oracle; there is no
fault-provoking input. The normal FA check uses independently randomized synthetic
arrays and its own CPU reference in each process. This validation makes no
numerical-output-defect claim and no runtime-fault claim: the old build's failures
are padding/scratch content mismatches, not an observed fault and not evidence that
attention output was wrong.

## Executed procedure

Run from the repository that owns `outputs/`, with task-specific variables only:

```sh
validation_root="$(pwd -P)"
validation_old_source="$validation_root/outputs/worktrees/ggml-fa-tail-pad-cases"
validation_new_source="$validation_root/outputs/worktrees/ggml-fa-tail-pad/ggml"
validation_artifacts="$validation_root/outputs/haas-mac-flash-tail-pad-20261107"
```

`builds/old`, `builds/new` and `logs` were created fresh under `validation_artifacts`.
Each configure and build command below ran once, and each of the four measured
hardware test runs ran once, sequentially, in the order shown, under the compute
lease, with stdout, stderr and the actual exit code saved as separate files, for
example
`cmd >"$validation_artifacts/logs/<step>.out" 2>"$validation_artifacts/logs/<step>.err"; printf '%s\n' "$?" >"$validation_artifacts/logs/<step>.exit"`.

Configure, old then new (both exit 0):

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake -S "$validation_old_source" -B "$validation_artifacts/builds/old" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BUILD_TESTS=ON -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=OFF
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake -S "$validation_new_source" -B "$validation_artifacts/builds/new" -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_BUILD_TESTS=ON -DGGML_BACKEND_DL=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=ON -DGGML_OPENMP=OFF
```

Build, old then new (both exit 0):

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake --build "$validation_artifacts/builds/old" -j4 --target test-metal-flash-attn-pad test-backend-ops
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock cmake --build "$validation_artifacts/builds/new" -j4 --target test-metal-flash-attn-pad test-backend-ops
```

The test commands ran only once all four configure and build exit codes were zero.
The old build's pad-byte exit 1 stays distinct from that requirement: it is a
predicted test outcome, assessed with the case matrix below, not a build failure, and
it does not make all 20 old cases failures. Every raw outcome was preserved as
recorded.

Pad-byte check, old then new, as separate commands:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock "$validation_artifacts/builds/old/bin/test-metal-flash-attn-pad"
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock "$validation_artifacts/builds/new/bin/test-metal-flash-attn-pad"
```

Normal FA regression against the CPU reference, old then new, as separate commands:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock env LLAMA_TEST_FA_VEC_DISABLE=1 "$validation_artifacts/builds/old/bin/test-backend-ops" test -o FLASH_ATTN_EXT -p tail_pad=1 -b MTL0 -j 1
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock env LLAMA_TEST_FA_VEC_DISABLE=1 "$validation_artifacts/builds/new/bin/test-backend-ops" test -o FLASH_ATTN_EXT -p tail_pad=1 -b MTL0 -j 1
```

`LLAMA_TEST_FA_VEC_DISABLE=1` was set on both builds for the same reason: it
suppresses only the unrelated additional slice that ignores the params filter. The
inherited `GGML_`/`SPEECH_`/`LLAMA_` settings were empty, so that variable was the
runs' only override. Error tolerances are unchanged.

## Acceptance rules and measured outcomes

The rules below were fixed before the runs and applied to what the runs printed; each
table keeps the prediction beside the actual result. Unexpected outcomes are retained
and investigated, never forced to match.

**Standalone pad-byte check.** The test has 20 cases, crossing two element types (F32
and F16), two query counts (1 and 20) and five shapes/layouts. The rules: the old
build gives 8 OK controls (4 interleaved no-tail with keys 64, and 4 contiguous tail
with keys 65) and 12 byte-mismatch FAILs (4 interleaved with keys 65, 4 interleaved
unequal-head-width with keys 33, 4 strided with keys 65), exiting 1, with all 20 case
identities, zero mask mismatches, only the expected K/V gap mismatches and no
allocation, backend or command errors; the passing controls must remain passes, so
not all 20 cases are failures. The fixed build gives 20/20 OK, K/V/mask mismatch
counts all 0, exit 0.

| Check | Prediction | Old actual (`082fda79`) | New actual (`d27e2a6b`) |
|---|---|---|---|
| Standalone pad-byte, 20 cases | Old: 8 OK, 12 FAIL, exit 1. New: 20/20 OK, exit 0. | 8 OK (4 interleaved no-tail keys 64, 4 contiguous tail keys 65), 12 FAIL (4 interleaved keys 65, 4 interleaved unequal dk128/dv96 keys 33, 4 strided keys 65), exit 1; mask mismatch count 0 in every case, K/V mismatch counts positive in the failures | 20/20 OK, exit 0; K/V and mask mismatch counts all 0 |

Both actual results match the prediction.

**Normal FA regression.** Independent evidence: old and new each run all 24 selected
cases and pass 24/24 against the CPU reference, with no unsupported or skipped cases
and exit 0. The runner excludes unsupported and skipped cases from the summary
denominator, so a zero exit code alone is insufficient; the case counts were read.
These checks may pass on both versions; the pad check's mismatches remain the
evidence of the padding contract either way. The BLAS and CPU backends printed
`Skipping` because `-b MTL0` excludes them, which is backend selection, not case
skipping, and the CPU reference comparison still ran for each selected Metal case.

| Check | Prediction | Old actual (`082fda79`) | New actual (`d27e2a6b`) |
|---|---|---|---|
| Normal FA regression, 24 selected cases | Each side 24/24 OK, exit 0, unsupported 0, skipped 0 | 24/24 OK, exit 0, unsupported 0, skipped 0 | 24/24 OK, exit 0, unsupported 0, skipped 0 |

Each run covered F32 8, F16 8, Q4_0 4 and Q8_0 4 cases. Normal FA inputs randomize
independently per process, so these passes do not prove old/new bitwise equivalence.
On this pre-M5 device both the vectorized and the non-vectorized path are covered
through the query counts of the existing fixtures: 1 and 20 in the standalone pad
check, 1 and 32 in the quantized regression cases.

**Sandbox exclusion.** An initial sandboxed old pad check exited 1 before running any
cases, with Metal unavailable. Its raw evidence is preserved separately under
`logs/old-pad-bytes.sandbox.*` and is excluded from the old-failure proof above. The
same unchanged binary then ran through the ordinary approved escalated hardware
execution, and all four measured test runs above used actual Metal hardware.

## Environment and evidence

- Device MTL0, Apple M2 Ultra, where the tensor API is unavailable/disabled as on all
  pre-M5 hardware; shared Metal buffers were enabled. macOS 26.7.2 build25G332; Apple
  clang 21.0.0 (clang-2100.3.34.2), arm64-apple-darwin25.6.0; CMake 4.4.4.
- Two separate builds under the artifact root, `builds/old` and `builds/new`, both
  Release with identical flags and `-j4`, all four configure and build commands under
  the same lock; all exited 0.
- Artifact root (untracked, not committed): `outputs/haas-mac-flash-tail-pad-20261107`
  in the main repository, holding four distinct evidence sets — old-pad-bytes,
  new-pad-bytes, old-fa-reference, new-fa-reference — under
  `logs/{old,new}-pad-bytes.hardware.{stdout.log,stderr.log,exit,started-utc.txt,ended-utc.txt}`
  and the `logs/{old,new}-fa-reference.hardware` equivalents, with the configure and
  build outputs saved likewise. Binary hashes in `binaries.sha256`, raw log hashes in
  `logs.sha256`; `macos.txt`, `compiler.txt`, `cmake.txt`, `environment.json` and
  `old.commit`, `old.status`, `new.commit`, `new.status` retained.
  `completed-utc.txt` is `2026-11-07T00:30:20Z`.
- The initial `started-utc.txt` and the configure-old start and end timestamps are
  empty because `/usr/bin/date` was absent; they are preserved as unknown, with
  `timestamp-capture-warning.txt`, and the later commands used `/bin/date`.
  Configure-old's stdout, stderr and exit 0 are intact: nothing was rerun solely to
  fill the missing timestamps, and no timestamps were invented.

## Limits

Real speech and model-stage checks, recognition accuracy, transcripts, performance
impact and Windows/Vulkan hardware are untested; the full stage checks and the
Windows machine's Vulkan run are still pending. This is correctness-only Metal
evidence, not whole-model validation and not release acceptance. The compute lease
was returned after the tests, before trial014.

## After validation

This parent change pins ggml to `d27e2a6bb32cf7e7e188375e2f58e7f6e1a5e524` and
records the correctness-only validation. The coordinator alone pushes.
