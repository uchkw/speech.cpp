# Haas Core ML encoder probe (experiment 034)

This directory is a source-only checkpoint of an independent experiment for the Haas Mac campaign:
it converts speech.cpp's pinned Haas-ja-v1.0 official NeMo frontend, encoder and `RNNTJoint.project_encoder`
into Core ML ML Programs and checks them against the official CPU F32 reference dumps and through speech.cpp's
unchanged CPU-4 TDT decoder. The official modules are restored unchanged from the pinned BF16 archive into F32,
with length and masking preserved, and nothing here replaces an official module, re-derives a checkpoint from
GGUF, chunks audio, introduces a new precision or changes padding, masking or attention. Nothing in this
checkpoint has been executed: no environment has been resolved, no conversion has run, no prediction has been
made and no number has been measured.

## What is here

The inherited draft's thirteen files, one new admission driver, plus this documentation:

| File | Responsibility |
| --- | --- |
| `pins.json` | The pins: runtime base, ggml gitlink, reference commit, checkpoint, the F16 TDT control model (file, revision, size, SHA-256), converter candidate, the eleven cases |
| `env/pyproject.toml`, `env/.python-version` | The candidate converter environment, to be copied under `outputs/` and resolved there |
| `contract.py` | Identities, the shape contract, the per-invocation disk preflight and the shared identity helpers (standard library only) |
| `numerics.py` | Finite F32 array reading, SNR differences, stage gates, typed output validation |
| `export.py` | Restores the official stages, verifies F32 replay, traces, converts, and records the identity of everything it produced and used |
| `check.py` | Checks the Core ML artifacts against the official dumps per precision, kind and compute unit, after verifying the export's identities |
| `check_native.py` | Verifies the native runner's raw normal projection is bit-exact with the same normal Python artifact and binds the compiled package and both files' identities |
| `Run.swift` | Native probe: anticipated compute plan (`plan`), serial prediction (`predict`) and signposted runs (`trace`); binds the exact package it compiles |
| `tdt-check.cpp`, `CMakeLists.txt` | The standalone C++ check that runs the candidate projection through the unchanged CPU-4 TDT decoder |
| `check_tdt.py` | Freezes the TDT harness's build receipt and admits the verified candidate to that harness (standard library only) |
| `check_contract.py` | Standard-library fixtures for identities, shape guards, the disk plan and the admission gates |
| `check_numerics.py` | NumPy fixtures for the numeric gates (run in the frozen environment) |
| `README.md`, `evaluation.md`, `source-status.html` | This recipe, the checkpoint's status, the static status page |

## Hypotheses, and what each stage measures

- **Replay (export.py)** restores the pinned archive into the official modules and runs them on the CPU in F32.
  It checks that the official frontend reconstructs the dumped features, that the encoder and both projection
  paths replay the dumps above 90 dB, and that tracing changed nothing. It measures correctness of the
  restoration, not speed.
- **Conversion (export.py)** traces the unchanged `Wrapper` (normal and diagnostic) and the isolated
  `Projection`, and converts each to F32 and F16 ML Programs with flexible or enumerated input shapes. It
  measures that the conversion contract (names, dtypes, shapes, deployment target) holds, not that Core ML is fast.
- **Stage check (check.py)** predicts each case on each converted artifact and compares with the official dumps.
  F32 is a control (above 90 dB everywhere); F16 has gates only where the arithmetic of half precision explains
  a bound (encoder above 25 dB, isolated projection above 40 dB). The F16 composed error is recorded and never
  gated: no composed gate is invented from the isolated one, and higher precision alone is not evidence of
  better recognition. Its `python_prediction_seconds` is a correctness-run figure, not benchmark evidence.
- **Native probe (Run.swift)** compiles the normal artifact, records `MLComputePlan`'s anticipated device usage
  (`plan`), then predicts serially (`predict`) or under signposts (`trace`), recording compilation, cold load,
  input preparation, prediction and host read-back separately, first visits apart from repeats. It measures the
  synchronous prediction API of the normal artifact only, and it records the identity of the exact package at
  compilation.
- **Native parity (check_native.py)** checks the native raw projection is bit-exact with the same normal Python
  artifact, which is what makes the projection fed to the TDT check verifiably the candidate that was checked;
  its record carries the projection files' hashes and requires the compiled package to be identical to the
  checked and exported one.
- **TDT admission (check_tdt.py)** freezes the harness's build receipt at build time and admits a candidate only
  when the successful parity record for that manifest, the manifest/check/run identities (each bound to the
  supplied manifest file's own SHA-256), the exact ordered case list, the projection files' hashes, the pinned
  GGUF SHA-256 and the frozen receipt all match, with the tracked runtime and build inputs at HEAD identical
  to the pinned base; it then requires the harness's exit status and full output coverage, keeping the raw
  streams and exit metadata as evidence either way.
- **TDT check (tdt-check.cpp)** decodes the candidate projection and the ggml normal encoder projection with the
  unchanged unboosted greedy CPU-4 TDT decoder of the pinned speech.cpp and compares every decision: token ids,
  frames, durations, graphs and progress. The recipe reaches this standalone harness only through
  `check_tdt.py`. It is core correctness, not a worker request benchmark and not an official Mac KDA/TDT
  reference.

No stage measures worker end-to-end latency, and no stage establishes on which silicon Core ML actually ran.

## Pins and provenance

| Pin | Value |
| --- | --- |
| speech.cpp runtime base | `114c514430c8aa1455c077819f8e72e674a48b75` |
| ggml submodule gitlink | `e1a9ac12edbb57402618fd6936598603f5ee6266` |
| Reference checkout | `72cc29a01567aee120a990cc6df28fbf18a2232e` (`reference/fastconformer` clean at that commit) |
| Checkpoint | `third-intelligence/Haas-ja-v1.0` @ `660728d39019e5e065046154beca44ac9a66a540`, `Haas-ja-v1.0.nemo`, 1,232,384,000 bytes, SHA-256 `c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa` |
| TDT model | `pins.json`'s `tdt_model`: the F16 decoder control `Haas-616M-ja-v1.0-F16.gguf` at revision `6b965d5007ba780585de51549d57a8c1c0d67c7e`, 1,233,128,832 bytes, SHA-256 `f752e02f68c1dfff795900e40aa82e5302976fb5360080fcbd330d8b3c992cd8`, from the campaign's verified asset descriptor |
| Converter candidate | Python 3.11.11, coremltools 9.0 (`coremltools-9.0-cp311-none-macosx_11_0_arm64.whl`), torch 2.7.0, nemo-toolkit 2.7.3 |

The checkpoint pin is `pins.json`'s `checkpoint` object and every check compares content, not paths:
`haas_checkpoint.load_checkpoint()` verifies the archive's size and SHA-256, `restore_stages()` strictly loads the
official `AudioToMelSpectrogramPreprocessor`, `ConformerEncoder` and `RNNTJoint` in F32 from the original tensors,
and export.py re-hashes the official module source files and compares them with each dump's recorded provenance,
so a regenerated dump or a swapped artifact is caught by content. The export's manifest also records each
generated package's relative file inventory with SHA-256, the exact dump files used per case (meta, features,
encoded, joint_encoded, audio, hashed as used) and a snapshot of the checker sources; every later gate — the
stage check, the native parity check and the TDT admission — re-derives those identities and requires the same
content before accepting new outputs, so a path or a version alone never stands in for contents. The TDT
control GGUF's identity — exact file name, revision, byte size and SHA-256 — comes from the campaign's verified
asset descriptor (its F16 model row), never from the file under test, and is recorded in `pins.json`'s
`tdt_model` at this checkpoint, so the exported manifest embeds the same pin; the weights themselves are not
supplied with this source checkpoint.

The converter candidate has **not** been resolved or installed, and **no `uv.lock` has been generated or
committed**. The reference dumps were produced by the reference environment's own pinned stack; the candidate's
torch 2.7.0 differs from it, and the replay gates above 90 dB are the control that would catch a divergence.
A Python environment, Swift and C++ API/link compatibility with the pinned bases, the Core ML conversion and
output correctness are all unverified at this checkpoint.

## Shape contract and numeric boundaries

Inputs are `features` F32 `[1,128,P]` and `length` I32 `[1]` holding the valid frames `T`. The normal artifact
returns `joint_encoded` F32 `[1,Q,640]` and `encoded_length` I32 `[1]`; the diagnostic artifact adds `encoded`
F32 `[1,1024,Q]`; the isolated projection takes the official `encoded` `[1,U,1024]`. Valid frames are never
cropped before the model: the padded extent and the length input preserve NeMo's masking, and outputs are
validated over the padded shape before the valid rows are cut. The fixed set is `P=1008` with the two valid
pairs (T=996, U=125) and (T=1002, U=126); the enumerated set is the eleven cases over ten padded `P` shapes in
`pins.json`. Flexible and enumerated input shapes follow coremltools' guidance
(https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html), and input and output tensor types
follow its model input and output types guide
(https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html).

| Comparison | F32 | F16 |
| --- | --- | --- |
| Replay, trace and converted controls (frontend, encoder, composed, isolated) | > 90 dB | — |
| Diagnostic encoder (converted) | > 90 dB | > 25 dB |
| Isolated projection (converted) | > 90 dB | > 40 dB |
| Composed normal projection (converted) | > 90 dB | recorded only, no gate |
| Normal versus diagnostic composed output | compared independently, recorded | compared independently, recorded |

Normal, diagnostic and isolated artifacts stay distinct everywhere: separate `.mlpackage` files, separate output
directories per precision, kind and case, records that carry `kind` and `precision`, and timing scopes that say
what each number measures.

## Disk preflight

Before each invocation, `contract.disk_preflight` compares the volume's live free space with that invocation's
planning estimate and stops if the capacity is insufficient; every invocation re-checks live capacity, because
an observation from an earlier step is stale by definition (about 46 GiB free was observed on the machine during
preparation, and that observation is stale too). The estimates are transparent planning constants, not
measurements and not guarantees: package sizes are not pinned exactly and the converted models' sizes follow
from the shapes converted, so the numbers are deliberately high and the reserve covers the system and the
smaller writes around each step. What an invocation budgets is only what it still adds: the pinned archive,
1,232,384,000 bytes, and its dumps are already on the disk before the environment step, and no step budgets
them again. Export budgets the six `.mlpackage` directories it writes — the normal and diagnostic encoders in
F32 and F16, which carry the weights, and the much smaller isolated joint in each precision — together with the
temporary work of restoring the archive and converting. A native invocation compiles the package to a
temporary copy and retains one compiled copy, so its possible two-copy peak is budgeted, and `plan`, `predict`
and `trace` are separate invocations that each preflight again. The runner's `trace` mode only emits signposts;
an actual Instruments trace is a separate later step with its own row, budgeted only when a capture is
collected. Each run record carries the planned estimate and the observed free bytes side by side, keeping
estimation and actuals distinct.

| Step (GiB) | what the invocation adds | total |
| --- | --- | --- |
| environment | environment 8, cache 4, reserve 8 | 20 |
| export | packages 8, conversion temporaries 4, run data 1, reserve 8 | 21 |
| check | compiled temporaries 2, check outputs 1, reserve 8 | 11 |
| native | compiled temporary 2, compiled kept 2, native outputs 1, reserve 8 | 13 |
| instruments | trace 4, reserve 8 | 12 |

There is no exact package size and no unbounded artifact matrix here: one environment, one conversion per shape
set, one compiled copy per native run.

## Ordered recipe

**None of the commands below has been run.** Every block is future execution, gated by an assigned campaign
compute lease and held under the macOS campaign lock; builds use `-j4` and the TDT harness uses four CPU
threads. Each block stops at its first failure — do not continue past a failed fixed-set stage or case. The
setup block is pasted first and is the whole session's one fail-fast switch: `set -eu` (POSIX `sh` and `zsh`
alike) plus the `lock()` definition. On top of that, every prerequisite gate in every block — preflight,
`cd`, copy, environment resolve and sync, the fixture runs, the builds and the receipt freeze — ends in
explicit successful `&&` chaining or `|| exit 1`, so no pasted block can continue past an earlier failure even
where `set -e` exempts a failing `&&` operand. All
generated artifacts live under the campaign checkout's ignored `outputs/haas-mac-campaign/034-coreml` directory
— never in this tracked source, which `contract.py`, `Run.swift` and `check_tdt.py` enforce — and the setup
block's fail-fast `git check-ignore` verifies that ignore before anything is generated; the experiment worktree
itself does not ignore `outputs/`, so the recipe writes nothing under it.

The worktree holds the experiment's own commit, a descendant of the runtime base that carries these 034
sources, with the runtime subtree clean and ggml at its gitlink; it is not a bare checkout of the earlier base.
The remaining plaintext placeholders are externally supplied prerequisite inputs only: the worktree, the
reference checkout, the pinned archive, the dumps and the GGUF. The F16 control GGUF's pinned identity comes
from the campaign's verified asset descriptor and is already recorded in `pins.json`'s `tdt_model`; it is
never computed from the file being checked. zsh does not word-split unquoted variables, so a command
string held in a variable is not a reliable recipe; every heavy command runs through one lock function:

```sh
set -eu                                # one fail-fast switch for the whole pasted session (POSIX sh and zsh)
WORKTREE=/path/to/worktree             # at the experiment commit (a descendant of 114c514430c8aa1455c077819f8e72e674a48b75), ggml at e1a9ac12edbb57402618fd6936598603f5ee6266
EXP="$WORKTREE/experiments/haas-mac/034"
CAMPAIGN_CHECKOUT=/path/to/campaign-checkout # the outer campaign checkout holding the campaign artifact directory (not the experiment worktree)
OUT="$CAMPAIGN_CHECKOUT/outputs/haas-mac-campaign/034-coreml"
git -C "$CAMPAIGN_CHECKOUT" check-ignore -q -- outputs/haas-mac-campaign/034-coreml/preflight || exit 1 # fail fast unless the campaign checkout ignores the artifact directory
REF=/path/to/reference-checkout        # speech.cpp at 72cc29a01567aee120a990cc6df28fbf18a2232e, reference/fastconformer clean
CHECKPOINT=/path/to/Haas-ja-v1.0.nemo  # pinned by pins.json: 1232384000 bytes, SHA-256 c6ed3869…be8458aa
DUMPS=/path/to/dumps/Haas-ja-v1.0      # dump-encoder.py output; per-case folders named by the case ids
MODEL=/path/to/Haas-616M-ja-v1.0-F16.gguf # pinned by pins.json's tdt_model: 1233128832 bytes, SHA-256 f752e02f…c992cd8
lock() { /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock "$@"; }
```

1. **Standard-library source fixtures** (no environment, no lease needed; they include the admission-gate
   fixtures, which need no model, compiler, NumPy or subprocess):

   ```sh
   cd "$EXP" && python3 -B check_contract.py || exit 1
   ```

2. **Preflight the environment step**, then copy the project and freeze the environment under `outputs/` — a
   freshly created ignored directory, never the user's Python environment and never the tracked source. The
   resolution and the sync are heavy and run under the lock; later runs use the frozen environment and never
   re-resolve packages:

   ```sh
   mkdir -p "$OUT/env" || exit 1
   cd "$EXP" && python3 -B -c "import json; from pathlib import Path; from contract import disk_preflight; print(json.dumps(disk_preflight('environment', Path('$OUT'))))" || exit 1
   cp "$EXP/env/pyproject.toml" "$EXP/env/.python-version" "$OUT/env/" || exit 1
   cd "$OUT/env" && lock uv lock && lock uv sync --frozen || exit 1
   ```

   The lock is a candidate resolution of the pins above and stays under `outputs/`; nothing is committed.

3. **NumPy fixtures in the frozen environment:**

   ```sh
   cd "$EXP" && uv run --frozen --project "$OUT/env" python -B check_numerics.py || exit 1
   ```

4. **Export the fixed set** (CPU model work and conversion; lease and lock required; the step pref lights
   itself and records the identity of everything it produces and uses):

   ```sh
   lock uv run --frozen --project "$OUT/env" python "$EXP/export.py" \
     --reference-source "$REF" --dumps "$DUMPS" --checkpoint "$CHECKPOINT" \
     --lock "$OUT/env/uv.lock" --shape-set fixed --out "$OUT/export-fixed" || exit 1
   ```

5. **Stage check under each compute-unit option** — all four of `CPU_ONLY`, `CPU_AND_GPU`, `CPU_AND_NE` and
   `ALL` are the allowed set, `CPU_AND_NE` includes the CPU, and each invocation pref lights itself; the loop
   stops at the first failing case:

   ```sh
   for units in CPU_ONLY CPU_AND_GPU CPU_AND_NE ALL; do
     lock uv run --frozen --project "$OUT/env" python "$EXP/check.py" \
       "$OUT/export-fixed/manifest.json" --units "$units" --out "$OUT/check-fixed-$units" || exit 1
   done
   ```

6. **Native probe** — compile the runner with `-parse-as-library` (it is one file with `@main`), preflight,
   then plan, predict and trace with the argument order `MANIFEST PRECISION UNITS NEW_OUTPUT_DIR MODE ROUNDS`
   (`predict` needs at least three rounds). The three modes are distinct invocations, each pref lighted again
   by the recipe line before it — the runner itself does not check the disk budget, so a direct invocation
   skips that gate; never invoke it without the preflight line immediately before it:

   ```sh
   mkdir -p "$OUT/native" || exit 1
   lock swiftc -O -parse-as-library -o "$OUT/native/Run" "$EXP/Run.swift" || exit 1
   for mode in plan predict trace; do
     rounds=1; [ "$mode" = predict ] && rounds=3
     cd "$EXP" && python3 -B -c "import json; from pathlib import Path; from contract import disk_preflight; print(json.dumps(disk_preflight('native', Path('$OUT/native'))))" || exit 1
     lock "$OUT/native/Run" "$OUT/export-fixed/manifest.json" f32 CPU_AND_NE "$OUT/native-$mode-f32-ne" "$mode" "$rounds" || exit 1
   done
   ```

   Repeat for `f16` and the other compute-unit options as the measurement plan asks; the probe needs macOS 14.4
   or newer, while the ML Programs themselves deploy to macOS 13.

7. **Raw-output verification** — the native run must be paired with the Python check directory recorded with the
   same compute units, and it fails unless every case's raw projection is bit-exact with that same normal Python
   artifact and the compiled package is identical to the checked and exported one:

   ```sh
   lock uv run --frozen --project "$OUT/env" python "$EXP/check_native.py" \
     "$OUT/export-fixed/manifest.json" "$OUT/check-fixed-CPU_AND_NE" "$OUT/native-predict-f32-ne" || exit 1
   ```

8. **TDT comparison** — build the unchanged CMake harness against the pinned speech.cpp, freeze its build
   receipt at build time, then let the admission driver verify every record, construct the pairs itself (the
   exact Python normal artifact that `check_native.py` proved bit-exact with the native raw output, one
   `features.npy`/`joint_encoded.npy` pair per case, in manifest order) and run the harness:

   ```sh
   lock cmake -S "$EXP" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release || exit 1
   lock cmake --build "$OUT/build" -j4 --target haas-coreml-tdt-check || exit 1
   lock python3 -B "$EXP/check_tdt.py" freeze --worktree "$WORKTREE" \
     --executable "$OUT/build/haas-coreml-tdt-check" --receipt "$OUT/build/tdt-receipt.json" || exit 1
   lock python3 -B "$EXP/check_tdt.py" admit --worktree "$WORKTREE" \
     --receipt "$OUT/build/tdt-receipt.json" --manifest "$OUT/export-fixed/manifest.json" \
     --python-check "$OUT/check-fixed-CPU_AND_NE" --native-run "$OUT/native-predict-f32-ne" \
     --executable "$OUT/build/haas-coreml-tdt-check" --model "$MODEL" --device cpu \
     --out "$OUT/tdt-fixed-f32-ne" || exit 1
   ```

   The freeze runs once, immediately after the build and under the same lock: it records the executable's
   SHA-256, the harness sources' hashes and the worktree's runtime state (HEAD a descendant of the runtime
   base with the runtime and CMake inputs clean and their tracked contents at HEAD identical to the pinned
   base — an experiment-only descendant is fine, the whole HEAD need not equal the base — plus ggml's gitlink
   and clean working revision). Admission re-collects that state and verifies the executable bytes, the
   manifest (the checked, native and parity records' manifest hashes each equal the supplied manifest file's
   own SHA-256), the completed successful parity record, the exact ordered case list, the projection files'
   hashes and the pinned GGUF SHA-256 before the harness runs, and it requires exit status zero plus both
   decodings and one true `exact_tdt_decisions` line per pair; the harness's raw stdout, stderr and exit
   metadata are written into the fresh result directory before those assertions, so a rejected comparison
   keeps its evidence. The harness itself remains
   standalone and unchanged; its decoding always runs on the CPU with four threads, and `--device gpu`
   selects only the baseline encoder's device.

9. **Enumerated set** — only after every fixed-set gate has passed, repeat steps 4 through 8 with
   `--shape-set enumerated` and matching output directories, covering the eleven cases over the ten padded
   shapes.

Each run leaves human-inspectable evidence under its new directory: `run.json`, `f32-replay.json` and
`trace-*.json` with the manifest and inputs from export; `check.json` with per-case `metrics.json` and the raw
padded outputs beside them; `plan.json` or `result.json` with the native records beside the compiled model the
run retains; `output-check.json` from the parity check; and `tdt-receipt.json` with `admission.json`,
`harness-run.json` and the raw `harness-stdout.log`/`harness-stderr.log` from the build freeze and the TDT
admission. A failed stage writes `failure.json` naming the stage and the error into
the directory that invocation created — never into an existing directory — and the gate messages name the case
at fault.

## Compute units, deployment targets and placement evidence

The ML Programs are converted with `minimum_deployment_target` macOS 13; the native probe uses `MLComputePlan`
and so needs macOS 14.4 or newer. `plan` mode reports anticipated support and preference per operation with an
estimated relative cost — what Core ML expects, never what executed — as described in the sessions that
introduced the compute plan API (https://developer.apple.com/videos/play/wwdc2022/10027/) and its follow-up
(https://developer.apple.com/videos/play/wwdc2024/10161/).

The runner's `trace` mode only emits signposts for its own predictions in its own process; collecting an actual
Instruments trace is a separate later step; for example `xctrace record` under the same lock, after preflighting
the plan's `instruments` row, writing the `.trace` under `outputs/`. Only a separate, correlated Instruments
trace that covers the Core ML, Neural Engine and GPU instruments together can establish observed hardware
activity, and its availability has limits: the needed instruments depend on the Instruments version, the
machine and macOS (the Neural Engine track in particular), and capture itself perturbs latency. If those
instruments are unavailable, observed placement stays unestablished and the results must say so; nothing in
this checkpoint claims it.

The timing fields keep distinct boundaries and no number here is a sum of other numbers: `compile_seconds`
(MLModel compilation), `load_seconds` (cold load of the compiled model), `input_prepare_seconds` (host
preparation), `prediction_seconds` (the synchronous prediction API of the normal artifact), `host_output_seconds`
(reading and validating the output), `first_visit` against `repeat` phases, and `python_prediction_seconds`
(a correctness run only). Identity hashing and every verification sit outside these timers. Diagnostic and
stage timings are reported separately, no true worker end-to-end time is measured, F16-versus-quantized
arithmetic comparisons are outside this experiment, and no full Mac official reference is claimed — the
official reference in this experiment is the pinned CPU F32 NeMo dumps, nothing else.

coremltools 9.0's own dependency pins at the recorded commit are
https://github.com/apple/coremltools/blob/428d4b2658dfc44194f27f4f36870751be402ff7/coremltools/_deps/__init__.py ,
and the model this experiment converts is described by its authors at
https://github.com/third-intelligence/Haas-ASR/blob/e72638b6e352010c800ffd550af3a51b29e3dbb1/README.md .

## Status at this checkpoint

Nothing of the experiment itself has been executed: no environment has been resolved, no conversion has run,
no prediction has been made and no model or performance number has been measured, and a source checkpoint has
no speed, numerical-model or placement result. The local source checks of this revision were run on 2026-11-07
with `/usr/bin/python3` 3.9.6, and none of them ran an ML tool: the thirteen standard-library test methods of
`check_contract.py` passed, including their synthetic rejection cases; all eight Python files passed an AST
parse; `pins.json` and the pin environment TOML file parsed; the HTML page and the untracked-source whitespace
check passed; and all nine shell blocks of this README parsed under both `sh` and `zsh` syntax checks
(eighteen checks). These are real local measurements of the source tests only, not of the proposed converter.
What changed in this revision is recorded in `evaluation.md`; the acceptance gates that remain, and the order
they run in, are there too.
No `uv.lock`, generated model data, measured report or failed-run placeholder has been created: everything
above this paragraph is source and documentation.
