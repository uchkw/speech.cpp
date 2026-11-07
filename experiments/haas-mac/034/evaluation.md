# Evaluation status: source-only checkpoint

Experiment 034 stands as source and documentation alone. The runnable recipe in `README.md` is future
execution under an assigned compute lease and the campaign lock. Nothing has been measured, so this page
records what changed in this revision, what has actually been checked about the source, and the gates that
remain before any conclusion about the experiment's hypotheses exists. The pins, thresholds and stage
hypotheses are stated once, in `README.md`.

## What changed at this checkpoint

- The disk plan in `contract.py` budgets each invocation for only what that invocation still adds: export
  carries the six `.mlpackage` directories and the conversion's temporary work (the already-present archive
  and dumps are budgeted by no step), a native invocation carries its compiled temporary and retained copy at
  the possible two-copy peak, `plan`/`predict`/`trace` are separate invocations each pref lighted again, and
  an actual Instruments trace has its own later row, distinct from the runner's signpost mode.
  `check_contract.py` now rejects free space one byte below each step's plan, at the real step boundaries.
- `export.py` records each generated package's relative file inventory with SHA-256 and the exact dump files
  used per case (meta, features, encoded, joint_encoded, audio, hashed as used) beside the existing
  checker-source snapshot; `check.py`, `check_native.py` and `check_tdt.py` re-derive and verify those
  identities before accepting new outputs, so a path or a version alone never stands in for contents, and
  every hash and verification sits outside the prediction timers.
- `Run.swift` validates its inputs, then creates its result directory exclusively; a `failure.json` is written
  only into a directory this invocation created, so an existing successful or unrelated directory is never
  touched and a fresh failed run keeps its evidence. It binds the exact package identity at compilation and
  removes the temporary compiled copy once the retained copy exists. The recipe compiles it with
  `-parse-as-library`.
- `check_native.py` requires the native run's compile-time package identity to equal the Python check's and
  the export's, and records the projection files' hashes, which is what the TDT admission later binds; its
  parity record also carries the manifest's own SHA-256, so a successful parity cannot be replayed against a
  different manifest.
- `check_tdt.py` is new: a standard-library driver that freezes the C++ harness's build receipt at build time
  and, at admission, requires the completed successful native-vs-Python parity record for the supplied
  manifest (its SHA-256 checked against the actual `--manifest` file, not just record-to-record agreement),
  the exact ordered case list, the projection files' hashes, the pinned GGUF SHA-256 and the frozen receipt
  (HEAD a descendant of the runtime base, the runtime subtree clean and its tracked runtime/build contents at
  HEAD identical to the pinned base — an experiment-only descendant is allowed, the whole HEAD need not equal
  the base — and ggml at its gitlink), then constructs the pairs from the validated records and requires the
  harness's exit status and full output coverage, writing the raw stdout, stderr and exit metadata into the
  fresh result directory before those assertions so a rejected comparison keeps its evidence. Both freeze and
  admission verify the runtime state against the pins, and the freeze runs under the campaign lock. The C++
  harness and its CPU-4 decoding are unchanged, and no TDT or worker contract is reimplemented.
- `pins.json`'s `tdt_model` pin carries the F16 ggml decoder control's exact file name, revision, byte size
  and SHA-256 from the campaign's verified asset descriptor (`Haas-616M-ja-v1.0-F16.gguf`), never computed
  from the file under test; the weights themselves are not supplied with this source checkpoint.
- `README.md`'s recipe fails fast for the whole documented shell session: one POSIX `sh`/`zsh` `set -eu`
  setup pasted first, and explicit `&&` chaining or `|| exit 1` on every prerequisite gate — preflight, copy,
  environment resolve/sync, the fixture runs, the builds and the receipt freeze — with every heavy command
  (freeze included) routed through the one `lock()` and `-j4` preserved; the runner's disk preflight stays in
  the recipe, immediately before each of its invocations.
- `README.md`, this file and `source-status.html` were rewritten together and no longer repeat each other's
  pins, thresholds or hypotheses.

## What is actually supported

The local source checks of this revision were run on 2026-11-07 with `/usr/bin/python3` (3.9.6), and none of
them ran an ML tool: the thirteen standard-library test methods of `check_contract.py` passed, including their
synthetic rejection cases; all eight Python files passed an AST parse; `pins.json` and the pin environment
TOML file parsed; the HTML page and the untracked-source whitespace check passed; and all nine shell blocks of
`README.md` parsed under both `sh` and `zsh` syntax checks (eighteen checks). These are real local
measurements of the source tests, not of the proposed converter. Everything runtime remains unverified: the candidate environment (Python
3.11.11, coremltools 9.0, torch 2.7.0, nemo-toolkit 2.7.3) is not resolved or installed, no `uv.lock` exists
or is committed, and no NumPy fixture run, Swift or C++ build, heavy import, archive restoration, F32 replay,
Core ML conversion, stage check, native plan or prediction, parity check, TDT execution, Instruments capture
or performance measurement has happened. A source checkpoint has no speed, numerical-model or placement
result.

The two compatibility points once left open for that review are settled: the pinned `src/common/model-file.h:114`
declares `std::string str(const std::string & key) const;`, so `tdt-check.cpp`'s `ModelFile::str()` use matches
the pinned header, and the campaign checkout ignores the external output path
`outputs/haas-mac-campaign/034-coreml` through `outputs/haas-mac-campaign/.gitignore`, which the recipe's setup
block fail-fasts on with `git check-ignore`. Swift and C++ compilation remain unverified.

## Remaining acceptance gates

Each gate runs only after the previous one passes; the fixed shape set comes first and the enumerated set
repeats the same gates over all eleven cases. The numeric thresholds are in `README.md`.

| Gate | Criterion |
| --- | --- |
| G0 | Outer standard-library checks and the independent full-source review pass on this revision (operator-run). |
| G1 | The environment resolves and freezes from the copied project under `outputs/` with `uv lock` and `uv sync --frozen`; the lock stays out of the tracked source. |
| G2 | Export (fixed set) verifies the archive and official module provenance, the F32 replay controls pass, both precisions of all three artifacts convert, and the manifest records their inventories and the reference files used. |
| G3 | The stage check passes on every one of the four compute-unit options at `README.md`'s gates, with the F16 composed error and each normal-versus-diagnostic difference recorded, not gated. |
| G4 | The native plan records `MLComputePlan`'s anticipated device usage and estimated relative cost for the normal artifact, bound to the compiled package's identity. |
| G5 | Native prediction runs are internally stable (repeats bit-identical to first visits) with compilation, cold load, input preparation, prediction and host read-back recorded separately. |
| G6 | `check_native.py` proves the native raw normal projection bit-exact with the same normal Python artifact for every case, with the compiled package identical to the checked and exported one. |
| G7 | The TDT admission accepts the fixed pair: the receipt frozen at build time matches the executable and the runtime, the pinned GGUF matches the model bytes, and the harness reproduces the ggml normal encoder's greedy TDT decisions exactly with full output coverage. |
| G8 | The enumerated set repeats G2–G7 over the eleven cases and ten padded shapes. |
| G9 | Optional observation only: a correlated Instruments capture covering Core ML, Neural Engine and GPU. If the instruments are unavailable, observed placement remains unestablished and no adoption decision depends on it. |

## Next actions

Under a lease and the campaign lock run the recipe's steps in order through the fixed set (G1–G7) and only
then the enumerated set (G8). An adoption decision needs G6 and G7 at
minimum on both shape sets; placement questions need G9 and cannot be answered by the compute-unit option or
the compute plan alone. Results, when they exist, are compact JSON under `outputs/` first, with any report
built from them per the campaign's rules — none exists yet.
