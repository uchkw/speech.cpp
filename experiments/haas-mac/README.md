# Measure Haas changes on this fork

This is the bounded runner for the Haas Mac campaign in [issue 1](https://github.com/uchkw/speech.cpp/issues/1).
It uses the worker protocol and standard-library Python on macOS. General benchmarks belong in speech-bench.
The versioned JSON reports are authoritative; `render.py` makes standalone English HTML without external assets.

Keep models, PCM, build products, transcripts and logs in an ignored artifact directory. Examples below use
`$ARTIFACTS` for that directory and `$PREVIOUS` for the existing precision experiment. Set both to absolute paths.

```sh
python3 -B experiments/haas-mac/compare.py assets --previous "$PREVIOUS" --out "$ARTIFACTS/assets"
python3 -B experiments/haas-mac/compare.py build --source "$BASELINE_SOURCE" --out "$ARTIFACTS/builds/anchor"
python3 -B experiments/haas-mac/compare.py build --source "$CANDIDATE_SOURCE" --out "$ARTIFACTS/builds/candidate"
python3 -B experiments/haas-mac/compare.py run --measure 001a \
  --baseline "$ARTIFACTS/builds/anchor/build.json" \
  --candidate "$ARTIFACTS/builds/candidate/build.json" \
  --anchor "$ARTIFACTS/builds/anchor/build.json" \
  --assets "$ARTIFACTS/assets/assets.json" --types q4_k,q8_0,f16 --out "$ARTIFACTS/001a"
```

`build` refuses a dirty source or a submodule that differs from its pinned commit. It builds Release with native CPU
and Metal enabled, at four compile jobs, and hashes both executables. Build and inference commands hold
`/private/tmp/speech-haas-mac-perf.lock`. Do not compile outside that lock while a timing run is active.
The output directories must be new. No command downloads assets, installs packages, pushes or changes an app setting.

The default comparison uses four CPU decoder threads, 30 duration-stratified inputs over three alternating rounds,
and a separate 650-input quality run with timestamps. Every latency input is warmed once in each fresh worker.
Four threads are an explicit tuned campaign setting, not the runtime default. Unspecified threads (`threads=0`)
use the machine's performance cores, which are 16 on this M2 Ultra. These code comparisons use the tuned anchor;
measure 021 compares automatic threads with four separately, without counting the preliminary thread tuning again.
Load to ready and the first request are recorded separately. `--smoke` uses three inputs and is only a harness check;
it cannot support acceptance. A fixed anchor is rerun when it differs from the immediate baseline. Gains are direct
paired measurements, not compounded percentages. The stage tool uses three PCM-derived `audio.npy` inputs with
explicit threads; these files are benchmark audio, not tensors from the official implementation.

For a promising candidate with a completed primary run, prepare a separate six-input latency supplement:

```sh
/usr/bin/python3 -B experiments/haas-mac/short.py prepare --assets "$ARTIFACTS/assets/assets.json" --out "$ARTIFACTS/short-prefixes"
/usr/bin/python3 -B experiments/haas-mac/short.py run --primary-run "$ARTIFACTS/003/full" \
  --inputs "$ARTIFACTS/short-prefixes/inputs.json" --out "$ARTIFACTS/003/short"
```

These are two- and four-second prefixes of the three existing stage recordings, with original and derived PCM hashes
and exact sample boundaries. They are not naturally completed short utterances. The full-recording labels are not
prefix references: both CER fields remain null. The supplement compares every latency text with the baseline over
three alternating rounds and reports six-input and duration-stratum latency, separately from the primary result.
It inherits the primary binaries, models, environment and thread setting and requires the same recorded Python
version and executable, plus the identical recorded `worker.py` timer source. New primary manifests record
`sys.executable`; old manifests are refused, not backfilled.
Use `/usr/bin/python3` for the campaign's primary runs. The supplement freezes its runner files, uses the shared lock,
and never changes the primary dataset, result or verdict. Its JSON can be shown as a separate diagnostic in the trial
report; it cannot establish short-utterance CER or adoption on its own.

A separate latency repeat can check variation in a completed primary comparison without rerunning quality:

```sh
/usr/bin/python3 -B experiments/haas-mac/latency-repeat.py run --primary-run "$ARTIFACTS/005-production/full" \
  --runner "$ARTIFACTS/005-production/runner" --assets "$ARTIFACTS/assets/assets.json" \
  --out "$ARTIFACTS/005-production-latency-repeat"
```

Commit the driver before execution. It verifies the primary frozen worker and comparison arithmetic, Python,
binaries, models and PCM, then holds the shared lock for all work. It uses the original 30 recordings with all-input
warmup in each fresh worker and three AB/BA/AB rounds. It records the original block order, including its anchor,
while the repeat has only baseline and candidate. Every latency text is compared; quality, CER, stages and the
six artificial prefixes are not rerun. The new block has its own summary with per-input, stratum and round paired
ratios and cannot be pooled into the primary or prefix measurements. The existing worker timer is unchanged.

`--screen` selects the shortest, middle and longest of the existing inputs in each duration stratum: nine inputs over
three rounds, plus nine separate quality requests. It is useful for rejecting an unpromising variant before the full
run, and cannot support acceptance. `--smoke` and `--screen` are mutually exclusive. Without `--anchor`, the report
contains no cumulative gain. An explicitly supplied anchor with the same binary and environment as the baseline is
recorded as an alias of that measured side, without duplicate requests.

For a switchable prototype, pass the same build manifest on both sides and record each environment explicitly:
`--baseline-env GGML_METAL_MM_TILE_N=32 --candidate-env GGML_METAL_MM_TILE_N=64`.
Only named campaign variables are allowed. Before worker or stage launch, the runner refuses every inherited
`GGML_*` or `SPEECH_*` variable that the invocation does not explicitly record, including a value of `0` for
presence-based flags. Builds refuse all such overrides; configure flags remain explicit in the build command.
This check runs before timing and does not retrospectively establish whether an old run inherited an override.
The CPU prepacking prototype uses `SPEECH_HAAS_CPU_REPACK=0` and `=1` on its two
sides; the selective FFN dequantization prototype uses `SPEECH_HAAS_FFN_F16=0` and `=1`, and the TDT resident-state
prototype uses `SPEECH_HAAS_TDT_RESIDENT_STATE=0` and `=1`. These switches require explicit binary values.
The independent joint-graph, CPU-plan and encoder-QKV prototypes use `SPEECH_HAAS_TDT_JOINT_REUSE`,
`SPEECH_HAAS_TDT_CPU_PLAN` and `SPEECH_HAAS_ENCODER_QKV`, respectively, each explicitly set to `0` or `1`.
Build manifests retain their configure/build commands; public reports preserve their flags without source or
build directory paths. The adopted implementation must be rebuilt without experiment switches and
measured again. An exploratory result without a verified build manifest may be imported as a rejected or conditional
trial, with its original provenance and evidence hashes; it must not be relabeled as a verified build.

Literal CER sums Unicode code-point edits and reference lengths. Repository CER first applies NFKC and removes
Unicode punctuation, symbols, separators and whitespace. It does not fold case or accept alternative spellings.
Files, rather than the nonunique FLEURS numeric IDs, are the pairing keys. Exact comparisons cover text and the public
token and segment text/times returned by the worker. Raw TDT token IDs, frame indices and duration decisions are not
exposed and are not claimed to match. Text from every latency round, with timestamps disabled, is also compared and
recorded. Exact acceptance rejects a changed latency text even when the separate timestamp-enabled quality run agrees.

Write a compact `checks.json` with a `checks` array. Each entry has `name`, `status`, `required`, `scope` and a relative
`evidence` path. Status is one of `passed`, `failed`, `refused`, `skipped` or `unavailable`. Record the reviewed verdict:

```sh
python3 -B experiments/haas-mac/compare.py record --measure 001a --title "First GEMM variant" \
  --run "$ARTIFACTS/001a" --checks "$ARTIFACTS/001a/checks.json" --artifact-root "$ARTIFACTS" \
  --verdict rejected --reason "State the measured result and its limits."
python3 -B experiments/haas-mac/check.py --previous "$PREVIOUS"
```

Trial IDs such as `001a` and `001b` belong to measure `001`; a failed variant does not close the whole measure.
Never overwrite a recorded trial with a different run. Unattempted measures remain pending. Use `in_progress` for a
measure with continuing trials, `not_applicable` for an idea already implemented or outside the workload,
`conditional` for a stated tradeoff, and `blocked` when required evidence cannot be obtained. `validated_mac` requires
the full Mac run and all required Mac checks; `accepted` requires every required check, including other platforms.
Neither status treats a refused or missing check as a pass. `pending_official_reference` keeps the missing official
tensor check explicit. The runner does not infer an improvement from a verdict; reviewers compare the gain with the
round variation before adopting it.

Commit the compact JSON, rendered HTML and implementation in meaningful units. Preserve rejected implementations
on their experiment branches. The preliminary precision report has an unverified source commit and is not evidence
of a code speed-up. No report should contain raw transcripts, model weights or absolute user-specific host paths.
