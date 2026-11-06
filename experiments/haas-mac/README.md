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
Load to ready and the first request are recorded separately. `--smoke` uses three inputs and is only a harness check;
it cannot support acceptance. A fixed anchor is rerun when it differs from the immediate baseline. Gains are direct
paired measurements, not compounded percentages. The stage tool uses three PCM-derived `audio.npy` inputs with
explicit threads; these files are benchmark audio, not tensors from the official implementation.

For a switchable prototype, pass the same build manifest on both sides and record each environment explicitly:
`--baseline-env GGML_METAL_MM_TILE_N=32 --candidate-env GGML_METAL_MM_TILE_N=64`.
Only named campaign variables are allowed. The source-verified runner refuses an inherited experiment variable
that the invocation does not record. The adopted implementation must be rebuilt without experiment switches and
measured again. An exploratory result without a verified build manifest may be imported as a rejected or conditional
trial, with its original provenance and evidence hashes; it must not be relabeled as a verified build.

Literal CER sums Unicode code-point edits and reference lengths. Repository CER first applies NFKC and removes
Unicode punctuation, symbols, separators and whitespace. It does not fold case or accept alternative spellings.
Files, rather than the nonunique FLEURS numeric IDs, are the pairing keys. Exact comparisons cover text and the public
token and segment text/times returned by the worker. Raw TDT token IDs, frame indices and duration decisions are not
exposed and are not claimed to match.

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
