# Compare automatic and explicit CPU decoding threads for Haas on Metal

This source-only experiment starts at speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`, on branch `codex/exp-haas-021`. It includes no other runtime experiment.
It has not been built, run against a model, timed or accepted. Four threads are a candidate, not an established
optimum across Macs or input lengths.

## Selection and ownership

`SPEECH_HAAS_AUTO_THREADS` is a default-off experiment. In a macOS build, an unset thread preference for the exact
`Haas-ja-v1.0` model with KDA prediction on a Metal device reads this switch when the model loads. Unset and `0` retain
the machine default. `1` selects `min(4, machine_default)`. Other values in this scope are errors. Explicit thread
counts, a CPU model, other backends and other models do not read the switch and retain their existing behavior.
The selected device's backend registry identifies Metal; a device description or the presence of a GPU is not enough.

The C API rejects `speech_load_params_set_threads(params, 0)`. Automatic selection means omitting that setter, or
omitting `--threads` at the CLI. The private load-parameter object retains zero until `speech_model_load` resolves it.
The family's private callback chooses only that unset preference, after metadata and the requested device are known
and before a backend or engine is constructed. Explicit 16 never becomes four. Passing the same resolved integer to
the backend, engine and loaded information preserves the meaning of `speech_model_info_threads` and the worker's
`ready.model.threads`.

The FastConformer recognizer stores this count for the lazy CPU decoder copy. That copy uses it both to configure the
CPU backend and to construct its persistent thread pool. No pool implementation, graph, weight, tensor type, request
option or decoding-device default changes. Metadata-only information does not call the callback and still reports
zero CPU threads through the C getter, with no loaded `threads` field in its JSON. A GPU model that never computes on
the CPU retains its existing reported zero.

CPU-only Haas keeps its machine default because its encoder also computes on the CPU; changing it would broaden this
decoder experiment. A Metal F16 model still decodes on its existing default device unless a request selects the CPU;
its information reports the preference available to that CPU decoding. The sweep explicitly selects CPU decoding for
every type, including F16.

## API and adoption boundary

This default-off prototype leaves API major/minor 4.1 and the worker protocol unchanged. It adds no exported function,
option, enum or public structure. The callback is private. The documented ordinary default remains the performance
core count, and existing callers see it unless they explicitly enable this experimental environment variable.

Adopting a different automatic default is a separate decision. The public header and user documentation currently
promise performance cores, and the repository's version rule treats a change an existing caller notices as a major
change. Production adoption must settle that contract and the API-major decision explicitly, update the relevant user
documentation, and record the lasting choice if its reasons require a decision record. Keeping a misleading reported
16 while computing with four is not an alternative. This trial makes no production-default or release decision.

## Prepared checks

`021-thread-check.py` exercises the exported C API of a candidate shared library. It loads weights and initializes
devices, but does not run inference. Run it only after the coordinator grants the compute slot:

```sh
/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock \
  python3 experiments/haas-mac/021-thread-check.py "$LIBSPEECH" "$HAAS_MODEL" --control-model "$OTHER_MODEL"
```

It checks metadata under an invalid switch, the unchanged public refusal of zero, unset/off/on, null parameters,
invalid automatic selection, explicit 1/2/4/8/16, an invalid switch ignored by explicit 16 and by the CPU model, and an
optional unrelated model. Every loaded case requires the C getter and information JSON to agree. Its positive cases
require an actual Metal device. Missing control weights, Linux/Windows behavior, lazy CPU pool execution and official
dumps are not counted as passed. CPU decoding through the real worker is a separate required run.

Changing CPU thread count can change reduction arithmetic. Before accepting a candidate, compare its output to both
the automatic baseline and the explicit-four campaign baseline. The worker exposes text and public token/segment
text and times, not raw TDT IDs, frames or duration decisions; an exact worker comparison cannot certify those hidden
values. Preserve the existing stage-check bounds and record any unavailable official inputs as unverified.

## Screen and quality plans

The sweep tool freezes `compare.py`, `worker.py` and `campaign.json` from runner commit
`a239cdb483ff90e05969bdecfb5efe39137119da`. Its worker SHA-256 is
`46483a7bde5bf3d09f0898224b71e99288f17acec33912a4b09752b0acc09c9b`.
It keeps the campaign default of explicit four threads. The worker's auto representation is `threads=0` plus a
positive `expected_threads`; it omits the CLI option and verifies `ready.model.threads` before sending any audio.
Explicit counts derive their expectation from the requested count. Each result saves requested mode/count, expected
count and resolved count in `thread_selection`, as well as the actual command and ready object.

The default screen has nine conditions, with three rounds on nine duration-stratified inputs:

| Condition | Build and switch | Requested threads | Expected on this M2 Ultra |
|---|---|---|---|
| baseline-auto | Original, 0 | Omitted | 16 |
| baseline-4 | Original, 0 | 4 | 4 |
| candidate-auto-off | Candidate, 0 | Omitted | 16 |
| candidate-auto-on | Candidate, 1 | Omitted | 4 |
| candidate-1/2/4/8/16 | Candidate, 1 | 1/2/4/8/16 | The explicit count |

The machine default is an explicit preparation input and must first be observed through an ordinary API load; a
different count causes the worker to stop before warmup. Conditions rotate their execution order each round. Every
fresh worker warms the same nine inputs. All other experiment switches remain absent; inherited `SPEECH_` or `GGML_`
overrides not recorded in the specification are refused. Builds must have verified source, the fixed ggml and matching
build flags. The coordinator also verifies that their runtime source is the base and this trial, without a combined
optimization. No timing is performed during preparation:

```sh
python3 experiments/haas-mac/021-thread-sweep.py prepare \
  --baseline "$BASE_BUILD_JSON" --candidate "$CANDIDATE_BUILD_JSON" --assets "$ASSETS_JSON" \
  --machine-default 16 --types q4_k --out "$SCREEN_DIRECTORY"
```

`run "$SCREEN_DIRECTORY"` executes that prepared plan only after a granted compute lease. It verifies the frozen
runner, driver and every job specification, and each worker verifies its binary, model and audio hashes. The shared
lock covers each complete worker run. A failed job stops the plan and is recorded as failed, without retrying it as a
success. `execution.json` records completion and result hashes, not a performance or correctness verdict.

After reviewing the screen, prepare full quality only for the selected condition, for example with
`--quality candidate-auto-on` and a new output directory. That plan runs the full corpus once for the selected
candidate, baseline-auto and baseline-4, with timestamps enabled; it does not repeat full quality for every sweep
point. A winning explicit count other than the current auto candidate does not silently change the implementation:
changing the candidate requires a new source checkpoint and measurements.

Report medians across rounds per input, then the campaign's p50/p90/p99 and RTFx. Compare each screen condition
separately with baseline-auto and baseline-4, including round spread, load time, first request and peak memory. Keep
the auto-default benefit separate from code gains already measured at explicit four threads; do not add it to 010 or
the campaign's cumulative speed-up. The stage timing executable constructs the recognizer directly and cannot prove
the C API's automatic selection. Any stage diagnosis uses the observed explicit count and is labeled accordingly.

018 changes frontend frame scheduling, and 019 changes its FFT. Neither is included or enabled here. The former caps
its workers by the same preference if later combined, so a combined build requires its own paired measurement. This
checkpoint has no model/GPU execution, full quality result, Vulkan pass, performance result or acceptance claim.

## Source checkpoint verification

`git diff --check` and Python AST parsing pass. An offline synthetic preparation fixture produces 27 screen jobs and
three selected-quality jobs with the pinned worker, expected counts and matching specification hashes. It invokes
neither the plan's `run` command nor any binary, library, model, build or GPU. These checks validate preparation only;
the C API load check and every numerical and performance check remain unexecuted.
