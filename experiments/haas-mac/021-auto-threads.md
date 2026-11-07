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
different count causes the worker to stop before warmup. Round zero uses the table's order, round one reverses the
entire order, and round two rotates the original order by half its length. The plan saves each model type's actual
condition order for every round. This reverses the relative order of explicit counts and spreads their positions;
three rounds do not balance all nine conditions over every position, so the screen cannot exclude all order effects.
Every fresh worker warms the same nine inputs. All other experiment switches remain absent; inherited `SPEECH_` or `GGML_`
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
`021-thread-sweep-check.py` checks the deterministic order without a model, including the reversed positions of
explicit one and sixteen, every condition exactly once per round, and an optional prepared plan's saved job order.
Offline preparation from `581233f` reproduces the fixed ascending relative order in its second round. The revised
preparation passes this order check against its saved plan and job specifications; no worker is executed in either case.

## Completed evaluation

The evaluation ran on 2026-11-07 on Apple M2 Ultra on macOS 26.7.2 with MTL0. 114c514430c8aa1455c077819f8e72e674a48b75 is the original source commit and 37f41cebe928336af11fbd7cac2a3ccfbb067225 is the candidate source commit. Both builds use the ggml commit e1a9ac12edbb57402618fd6936598603f5ee6266, and the frozen sweep worker comes from a239cdb483ff90e05969bdecfb5efe39137119da. 74 is the count of generated compiler-flag files and 74 is the count of generated link files whose normalized contents match between the builds. Normalization of the compiler flag and link files replaces only the absolute source and build directories. The dataset is Google's public FLEURS Japanese test split.

### Experiment

All latency and timestamp-quality workers used the local Metal encoder with decoding requested on the CPU, and readiness and real CPU decoding were observed separately. The screen used 9 fixed public inputs across 9 conditions covering the Q4_K, Q8_0 and F16 precisions. The warmups are equal across conditions and have the same size as the fixed screen set. When the CLI thread option is omitted the harness records the request as automatic and checks the declared positive expected count against the actual ready.model.threads. The numeric zero sentinel for an omitted request is not equal to its resolved positive count. In each condition the worker rejects a mismatched effective count before warmups, and explicit settings keep their requested count.

Each condition ran for 3 latency rounds under three round orders, which still leave ordering noise. The per-file median latency is taken across rounds, and p50, p90 and p99 are computed with the linear method across files. RTFx for a condition is the total audio seconds of the input set divided by the sum of its per-file median latencies. Paired comparisons take, for each input, the median across its latency rounds separately in each condition, and divide each candidate per-file median by the reference per-file median for that input. The reported paired ratio $r$ is the median of those per-file ratios across the fixed screen inputs. A round's ratio uses candidate and reference observations from that round only, then takes the median across the same inputs. The percentage reduction is $100(1-r)$, positive for lower latency, and is interpreted against the same-count control's round-to-round variation, whose drift is treated as noise with no code improvement credited to it. This is distinct from taking a ratio of the aggregate p50s.

The thirty-input latency supplement and the short-input supplement are separate items, and neither was run. No full latency run is reported and no non-Apple hardware claim is made.

### Performance

The first table reports the pivotal conditions with their paired comparisons.

| Precision | Original auto16 p50 / p90 (ms) | Original explicit4 p50 / p90 (ms) | Candidate auto4 p50 / p90 (ms) | Auto4 vs original auto16, paired % (three rounds) | Auto4 vs original explicit4, paired % | Candidate explicit4 vs original explicit4, paired % (three rounds) |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_K | 88.65 / 140.08 | 79.69 / 131.82 | 81.92 / 129.71 | +6.27 (+5.21 / +7.76 / +6.33) | -0.71 | +0.05 (-0.88 / -6.13 / +3.02) |
| Q8_0 | 84.90 / 135.43 | 79.44 / 125.30 | 77.11 / 123.86 | +7.49 (+5.66 / +8.29 / +7.67) | +1.25 | +1.20 (+3.34 / +1.49 / -0.32) |
| F16 | 83.52 / 133.12 | 79.48 / 127.62 | 76.36 / 123.98 | +6.63 (+6.89 / +6.50 / +6.63) | +1.75 | +1.05 (+2.10 / +5.69 / -1.16) |

With the selector switched off, the original and the prototype both retain the omitted machine default of 16 threads. Turning the prototype selector on changes the omitted count to 4 threads. The two selectors do not create two independent application gains: the current application already chooses 4 threads explicitly, so zero incremental application or cumulative runtime gain is credited. The trial is a conditional configuration candidate. The prototype stays default-off and is not adopted in production. The public API promises performance cores by default when identifiable and otherwise physical cores, and changing the production default still needs a compatibility, major-version and documentation decision.

The next table lists every tested explicit count.

| Precision | Explicit threads | p50 (ms) | p90 (ms) | p99 (ms) | RTFx | Peak footprint (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_K | 1 | 84.05 | 134.45 | 154.86 | 163.73 | 430.0 |
| Q4_K | 2 | 81.97 | 131.50 | 151.81 | 168.00 | 433.4 |
| Q4_K | 4 | 79.65 | 129.67 | 149.86 | 168.92 | 434.5 |
| Q4_K | 8 | 83.09 | 133.14 | 153.95 | 165.87 | 438.3 |
| Q4_K | 16 | 88.50 | 140.21 | 161.25 | 156.99 | 438.9 |
| Q8_0 | 1 | 80.11 | 128.97 | 149.92 | 172.10 | 719.3 |
| Q8_0 | 2 | 77.12 | 124.33 | 147.02 | 176.16 | 723.0 |
| Q8_0 | 4 | 75.66 | 124.26 | 144.62 | 178.28 | 729.8 |
| Q8_0 | 8 | 78.94 | 126.94 | 146.22 | 175.04 | 723.7 |
| Q8_0 | 16 | 83.38 | 134.27 | 153.50 | 165.32 | 725.2 |
| F16 | 1 | 81.79 | 133.82 | 154.60 | 167.09 | 1283.9 |
| F16 | 2 | 79.29 | 126.55 | 146.72 | 173.82 | 1276.8 |
| F16 | 4 | 76.17 | 127.43 | 148.01 | 176.83 | 1286.5 |
| F16 | 8 | 78.02 | 128.55 | 148.01 | 173.23 | 1291.2 |
| F16 | 16 | 83.74 | 135.11 | 154.76 | 165.09 | 1293.2 |

The Pareto sets below are observed outcomes for this hardware and input set. The performance frontier balances four objectives, with smaller p50, p90 and p99 and larger RTFx, and the optional fifth objective is process memory: the maximum macOS peak process footprint over the three latency workers and one timestamp-quality worker in that condition. Among the five explicit counts, the selected count of 4 threads has the lowest p50 and the highest RTFx in all three precision tables, while F16's lower explicit count trades a slower p50 for slightly lower p90 and p99. No universal optimum is claimed for these frontiers.

| Precision | Performance frontier, explicit threads | Performance and peak-memory frontier, explicit threads |
| --- | --- | --- |
| Q4_K | 4 | 1, 2, 4 |
| Q8_0 | 4 | 1, 2, 4, 8 |
| F16 | 2, 4 | 2, 4 |

### Quality and adoption

The API, worker, options and readiness gate passed 14 public contract commands. C API load-selector checks on the Q4_K and F16 weight types covered metadata-only queries, the public zero setter rejection, invalid selector handling, ordinary and null parameters, explicit counts, and CPU versus Metal. Active cancellation and next-request recovery were observed in the API and worker with the thread setting omitted in both selector-off and selector-on modes, and both are automatic-CLI cases for the worker. The keyword guards were checked separately.

The full comparison covers 650 items on Q4_K only, across the original omitted, original explicit and prototype omitted conditions. Exact equality covers the public transcript text, every token's text/start/end and every segment's text/start/end across the three conditions, with 28,405 tokens and 744 segments per run. The literal and normalized character error rates 7.46679% (2,608 edits / 34,928 reference characters) and 4.65572% (1,520 / 32,648) are separate Q4_K full-corpus metrics scored against the references. Each error rate is identical across the three conditions. A character error rate cannot establish exact equality on its own, and equality was checked by comparing the actual public values. Q8_0 and F16 used the nine-input screen checks only, and hidden TDT IDs, frames, durations and logits remain unverified.

| Precision and sample | Inputs per condition | Conditions | Changed text / token / segment records vs each baseline | Literal CER | Normalized CER |
| --- | --- | --- | --- | --- | --- |
| Q4_K screen | 9 | 9 | 0 / 0 / 0 | 7.5188% | 4.4625% |
| Q8_0 screen | 9 | 9 | 0 / 0 / 0 | 7.3308% | 4.2596% |
| F16 screen | 9 | 9 | 0 / 0 / 0 | 7.3308% | 4.2596% |
| Q4_K full | 650 | 3 | 0 / 0 / 0 | 7.46679% | 4.65572% |

The configuration change remains a conditional candidate and has not been promoted to a shipping default.

The raw results behind this section are reproduced from the local evidence directory outputs/haas-mac-campaign/021: `outputs/haas-mac-campaign/021/evaluation.json` and `outputs/haas-mac-campaign/021/thread-sweep.tsv` for the aggregate evaluation and the thread sweep, `outputs/haas-mac-campaign/021/q4-screen-evaluation.json`, `outputs/haas-mac-campaign/021/q8_0-screen-evaluation.json` and `outputs/haas-mac-campaign/021/f16-screen-evaluation.json` for the three per-precision screen evaluations, `outputs/haas-mac-campaign/021/full-q4-auto-on/execution.json` for the full Q4_K automatic-selector run, and `outputs/haas-mac-campaign/021/raw-freeze-scope.json` with `outputs/haas-mac-campaign/021/raw-freeze.sha256` for the raw freeze scope and its checksum.
