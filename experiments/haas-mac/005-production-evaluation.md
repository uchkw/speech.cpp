# Haas QKV production evaluation

Status: measurements complete; conditional adoption candidate. The checked outputs match, and an independent latency
repeat reproduces small paired median gains on the M2 Ultra. This switch-free implementation is retained in PR #5;
it is not merged into main or made the default baseline for other measures. No further identical repeat is required.

Q4_K's p90 increases at the same 22.62 s input in the primary and repeat, while Q8_0's long-input repeat gain is nearly
zero. The retained weight copies cost 40.5/76.5 MiB. This is not a change that makes every input faster. Keep it as a
candidate for combined evaluation after larger host-overhead improvements, including measure 023.

## Fixed conditions

Measured on an M2 Ultra on 2026-11-07: Metal encoder, CPU decoding with four threads, Python 3.9.6, runner
`b899922d5f71c5f66be33e0add13f1d310b5f74b`. Fresh baseline `1dbc333` and candidate `71c4fe9` use identical literal
compile/link flags and SDK after replacing only their directory prefixes; 74 common compile and link files match.
The candidate's runtime is `37b7107`; `71c4fe9` only narrows the new check's scope. No experiment environment variables
are set. The fixed anchor `f976ebd` is an independent reference: its implicit architecture/SDK flags differ from the
fresh pair. All builds use ggml `e1a9ac12edbb57402618fd6936598603f5ee6266` and the same pinned model files.

The primary evaluation uses 30 inputs, three rotating rounds and 650 quality inputs per type. Latency excludes model
loading, file reading and warmup. The paired reduction is computed from the median candidate/baseline ratio across
per-input three-round medians; it is not the change in aggregate p50. Load time below is the median of fresh workers with warm
filesystem caches. First-request times remain separate in the raw summary.

## Results

| Type / build | p50 / p90 / p99 latency (ms) | RTFx | Peak memory (MiB) | Load (ms) |
|---|---|---:|---:|---:|
| Q4_K baseline | 79.16 / 121.78 / 145.44 | 175.29 | 459.63 | 139.32 |
| Q4_K candidate | 77.20 / 126.04 / 142.47 | 176.60 | 497.11 | 143.19 |
| Q8_0 baseline | 80.85 / 128.67 / 149.47 | 172.66 | 752.20 | 193.25 |
| Q8_0 candidate | 78.20 / 126.93 / 148.75 | 176.13 | 827.16 | 196.98 |

Q4_K's paired reduction is 1.38% against the fresh baseline and 1.14% against the anchor; 22/30 input medians improve.
Its three rounds give 1.70%, 0.14% and 2.49%. The 20-30 s stratum gives -0.59%, and aggregate p90 increases.
Q8_0 gives 2.40% and 2.17%, with 25/30 input medians improved. Its rounds give 3.68%, -2.47% and 2.27%; all three
duration strata improve after aggregation. These results do not establish a gain in every round or input length.

The separate six-prefix supplement gives Q4_K 3.71% and Q8_0 2.75% paired reduction. Both improve all six input
medians, but Q8_0's middle round gives -0.74%. These are 2/4 s prefixes of three recordings, not naturally completed
utterances; prefix labels are unavailable and CER remains null. The permanent packed copies add 40.5 MiB and
76.5 MiB respectively. Whole-process peak changes include allocator and sampling variation.

## Correctness and limits

- Each type matches all 650 public quality outputs, including tokens and times, and all 90 latency texts against both
  references. All 18 short texts match. CER is unchanged on the labeled full recordings.
- Metal Q4_K/Q8_0 pass byte, stride and graph checks and 1,125 exact F32 tensor comparisons each at nine boundary
  lengths. The ordinary graph in this check ends at the encoder output. CPU and F16 paths pass separate graph controls.
- Original F32 weights pass the official 11-input encoder check on CPU (16 threads, minimum SNR 104.4 dB, bound 90 dB)
  and Metal (56.4 dB, bound 25 dB). F32 does not activate packing; this is distinct from the quantized paired evidence.
- Three existing recordings per type match the enabled prototype `d3796ef` bit for bit at ordinary
  `Recognizer::encode`, including `joint.enc`, and in the public worker outputs.
- The `0edd8e16` joint-encoder checker, scope headers and JSON reader are fixed separately and linked to the same seven
  production libraries. Metal F32/F16 each pass 11 official inputs at minimum SNR 77.0 dB, with the unchanged 40 dB
  bound. This isolates `joint.enc` from official `encoded.npy`; it neither activates packing nor replaces the worker check.
- Official dumps cover frontend, encoder and `joint.enc`, not full KDA/TDT decoding. Windows Vulkan and other Metal
  devices remain unverified. Stage timing is separate from worker latency; their reductions are not added.

## Retained primary evidence

Raw evidence remains under `outputs/haas-mac-campaign/005-production/`, outside Git. `checkpoint-summary.json` has
SHA-256 `721ab7b045a6e444f58e1199edd5f349690133fb1d214d41bfd0b4423bff062e`; `evidence-manifest.json` has
SHA-256 `5fa7745e5d82df8fa313c3a95386e61005c637874ce42abd1783516ddde59700`. They retain full/short manifests,
binary/library identities, official and paired checks, prototype continuity and the separately linked projection check.
`paired-analysis.json` retains each input, round and duration stratum, including regressions.

## Independent latency repeat

Driver `09cc038` repeated the same 30 latency inputs on 2026-11-07 with the primary frozen worker, Python 3.9.6,
binaries, models, PCM and four CPU decoder threads. Each fresh worker warms all 30 inputs; order is AB/BA/AB. There is
no anchor, quality, stage or prefix run. Each type matches all 90 paired texts and all 180 texts match their original
primary side and round. The repeat, primary and short results stay separate; no measurements are pooled.

| Type / build | p50 / p90 / p99 latency (ms) | RTFx |
|---|---|---:|
| Q4_K baseline | 78.31 / 122.61 / 146.63 | 174.58 |
| Q4_K candidate | 77.88 / 124.81 / 144.97 | 177.42 |
| Q8_0 baseline | 75.35 / 119.37 / 140.14 | 183.04 |
| Q8_0 candidate | 74.65 / 117.62 / 140.23 | 184.54 |

Q4_K gives 1.87% paired reduction, with 26/30 medians improved. Rounds give 2.30%, 2.01%, 1.79%; the 6-10/10-20/20-30 s
strata give 4.16%, 1.76%, 0.92%. Q8_0 gives 1.32%, with 24/30 medians improved; rounds give 1.02%, 1.32%, 0.76%, and
strata give 1.73%, 1.18%, 0.06%. Q8_0's paired median improves in every round, but its round-0 candidate p90 is
131.28 ms versus 120.81 ms. Positive paired medians do not imply improvement throughout the latency distribution.

Q4_K's p90 increases by 2.20 ms. Its four slower input medians, from retained raw results without another model run:

| Input | Audio (s) | Median difference (ms) | Round differences, candidate minus baseline (ms) |
|---|---:|---:|---|
| 5233295116612017093 | 19.74 | +0.236 | +6.249 / +0.236 / -5.769 |
| 17808260654685775088 | 22.62 | +2.995 | -0.422 / +2.995 / +1.154 |
| 16473122140572748688 | 22.86 | +0.819 | -0.743 / -3.070 / +1.741 |
| 12382486935508330615 | 26.04 | +1.262 | -0.738 / +1.522 / -0.219 |

Both sides' p90 interpolates ranks 27 and 28, the same 22.62 s and 24.72 s recordings, with weights 0.9 and 0.1.
Their changes are +2.995 ms and -4.933 ms, producing the +2.202 ms p90 change. The 22.62 s input also regressed in all
three primary rounds, with a +4.368 ms median difference. In the repeat no input is slower in all three rounds, and
7/10 long input medians improve. The tail increase is concentrated in this cohort rather than a uniform long-input
slowdown; its cause remains undetermined. Separate-side median differences need not match paired round differences.

Repeat evidence is under `outputs/haas-mac-campaign/005-production-latency-repeat/`. `run.json` has SHA-256
`e2dca7428a2529e15dc46ec1f7c42f2a3c7f2bff9895592ecedad36fcf7952d8`; `summary.json` has
`147ee6c95f558506f305f4e02a8aa442b42ea6121945c88d19a069dbcbb41fa2`. `independent-verification.json` recomputes
the metrics and checks identities and texts; `q4-tail-analysis.json` retains the four inputs and p90 contributors.
