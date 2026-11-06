# Evaluate the larger Haas position projection cache

Q4_K is a **conditional candidate for longer inputs** on the measured M2 Ultra. Its newly cached inputs improve,
but the six short prefixes do not, first eligible requests take longer, and the cache retains **39.9375 MiB more**.
Q8_0's full-run long-input benefit is weak and varies by round. F16 remains a screen-only candidate for longer inputs.
No type is adopted into production, no default or campaign baseline changes, and no further measurement of this
prototype is requested. An implementation without the switch requires its own evaluation.

## Fixed conditions

The measured runtime is `b47696771ff816224701e8942a81aeeda64a4fd4`, based on
`114c514430c8aa1455c077819f8e72e674a48b75`, with ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`.
The same clean binary compares `SPEECH_HAAS_POSITION_CACHE=0` and `1`; no other runtime experiment is included.
Release/native/Metal builds use four compile jobs and the shared campaign lock. `/usr/bin/python3` resolves to
Python 3.9.6. The worker explicitly uses four CPU decoder threads for every type, including F16.
Runner `b5799744ef7166a4c93dd775d8c7f8a05321379c` is frozen independently; its worker SHA-256 is
`fa7f31c2a69bfb42abe8741f07b56c1103f2b5a60e93d41c544ec7e3d88b8b0a`.
The fixed anchor uses implicit architecture/SDK flags, unlike this fresh build's explicit flags. It is an additional
reference, not the primary toolchain-matched comparison. Model, PCM, binary and runner identities are retained.

The unchanged timer starts before sending the first PCM chunk and ends at the result, excluding file reading,
loading and warmup. Every input is warmed per fresh worker. Each file's median over three rounds is computed first;
p50/p90/p99 then use those medians. Positive gains mean lower latency, using the median paired ratio.
The 30 latency inputs range from 6.36 to 28.20 seconds; their small-sample p99 is not a production-tail estimate.

## Correctness

Q4_K, Q8_0 and F16 each pass both differential modes on CPU and Metal with four CPU threads. Their actual position
weights retain their GGUF types; F32 intermediate tensors do not make these official F32 checks.

- Each type/backend has 430 strict finite, bit-exact position comparisons across
  `T = 385, 1, 170, 171, 172, 375, 383, 384, 385, 171, 1, 384, 1`.
  Selection, lazy construction, weights, central views, parent reuse, graph position products and actual buffer sizes
  pass. The 72-versus-32 gate remains strict at every length.
- Each type/backend has eight exact F32 comparisons through the ordinary encoder-plus-`joint.enc` graph with only
  its final output marked. Synthetic features cover T=385 and T=375; three real recordings and a return to the first
  cover normal recognition. Greedy IDs, frames, TDT durations, graph counts and text also match.
- The 624 cache-versus-zero comparisons per position check remain separate diagnostics. CPU results are exact.
  Metal has 144 differences per type, only at T=1, for both old and new caches. Minimum SNR is 88.93/92.35/150.39 dB
  for Q4_K/Q8_0/F16. These are not new 72-versus-32 differences or passes against zero; no bound is relaxed.
- Original official F32 weights and converter-locked encoder dumps pass 11 cases with each switch setting.
  CPU uses 16 threads: minimum 104.4 dB, unchanged bound 90 dB. Metal: minimum 56.4 dB, unchanged bound 25 dB.
  This covers the encoder, not complete official KDA/TDT decoding.
- Each screen has 9 exact public outputs and 27 exact latency texts per type against both references.
  Each full type has 650 exact texts/public token and segment arrays, plus 90 exact latency texts.
  Disabled-switch output also matches the anchor. CER is unchanged: Q4_K literal 7.4668%, repository 4.6557%;
  Q8_0 literal 7.5097%, repository 4.5761%.

Windows/Vulkan and complete official KDA/TDT stage references remain unverified. Performance is not generalized
to other Macs or CPU inference.

## Screen and independent repeat

| Type | Original gain | Original round gains | Repeat gain | Repeat round gains |
|---|---:|---|---:|---|
| Q4_K | 1.55% | 0.98%, 3.36%, -4.56% | 2.60% | 1.60%, 2.60%, 3.33% |
| Q8_0 | 1.79% | 2.02%, -13.55%, 2.41% | 3.74% | 4.61%, 1.45%, 3.89% |
| F16 | 2.99% | -6.24%, 2.10%, 3.09% | 0.69% | -0.27%, 2.41%, -0.59% |

The original Q8_0/F16 negative rounds slow all nine inputs, including already cached inputs; Q4_K's slows seven.
Q8_0's affected first request is 187.60 ms versus 161.64/163.98 ms in other rounds. Independent stage processes also
show unchanged-decoder variation, but are not aligned to worker rounds and do not establish the cause.
The four newly cached inputs improve by 3.98/3.21/3.49% originally and 3.26/4.06/3.16% in the repeat for
Q4_K/Q8_0/F16. Q4_K/Q8_0 repeat p90 and p99 also improve. F16's five already cached inputs regress by 0.89%
in the repeat, so F16 does not proceed to full or short measurement. Original and repeat results are not pooled.

## Full comparison

| Type | Paired gain | Anchor gain | p50, ms | p90, ms | p99, ms | RTFx |
|---|---:|---:|---|---|---|---|
| Q4_K | 3.39% | 1.82% | 86.56 to 82.20 | 133.78 to 126.06 | 163.04 to 148.72 | 162.21 to 169.87 |
| Q8_0 | 2.38% | 1.19% | 82.65 to 79.29 | 128.33 to 126.63 | 146.89 to 149.26 | 171.34 to 174.39 |

| Type | Round gains | 6–10 s | 10–20 s | 20–30 s | Already cached, 16 inputs | Newly cached, 14 inputs |
|---|---|---:|---:|---:|---:|---:|
| Q4_K | 3.30%, 2.73%, 2.24% | 2.17% | 3.60% | 5.15% | 1.81%, 11/16 improve | 6.58%, 14/14 improve |
| Q8_0 | 1.81%, -0.87%, 3.92% | 2.93% | 3.10% | -0.61% | 2.84%, 16/16 improve | 0.61%, 8/14 improve |

Q4_K's five slower medians are already cached inputs. Q8_0's six slower medians are newly cached 20.04–28.20 s
inputs, with increases of 0.26–3.24 ms. Its newly cached group gains are -1.47%, -0.87% and +7.15% by round.
Several inputs reverse direction across rounds, and p99 rises 2.37 ms; this is not one isolated tail.
Against the anchor its long stratum improves 3.45%, but the primary comparison does not reliably reproduce that gain.
Per-input absolute differences and all round directions are retained in `full-analysis.json`.

Separate stage medians support removal of position work: for 14.52 s, encoder build is 1.42 to 0.30 ms in Q4_K
and 1.38 to 0.28 ms in Q8_0; for 24.72 s it is 2.16 to 0.32 and 2.09 to 0.26 ms. These are three repetitions
per input in separate processes. Stage gains are not added across inputs or to worker gains; unchanged decoder
timings also vary.

## Short inputs and startup cost

Six 2/4-second prefixes from three existing recordings are measured over three alternating rounds. Both types match
all 18 texts. These truncated, related samples have no short-utterance references; CER is unavailable.

| Type | Paired warm gain | 2 s gain | 4 s gain | First request, ms | Warm p50, ms |
|---|---:|---:|---:|---|---|
| Q4_K | -1.19% | -2.09% | -0.28% | 58.38 to 62.86 | 33.95 to 34.21 |
| Q8_0 | -0.15% | -0.23% | -0.14% | 59.37 to 63.61 | 31.30 to 31.22 |

All prefixes fit the old cache; no short-input benefit is established. The first sandboxed launch fails to initialize
a Metal command queue before `ready`, with no inference. It remains in `short/`; the authorized unsandboxed run is
separately retained in `short-retry/` with the same binary, worker and environment.

One separate bounded pair per type measures no-warmup `ready`, the same first eligible 7.98 s input, its warm
repetition, and standard one-second engine warmup to `ready`:

| Type | No-warmup ready, ms | First eligible, ms | Warm same input, ms | Standard warmup ready, ms |
|---|---|---|---|---|
| Q4_K | 606.44 to 138.58 | 72.46 to 77.21 | 55.48 to 53.39 | 214.48 to 209.73 |
| Q8_0 | 184.15 to 185.45 | 71.68 to 76.79 | 50.49 to 50.89 | 251.18 to 254.33 |
| F16 | 274.63 to 271.05 | 68.97 to 77.28 | 50.00 to 50.36 | 326.97 to 330.31 |

The first Q4 load is an outlier, not evidence that a still-unallocated cache speeds loading. These single pairs do not
reset filesystem or Metal caches. First eligible time also includes pipeline setup and lazy CPU decoder copies; its
4.75/5.11/8.31 ms increase does not isolate cache construction. In ordinary full warmup, candidate cache creation
occurs on the longest input while the baseline creates it later; `first_request` is not a matched construction timer.

Actual retained buffers grow from 33,521,664 to 75,399,168 bytes, adding 39.9375 MiB, even for short requests.
Full process peak footprint is 455.38 to 454.02 MiB for Q4_K and 749.95 to 741.19 MiB for Q8_0. This separate
peak measurement does not erase the additional retained allocation. Full load medians are 155.88 to 146.12 ms
and 186.91 to 193.48 ms; ordinary first requests are 169.20 to 182.89 ms and 173.52 to 175.59 ms. These do not
establish a cold-start or memory-saving benefit.

## Evidence

Ignored artifacts are under `outputs/haas-mac-campaign/010/` in the main workspace. The freeze retains 613 original
files (30,937,986 bytes) and verifies 24 external binary/library/build-flag files against their recorded hashes.
Runtime and checkout are unchanged at that freeze. The checkpoint and manifest add metadata without replacing raw
outputs or the runner's conditional status. The report owner imports a conditional, unadopted result.

| Artifact | SHA-256 |
|---|---|
| `raw-freeze.json` | `bb87ba48569712fc12117fb0886033aa0ca09a983ac8dde85010815771721628` |
| `checkpoint-summary.json` | `3b47a82cbe865d8d595e6023807c1fa8a4eb0bf8c1203fba21c2fe56036ab7a5` |
| `evidence-manifest.json` | `aff3b279c2195d86f25fd25dfc743a70ae1622aea2d58d079448fa830491090b` |

The checkpoint contains separate screen/repeat/full/short aggregates, per-input and per-round differences, startup
measurements, cache bytes, correctness scope, build identity and the coordinator's decision. Large artifacts,
models, audio and dumps are not committed.
