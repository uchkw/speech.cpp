# Reject the GLU-only Metal prototype after its Mac screen

Trial 011a preserves the checked output but does not show a stable latency benefit beyond round variation.
It remains a default-OFF experiment and is not adopted or combined with other variants. The broader measure 011,
pointwise projection plus GLU, remains in progress. No full 650-input or sub-6-second follow-up was run.

## Conditions and provenance

Measured on 2026-11-07 JST: Haas-616M-ja-v1.0, Q4_K/Q8_0/F16, Apple M2 Ultra (24 CPU cores, 16 performance cores,
76 GPU cores, 192 GB), macOS 26.7.2, Release/native/Metal. The CPU decoder uses an explicit four threads; the runtime
automatic default is 16 on this machine. Python is 3.9.6, invoked through `/usr/bin/python3`. Builds use `-j4`, and
builds, GPU execution and substantial hashing are serialized with the shared campaign lock.

| Role | speech.cpp | ggml |
|---|---|---|
| Candidate runtime | `54dd7a5df9b3646b03581c9b4cd27cb3a3b55600` | `232e507d0c35312b77a57720a4c7095b3c77ff5e` |
| Original kernels, identical tests | `d8971602369b9fe4478d52e797bde48d718729cb` | `caa352ff64190598e6661a383c4768110dc47fd9` |
| Fixed campaign anchor | `f976ebd8e2fca1f15ddc2e6fc9f99e7090b0f9ea` | `e1a9ac12edbb57402618fd6936598603f5ee6266` |

Original and candidate have matching configure/build options and all 74 speech and 28 ggml `flags.make` files
after normalizing source/build paths. They explicitly specify arm64 and the Xcode macOS SDK. The fixed anchor
uses implicit architecture/SDK settings; equal performance from those settings is not established. The primary
comparison is the same candidate binary with only `GGML_METAL_GLU=0` versus `1`. The anchor has no override.

The runner is the unchanged snapshot of `a239cdb483ff90e05969bdecfb5efe39137119da`. The runtime source did not
change during evaluation. Build manifests record the built source, compile commands and executable hashes.

## Screen measurements

The screen uses nine FLEURS Japanese inputs, three per duration stratum, over three alternating rounds. The
campaign corpus spans 6.36-28.2 seconds. Every input is warmed in each fresh worker. Timing covers sending the
first PCM chunk through receiving the terminal result; loading, file reads and warmup are separate. Each input
is reduced to its median across rounds. The paired gain is one minus the median of ON/OFF latency ratios,
expressed as a percentage, not the ratio of aggregate p50 values. Positive percentages mean lower latency.

| Type | OFF p50 / p90 / p99 ms | ON p50 / p90 / p99 ms | OFF / ON RTFx | Paired gain |
|---|---:|---:|---:|---:|
| Q4_K | 78.185 / 124.993 / 146.101 | 77.664 / 124.186 / 144.055 | 175.461 / 176.511 | +0.516% |
| Q8_0 | 75.411 / 119.677 / 139.679 | 74.887 / 122.128 / 142.433 | 181.552 / 182.559 | +1.628% |
| F16 | 72.826 / 118.577 / 139.950 | 72.878 / 119.180 / 140.491 | 186.261 / 184.292 | -1.219% |

| Type | Paired gain per round | Median per-input round range OFF / ON | Gain versus fixed anchor |
|---|---:|---:|---:|
| Q4_K | +2.034% / +0.011% / +0.721% | 1.570% / 3.320% | +1.492% |
| Q8_0 | +0.065% / +0.633% / +4.159% | 7.208% / 3.700% | +0.151% |
| F16 | +0.174% / -1.219% / -1.236% | 1.047% / 3.191% | +1.236% |

The range is `(maximum - minimum) / median` across the three observations of each input, then the median across
nine inputs. It is an observed-variation description, not a confidence interval. Q4_K and Q8_0 gains are smaller
than that variation, and F16 is slower. The small sample cannot establish production tail latency or justify
adoption. The coordinator approved rejection after this screen.

| Type | OFF / ON load ms | OFF / ON first request ms | OFF / ON peak footprint MiB |
|---|---:|---:|---:|
| Q4_K | 139.152 / 138.485 | 168.388 / 166.532 | 428.829 / 431.298 |
| Q8_0 | 183.664 / 184.107 | 161.697 / 159.040 | 717.329 / 713.251 |
| F16 | 273.909 / 274.402 | 162.467 / 161.253 | 1281.189 / 1282.908 |

Peak memory is the process peak footprint from `/usr/bin/time -l`, not separate GPU VRAM. Fresh-process load and
first-request measurements do not reset filesystem or system Metal caches. Stage timings and all anchor metrics
remain in the raw summary; diagnostic graph logging and counters do not contribute to these measurements.

## Output and correctness

For each type, all nine timestamp-enabled results match across OFF, ON and anchor in public text, token text/times
and segments. Text from every one of the 27 latency requests also matches, with no repeated-text changes. The
worker does not expose private TDT token IDs, frame indices or duration decisions.

Screen CER is identical across the three sides: Q4_K literal 40/532 (7.5188%) and repository 22/493 (4.4625%);
Q8_0 and F16 literal 39/532 (7.3308%) and repository 21/493 (4.2596%). These are nine-input screen results, not
650-input corpus estimates.

- Direct Metal: original, candidate OFF and candidate ON each pass 76 comparisons and two emulated poison guards.
  The 76 contain 38 eligible cases and 38 controls, including 12 distinct-wrapper controls. ON fuses exactly 38;
  original/OFF fuse zero. Both independent original-vs-candidate output files compare byte-for-byte.
- Metal versus CPU reference: all 64 `CONFORMER_GLU` cases pass on each of the three configurations. The extra
  mapped-buffer controls are Metal-only. Related operations pass 158 supported cases per configuration; the same
  111 unsupported cases are excluded, not counted as passes.
- Normal encoder then joint encoder projection: Q4_K/Q8_0/F16 and seven output lengths (1, 31, 32, 33, 171, 172,
  375) produce exactly 24 normal ON fusions, zero normal original/OFF fusions and zero retained-intermediate
  fusions. All normal/retained projected outputs agree exactly. The six external original/OFF/ON file comparisons
  also match all retained intermediates and normal projected outputs. This is 126 graph executions, separate from
  timing. Counter initialization preserves the normal two-command-buffer boundary.
- Official frontend: 11 F32 CPU cases pass on candidate ON, minimum SNR 103.3 dB at the existing 90 dB bound.
- Official encoder: original and candidate ON F32 CPU each pass 11 cases, minimum 104.4 dB at the existing 90 dB
  bound. Original/OFF/ON F32 Metal and F16 Metal each pass 11 cases, minimum 56.4 dB at the existing 25 dB bound.
  The official stage checks total 99 case executions. They retain layer outputs and do not establish normal
  fusion counts; the independent normal helper does.

The first `GGML_METAL_GRAPH_DEBUG` capture exited successfully but captured no node trace: the default speech
log sink filters Debug messages. That failed capture remains under `graph-debug/`. A separate diagnostic launcher
sets a log sink and calls the unchanged stage source, with source/launcher/binary hashes retained. The corrected
`graph-debug-v2/` capture has 1964 OFF versus 1916 ON node lines and 0 versus 48 `UNARY+MUL` fusions across two
normal encoder executions (warmup plus one loop). It uses no counters, `GGML_METAL_FUSION_DEBUG`, retained
intermediates or runtime changes. Its binary, flags and times are excluded from the worker screen.

## Evidence and limits

Raw evidence is retained under `outputs/haas-mac-campaign/011a/`: `evaluation.json`, `checks.json`,
`check-summary.json`, `screen-analysis.json`, `screen/run.json`, `screen/summary.json`, `source-checkpoints.json`,
`matched-build-flags.json`, per-run logs and results, the direct/normal binary comparisons, and build manifests.
`evidence-manifest.json` freezes the raw files and retained executable/library hashes. Large artifacts remain ignored.

Official frontend/encoder/joint.enc dumps exist, but this trial runs the existing frontend and encoder official
checks; normal joint projection is checked by strict differential comparison. Official KDA/prediction and full
TDT validation, Windows Vulkan execution, the full corpus and sub-6-second follow-up are not claimed. The
intermediate pre-guard kernel was not executed to reproduce its missing same-buffer guard. No check threshold
was loosened. No runtime change is merged, no production default changes, and no cumulative gain is claimed.
