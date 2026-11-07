# Evaluate prepared depthwise weights

The current prototype is **rejected because ordinary Metal output fails strict equivalence**. Its prepared weights
are byte-exact and the CPU check passes, but Q4_K has eight nonmatching final encoder-to-joint comparisons on Metal.
No latency, load-time, process-memory gain, CER or public ASR output comparison was measured. This is not a measured
0% speed result. The experiment remains default OFF, with no production adoption or campaign baseline change.

## Fixed conditions

| Item | Recorded identity |
| --- | --- |
| Candidate runtime and check | `dd0a27e412842bde002dc76cb02aaea82c94799f` |
| Clean original control build | `114c514430c8aa1455c077819f8e72e674a48b75` |
| ggml in both builds | `e1a9ac12edbb57402618fd6936598603f5ee6266` |
| Frozen campaign runner | `a7e227c2b9067ee0b73a4582acf2e453cff5f179` |
| Hardware and executed backends | Apple M2 Ultra, CPU and `MTL0` |

Both clean builds used Release/native/Metal, four build jobs and the shared campaign lock. All 74 common
`flags.make` and 74 common `link.txt` files match after source/build path normalization, including their arm64 and
SDK settings. The existing runtime, check and ggml sources remained unchanged throughout evaluation.
The frozen runner's build command ran under `/usr/bin/python3` 3.9.6; its performance command never ran.
The original control was built but not used for a cross-binary numerical comparison or timing.

All executed differential checks use four threads, the same Q4_K GGUF and the same candidate binary's internal
OFF/ON encoder selection. The 26 depthwise weights are F32 even in this Q4_K file. Other model weights retain their
actual quantization, and F32 intermediate states do not make this an official F32 reference check. The three real
audio inputs are the existing 7.98, 14.52 and 24.72 second stage recordings.

The four existing model files, three stage audio arrays and 66 official reference arrays matched their recorded
hashes before checks. Verifying the available original F32 model and official corpus is not an executed official
reference comparison. No new model or audio data was created.

## Observed checks

| Run | Expectations | Failed | Final joint output |
| --- | ---: | ---: | --- |
| Ordinary CPU Q4_K | 426 | 0 | All ten comparisons exact |
| Ordinary Metal Q4_K | 426 | 8 | Five synthetic and three real-audio OFF/ON comparisons differ |
| Metal Q4_K with fusion disabled, diagnostic only | 426 | 0 | All ten comparisons exact |
| Log-only Metal Q4_K, synthetic inputs only | 339 | 5 | Reproduces all five ordinary synthetic differences |

These counts include contract expectations and finite-value checks; they are not counts of independent audio
samples. Each normal run has eight OFF/ON joint-output comparisons plus two self-consistency comparisons. The
synthetic sequence starts at 129 feature frames, then uses 1361, 1369, 3000 and 129 again, giving encoder lengths
17, 171, 172, 375 and 17. The three real inputs give 100, 182 and 309 encoder frames.

Both CPU and ordinary Metal pass the following controls despite Metal's final-output failure: all 26 prepared
weight payloads equal their actual ordinary CONT products; original model bytes remain unchanged; final kernel
shapes and byte strides agree; total CONT nodes fall from 75 to 49; and prepared addresses and allocation remain
stable across shapes. The original encoder still works after the prepared owner is destroyed. The measured prepared
payload and retained backend allocation are both 903,168 bytes, exactly 882 KiB. Preparation is eager during encoder
construction, but its startup cost and whole-process peak memory were not measured.

The ordinary graphs mark only the final `joint.enc` output. Their separate weight-retention diagnostics match each
side's normal output exactly on the tested short input. The failed OFF/ON comparison therefore remains a failure
of the normal graph, not a corrected instrumentation-only failure. The prepared short revisit and original output
after prepared-owner destruction also remain exact.

| Ordinary Metal input | Maximum absolute difference | SNR, dB |
| --- | ---: | ---: |
| 129 feature frames, first and return | 0.00489997864 | 67.8287624 |
| 1361 feature frames | 0.0240705013 | 60.0137298 |
| 1369 feature frames | 0.0260584354 | 60.0794250 |
| 3000 feature frames | 0.0456334352 | 57.6434284 |
| 7.98 second audio | 0.0160903931 | 65.5662015 |
| 14.52 second audio | 0.0193753242 | 66.7786338 |
| 24.72 second audio | 0.0350699425 | 66.2605462 |

## What the bounded diagnostics establish

One same-binary run with `GGML_METAL_FUSION_DISABLE=1` makes every Q4_K comparison exact. Its environment,
command, exit code and log are separate from the original failure. It is not a production change or performance
result and does not change the strict gate.

A reviewed log-only launcher then calls the existing check main unchanged. It installs an all-level stderr sink
with callback boundaries, byte lengths, sequence and thread identifiers under a mutex. It changes no tensor output,
graph, allocator, threshold or runtime library. One Q4_K GPU4 process, with no audio arguments and only
`GGML_METAL_GRAPH_DEBUG=1`, reproduces the original five synthetic differences. Fusion debug, capture and fusion
disable are unset. Its compile/link flags, seven retained static libraries, source copies and executable are recorded.
The launcher alone was generated by zcode GLM-5.3/max from three public source files and synthetic instructions;
no model, audio or raw execution result was sent to that coding API. Independent Sol review found no blocker.

The first 129-frame ordinary/prepared pair, excluding position-cache construction, has 120 versus 119 observed
three-operation NORM fusions. The first difference is the tenth LayerNorm, corresponding by encoder build order
to `blk.1.out_norm`, the second block's final norm. The ordinary graph fuses its NORM/MUL/ADD at `node_178`, local
index 2. In the prepared graph, `node_168` NORM is local index 111, followed in the next local partition by MUL at
0 and ADD at 1. Excluding removed weight CONT nodes, the operation/fuse sequence matches until that point.
The raw callback records preserve this observation without joining lines from different threads.

This identifies a fusion change across a partition boundary, not the first internal tensor with a numerical
difference. Fused arithmetic can remove intermediate F32 stores and change rounding or contraction, but the
specific FMA/reduction behavior and amplification through later blocks were not measured. An alias or concurrency
defect has not been excluded. The source configures `n_cb=1`; that is a source-based inference, not a runtime
measurement derived from these local node indices. No capture or further tensor observation was performed.

## Decision and retained evidence

The coordinator closed this checkpoint as rejected without relaxing thresholds or adding another run. Q8_0/F16
inference, original/OFF/ON continuity, official F32 CPU/Metal checks, fixed-anchor comparisons, performance screens,
full corpus, short prefixes, CER, public text/tokens/times and cancellation checks were not executed. Windows/Vulkan
remains unverified. These are local checks, with no CI success claimed.

`outputs/haas-mac-campaign/014/` retains the original failures, both bounded diagnostics, public launcher source,
build and input identities, commands and frozen runner. `evaluation.json`, `checks.json` and `check-summary.json`
give the final checks-only dataset; `evidence-manifest.json` and `raw-freeze.sha256` identify retained evidence.
The earlier pending diagnostic checkpoints remain unchanged. Large artifacts and logs are not committed.

The rejection concerns this prototype's unmet strict contract. It does not permanently reject weight preparation
or establish that every different fusion layout is defective. Any later design that controls fusion or combines
another optimization requires a separate experiment and its own validation; no such combination is included here.
