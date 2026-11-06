# Reuse the TDT blank-joint graph

This independent prototype starts at campaign commit `114c514`, with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It contains neither resident KDA
state nor CPU execution-plan retention. `SPEECH_HAAS_TDT_JOINT_REUSE=0` selects
the existing path; `1` selects the experiment. Other values are errors.

The existing prediction-step graph and the blank-joint allocator are already
reused. This trial also keeps the blank joint's graph and tensor allocation
within one TDT request. The first blank evaluation constructs `JointStep`; later
ones set both frame and prediction inputs and use `compute_again`. Prediction
state reads/uploads, host argmax, duration advancement, frame copies and output
readback remain in the same decoding sequence. The graph is destroyed when
the request returns, including cancellation. A new request starts empty.

`fastconformer-joint-reuse-check` compares all logits of 128 fresh/reused joint
evaluations while interleaving two requests and changing prediction histories.
It also compares complete TDT token IDs, frames, durations, graph counts and
progress/cancellation points. The deterministic F32 fixture covers consecutive
blank chains after different predictions, zero duration, the symbol limit,
end overshoot, boosting and a request after cancellation. The fixture is copied
from the independent state experiment; it is not an official model or dump.

After a locked Release/native/Metal build with four jobs, run the check with a
model path or `--synthetic OUTPUT_DIR`, then `cpu` or `gpu`, and a thread count.
Use Q4_K, Q8_0 and native F16 with four CPU threads, plus the synthetic CPU F32
and Metal checks. These comparisons do not replace unavailable official Haas
F32 stage references or Windows Vulkan validation.

Measure with the common campaign runner, a fixed anchor and explicit `0`/`1`
environments. Begin with nine inputs and three alternating rounds; expand only
if the request improvement is clear. Preserve load, first request and peak
memory separately. All types use the modified decoder when enabled, including
native F16 under the campaign's explicit CPU decode option. F16 is therefore
not an unchanged-path control for this measure.

## Evaluation on November 7, 2026

The prototype is rejected at the screen stage. The M2 Ultra worker comparison
uses Python 3.9.6, four CPU decoder threads and nine inputs in three alternating
rounds against the fixed campaign anchor. Median paired request reductions are
1.57% for Q4_K, 0.18% for Q8_0 and 4.08% for F16. All 81 latency texts and 27
timestamped public outputs match exactly. Q4_K request ratios by round are
0.98395, 0.99818 and 0.92974; a stable reduction is not established.

The supplementary stage comparison uses three inputs in three alternating
fresh-process rounds, with three warm observations per input. Q4_K decoder
totals have candidate/baseline ratios 0.98337, 0.95680 and 1.00416; core ratios
are 1.00396, 0.99572 and 1.00639. Its 7.98-second input's decoder improves by
8-13% in each round, but this local result does not generalize to all inputs or
request latency. Q8_0 decoder ratios are 0.89479, 1.02086 and 1.13170; F16
ratios are 0.85047, 1.18136 and 0.70671. These sign changes do not establish a
consistent gain. All 81 paired stage text/token-count/graph-count observations
match. No 30-input latency or 650-input quality extension is performed.

The measured clean source is `47a3699d54e6776f88880fe4281e2afa4fd561a0`.
All nine differential conditions pass: synthetic F32 on CPU with one and four
threads and on Metal with four threads, and Q4_K/Q8_0/F16 on CPU and Metal with
four threads. Every condition compares 128 joint logits bit for bit. Each
synthetic case passes 307 assertions; each model/device case passes 284. The
controlled cancellation occurs after the reused graph has run, and the next
request starts clean. The synthetic prediction and logits remain sensitive to
different histories with the same final label and frame.

The build, check and screen manifests, logs, load/first-request/peak-memory
measurements and source snapshots are retained under external `004a/` and
`builds/004a-joint-reuse-reviewed/` campaign artifacts. The earlier build at
`0be4200` is not used for inference measurements. Official Haas stage and
Windows Vulkan validation remain unavailable.

`stage-pairs.py` is the exact supplementary stage driver used for this trial.
It takes `--run` with the screen's `run.json`, `--assets` with the matching
`assets.json`, and `--out` with a new directory. Run it with `/usr/bin/python3`;
it checks Python, binary, model and audio identities and holds the shared lock
separately for each process. This evaluation changes no inference source after
the measured checkpoint and makes no adoption claim.
