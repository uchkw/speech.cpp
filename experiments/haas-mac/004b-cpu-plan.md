# Retain the TDT prediction-step CPU plan

This independent prototype starts at campaign commit `114c514`, with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. It contains neither resident KDA
state nor blank-joint graph reuse. `SPEECH_HAAS_TDT_CPU_PLAN=0` selects the
existing path; `1` selects the experiment. Other values are errors.

`DecoderStep` already keeps its fixed graph and tensor allocation. The trial
adds a CPU execution plan after its first ordinary computation and retains it
until the request destroys the step. The plan is freed before its graph and
allocator. It captures the CPU thread pool, thread count and abort callback;
these stay fixed during the request, and the next request creates a new plan.
Input uploads, full state readback, prediction and logits readback remain as
before. The separate blank-joint graph is unchanged.

TDT selects this plan only when its actual decoding backend is the CPU. The
GPU still calls the ordinary graph-compute path. An explicit CPU-plan request
to a non-CPU `DecoderStep` is an error, rather than an attempted plan followed
by fallback. `Graph::compute_again` accepts the owning step's plan and uses
the existing error translation around its synchronous CPU computation.

`fastconformer-cpu-plan-check` compares full prediction state, prediction output
and joint logits after every step in two interleaved requests. It checks the
CPU abort callback before and after a plan is retained, recovery with the same
callback data, and a new request after removing the callback. TDT fixture cases
compare token IDs, frames, durations, graph counts and cancellation points;
they also cover boosting, blank chains and zero duration. On a GPU the check
verifies the unchanged path and rejects an explicit CPU-plan request. The F32
fixture is differential evidence, not an official NeMo/FLA reference.

After a locked Release/native/Metal build with four jobs, run the check with a
model path or `--synthetic OUTPUT_DIR`, then `cpu` or `gpu`, and a thread count.
Use Q4_K, Q8_0 and native F16 with four CPU threads, plus the synthetic CPU F32
and Metal checks. Official Haas stage references and Windows Vulkan remain
separate requirements, unavailable in the current environment.

Measure with the common campaign runner from the same fixed anchor as 004a,
using explicit `0`/`1` environments and initially nine inputs in three
alternating rounds. A combined 004a/004b result is not this trial. Preserve
load, first request and peak memory separately. Native F16 on the CPU also
uses the plan, so it is not an unchanged-path control.

## Evaluation on November 7, 2026

The prototype is rejected at the screen stage. With Python 3.9.6 and four CPU
decoder threads on the M2 Ultra, nine inputs in three alternating worker
rounds give paired request reductions of 0.09% for Q4_K, 0.21% for Q8_0 and
1.02% for F16 against the fixed campaign anchor. Q4_K paired round ratios are
1.01372, 1.00279 and 0.99901. All 81 latency texts and 27 timestamped public
outputs match exactly. These small differences do not establish a gain.

The supplementary three-input stage comparison uses three alternating
fresh-process rounds, each with three warm observations per input. Decoder
total candidate/baseline ratios are 0.95565/0.89024/1.02030 for Q4_K,
0.96432/0.91807/1.12278 for Q8_0, and 0.96178/1.00427/1.03816 for F16.
Every type has rounds that improve and rounds that regress. Median ratios
across input/round pairs for core time are 0.99621, 0.99413 and 0.99383,
respectively. The much larger Q4_K decoder difference in the screen's separate
single-process stage measurement does not reproduce. All 81 paired stage
text/token-count/graph-count observations match. No full 30-input latency,
650-input quality or short-prefix extension is performed.

The measured clean source is `1f1a1f60411df0e16d7c4f80ca1b39b92ec0a70f`,
built with Release/native/Metal and four build jobs. All nine differential
conditions pass. Synthetic F32 on CPU with one and four threads passes 164
assertions each; Metal passes 154. Q4_K/Q8_0/F16 each pass 135 on CPU and 125
on Metal, with four threads. These cover full state/prediction/logit equality,
CPU abort and recovery, a fresh request with a new callback setting, explicit
non-CPU plan refusal, TDT cancellation after reuse, and synthetic history
sensitivity with identical final label/frame inputs.

Build and check manifests, raw observations, load/first-request/peak-memory
measurements and source snapshots remain in the external `004b/` and
`builds/004b-cpu-plan/` campaign artifacts. Official Haas F32 stage references
and Windows Vulkan remain unavailable. No adoption is claimed.

`stage-pairs.py` is the byte-identical supplementary driver used in 004a and
004b. Replay with `/usr/bin/python3`, `--run` pointing to the screen's
`run.json`, `--assets` to the matching `assets.json`, and `--out` to a new
directory. It checks Python, binary, model and input identities and holds the
shared lock separately per process. This evaluation commit changes no
inference source after the measured checkpoint.
