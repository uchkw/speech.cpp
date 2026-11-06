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

Status: source preparation only. No build, check or performance result is
claimed by this document.
