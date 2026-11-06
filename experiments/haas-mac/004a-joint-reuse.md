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

Status: source preparation only. No build, check or performance result is
claimed by this document.
