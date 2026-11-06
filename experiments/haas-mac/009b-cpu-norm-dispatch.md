# Prototype single-row CPU affine normalization dispatch

Trial 009b is the still-unevaluated CPU small-operation candidate under measure 009. Trial 009a's rejected packed
Q/K/V projection result remains unchanged. This is an independent experiment from campaign
`114c514430c8aa1455c077819f8e72e674a48b75` and ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`; no 009a runtime,
other trial runtime, or other ggml optimization is included.

## Status and original control

This checkpoint is source only. No build, check executable, inference, timing, model hash or Python driver has been
run. Source inspection and `git diff --check` are the only validation so far. The campaign compute lease belongs
to trial 010. Neither repository has been pushed by the owner of this trial.

The cases-only ggml commit is `4bbef873bb6fb0e741244bb5804abf1c742c3198`, on `codex/haas-009b-cases`. It keeps
the original e1a9 runtime and adds the same direct operation cases that the candidate runs. The parent cases-only
branch, `codex/exp-haas-009b-cases`, adds the dedicated check and pins that ggml commit. It is the original runtime
control for the three-way comparison, rather than a candidate whose switch is merely off.

That parent cases-only commit is `3f3655a05ccc14838f2832f549401ba3ef9779ac`. The candidate parent branch pins
ggml `21027944523aabaac5e15b44a8adaeeb28857ff0`, on `codex/haas-009b-norm-dispatch`, whose only runtime change is
47 added lines in the CPU dispatcher. Runtime support is not present in the cases-only branch, so its feature
and trace environment variables have no runtime effect. Check sources are identical in the original and candidate
parent commits.

## Candidate boundary

`GGML_CPU_NORM_MUL_ADD_FUSION=0|1` enables the new dispatcher group. Unset means `0`.
`GGML_CPU_NORM_MUL_ADD_FUSION_TRACE=0|1` emits one diagnostic log message per actual group, from thread zero.
Unset means `0`. Both are read once by `ggml_cpu_init()` and stay fixed for the process. Invalid values abort with
the variable's name. The primary timing environment must explicitly use `TRACE=0`; the counter check uses `TRACE=1`
in separate processes. The existing `GGML_CPU_DISABLE_FUSION` switch is not the comparison switch: it also turns
off the existing RMS normalization fusion.

The new path accepts consecutive `NORM -> MUL -> ADD` nodes only when all operands and results are F32, contiguous,
nonempty, of the same shape, and have one row. This is a generic CPU shape rule, not a Haas-name or width-640 rule.
Widths 1, 639, 640, 641 and 1025 are direct positive controls. Scalar bias broadcasting, multiple rows and strided
inputs are outside this prototype. The existing fusion guard also excludes intermediate outputs, fanout and
intermediate views. Swapped equal-shape operands, offset contiguous input views, shared input operands and an
in-place final add are supported by the same original kernels. In-place normalization or multiplication does not
meet the intermediate-view guard.

The dispatcher calls the existing normalization, multiplication and addition functions, in that order, on thread
zero. Each original function already assigns a single row to thread zero. The original thread parameters, buffers,
stores, normalization reductions and separate multiplication/addition rounding remain intact. Accelerate builds
keep their original vDSP calls. There is no new arithmetic kernel, FMA contraction, graph operator, allocator,
weight copy or change to another existing fusion. An installed CPU abort callback excludes this new path even
when that callback never requests cancellation. Reference-mode CPU plans also exclude it.

Haas's two KDA pre-norms and final norm yield three groups per prediction step, removing six inter-node barriers.
Graph/node counts, matrix products and intermediate memory traffic remain unchanged. Blank steps that run only
the joint do not benefit. With one CPU thread, the original barrier already returns immediately. The new predicate
and its unsuccessful matches can cost time and must be included in the measurement.

The three-recording 009a-off stage medians put the entire decoder at 11.36%, 11.65% and 14.10% of core time for
Q4_K, Q8_0 and F16. These are coarse, unattainable ceilings from deleting the entire decoder, not a prediction for
this six-barrier change or a claim about public worker latency. There is no measured gain for 009b.

## Checks prepared for the next lease

- `test-backend-ops -o NORM_MUL_ADD_ROW -b CPU -j 1` has 21 whole-graph cases for the widths above, swapped operands,
  retained normalization/multiplication outputs, fanout, views, scalar bias, multiple rows and explicit in-place
  aliases. CPU comparisons require exact float bits against the existing reference CPU plan. GPU comparisons keep
  the framework's existing arithmetic bound. This executable's CPU thread count is its existing hardware-derived
  setting; `-j 1` serializes test cases and does not mean CPU1. Record unsupported/skipped GPU cases separately.
- `fastconformer-kda-norm-fusion-check` takes an explicit thread count. Run synthetic F32 and real Q4_K/Q8_0/F16
  weights on CPU1 and CPU4. It executes actual `DecoderStep` objects with two interleaved histories of 17 steps,
  requires exactly three trace events for every enabled normal step and zero when off, and compares all recurrent
  and convolution state, projected prediction and logits bit for bit against a separately retained-intermediate
  graph. The retained graph must report zero new fusion events. Both histories must change at every step and
  retain different final states. Label and duration argmax indices are checked separately.
- The same check covers the five direct positive widths at that explicit CPU1/CPU4 setting. A direct three-op
  graph checks callback-never-stop, stop-after-one-node and stop-after-two-nodes, actual callback firing/counts,
  zero new fusion events with a callback installed, recovery with new input after removal, and reference-mode
  exclusion. These are CPU plan controls; public TDT/API cancellation still needs its own runtime check.
- Metal runs the normal graph as a zero-fusion control. It saves the same normal outputs for an original/off/on
  exact comparison. It does not use the retained graph as a GPU reference, because retaining those intermediates
  also changes Metal's existing fusion opportunities. No Metal arithmetic change is intended by this CPU patch.
- `experiments/haas-mac/009b-check.py` runs original, candidate-off and candidate-on in separate processes under
  the shared lock. It records return codes, commands, environment, binary/model hashes and all three pairwise exact
  comparisons of input frames, state, prediction, logits, direct-row results and decisions. Largest differences and
  SNR accompany strict failures without converting them to passes. Use a fresh output directory for every run.
  Run it only while owning the compute lease; model/evidence hashing is also part of that work.

The check source is in `checks/fastconformer/` because it asserts stage behavior and exits nonzero on failure;
the environment-driven three-way helper stays in `experiments/haas-mac/`. It is not registered as an official dump
check in `checks/run.py`. The synthetic decoder fixture is unchanged from the earlier state-check fixture and
does not bring another trial's runtime into this branch.

Only after these checks, compare the three existing real recordings' emitted IDs, frames, durations and text,
including the GPU-loaded model's lazy CPU decoding path and cancellation/keyword behavior. Then use the frozen
campaign runner, explicit CPU4, same-binary off/on and fixed anchor for the nine-input, three-round screen with
public quality equality. Keep `TRACE=0` for all timings. Full evaluation depends on a clear screen gain.

Official changed-stage KDA/prediction F32 reference checks and Windows Vulkan have not run. This statement does
not claim that frontend, encoder or `joint.enc` references are absent. No production adoption or permanent fork
policy is settled by this source-only prototype, and the trial branch is not a production merge candidate.
