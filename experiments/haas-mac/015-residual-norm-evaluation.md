# 015 evaluation: Residual addition fused with affine layer normalization

Decision: rejected. The current prototype did not show a stable whole-request
improvement across rounds and model types. Nothing is adopted: no production
implementation, no default or threshold change, no full 30-input/650-quality run
and no short6 supplement. Correctness held everywhere it was measured: the
original/candidate differential byte checks were exact for the checked
output, the official F32 stage checks passed their unchanged SNR thresholds,
and no new tolerance was introduced.

## Conditions

Apple M2 Ultra (24 CPU cores, 76 GPU cores, 192 GB RAM), macOS 26.7.2, Python
3.9.6. Clean Release/native/Metal arm64 build with explicit SDK and `-j4`; 74
`flags.make` and 74 `link.txt` match the retained original after normalizing
only source/build paths, and one initial build without explicit `-arch arm64`
was preserved and excluded. Selector `GGML_METAL_RESIDUAL_NORM=0|1`, default
OFF. Pins: speech.cpp `5bb4531bfd9608ad61bb1573f8cc9b6f95d0fd3d`, ggml candidate
`eb97e21752e680fa92f402cc80a84a4700e60cea`, original identical-cases ggml
`8f5f4e2932255663d46f3fedbc180464b2db4ea6`, control parent
`114c514430c8aa1455c077819f8e72e674a48b75` with control ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`, frozen runner
`bc9198776ba66074d6fca55fa2c0a79be7690876`, frozen anchor
`f976ebd8e2fca1f15ddc2e6fc9f99e7090b0f9ea`.

## Numerical gates

Direct `test-backend-ops -o RESIDUAL_NORM`: 71 logical cases per build and 284
poisoned-and-restored executions per build (two deterministic input generations
under OFF and ON); the complete original/candidate dumps compare byte for byte,
and 158 regression cases per build pass over ADD, NORM and NORM_MUL_ADD.

Official F32 features-to-encoder stage check (not an official full KDA/TDT
check): 11 inputs x 3 variants x {CPU, Metal} = 66 case runs, all exits zero.
Minimum printed encoder SNR 104.4 dB on CPU against the preserved 90 dB
threshold, and 56.4 dB on Metal against 25 dB. Provenance only: the initial
case-directory invocation completed zero cases and made no tensor comparison;
with the correct reference root `dumps-converter-lock`, six correct invocations
each enumerated all eleven inputs.

## Normal-path evidence

Q4_K, Q8_0, F16 and F32, 14 inputs per type (3 three-stage and 11 official
audio) and three variants each (original, candidate OFF, candidate ON): 168
ordinary graph executions through the preserved `015-encoder-trace.cpp` helper,
with the final joint projected F32 bytes equal on all sides. 24 new
ADD+NORM-affine fusions are observed per input, with a dispatch reduction of 24
per input; the first Q4 input falls from 958 dispatches (OFF) to 934 (ON).
Fusion statistics were never initialized. Two local encoder index sequences and
their boundaries are observed directly in the graph logs and are identical on
all sides. `n_cb=1` is inferred from the pinned source's mapping, and no
independent command-buffer count was measured; the 24-dispatch reduction does
not establish the command-buffer count.

## Request-speed screen

Conditions: the same candidate binary ran OFF and ON, the fixed campaign anchor
ran separately, decoding used the CPU with four threads, all inputs were warmed
up, and each type ran nine inputs over three alternating rounds. Public outputs
are exact: text, tokens/segments and times unchanged in all nine
timestamp-enabled comparisons and all 27 latency-text comparisons per type.
Raw TDT IDs/frames/duration decisions were not exposed, so raw TDT identity is
not asserted.

| Type  | Paired median | Faster inputs | Round 1        | Round 2        | Round 3        |
|-------|---------------|---------------|----------------|----------------|----------------|
| Q4_K  | +1.42%        | 7/9           | -0.07% (4/9)   | +1.89% (8/9)   | +1.46% (7/9)   |
| Q8_0  | -0.72%        | 3/9           | -1.18% (4/9)   | -0.18% (3/9)   | -0.82% (4/9)   |
| F16   | -1.63%        | 3/9           | +0.40% (5/9)   | -3.59% (2/9)   | -1.27% (3/9)   |

Q4_K is the only type with a positive paired median, and even it opened at
-0.07% with only 4/9 inputs faster in round 1; its duration strata (three
inputs each) stay small and positive: 6-10 s +1.71%, 10-20 s +1.93%, 20-30 s
+1.03%. Q8_0 is negative in all three rounds, and F16 swings from +0.40% to
-3.59%; both end at 3/9 faster inputs. A roughly one-percent median on one
quantized type, over nine inputs, without round-to-round or type-to-type
stability does not establish a stable general request gain; that is the
rejection, and it is why the full and short6 runs were not expanded. No
confidence intervals were computed.

| Type  | Load ms OFF/ON  | First request ms OFF/ON | Peak memory bytes OFF/ON   |
|-------|-----------------|-------------------------|----------------------------|
| Q4_K  | 150.47 / 152.15 | 168.67 / 169.67         | 454,083,688 / 452,166,688  |
| Q8_0  | 196.90 / 196.04 | 166.36 / 161.10         | 755,287,360 / 758,449,400  |
| F16   | 290.41 / 293.67 | 164.53 / 166.90         | 1,354,024,664 / 1,344,489,152 |

Unrounded values, latency percentiles, rtfx and per-stratum medians are in
`015-residual-norm-evaluation.json` beside this note. The separate
encoder-stage timings (7.98, 14.52 and 24.72 s inputs) remain a non-additive
diagnostic; no speedup is taken from them.

## Limits

One M2 Ultra on Metal: no Windows/Vulkan, no other Apple generation, and no
device generality is claimed. The official check covers only the F32
features-to-encoder stage; official joint or full decoder correctness is not
inferred. Raw TDT identity is not established. No full 30-input/650-quality
run, no short6 supplement, no production implementation, default or threshold
change. Numbers come from the coordinator's sanitized aggregate
`approved-facts.json`; the underlying local evidence is referenced, not copied,
at `outputs/haas-mac-campaign/015`: `execution/gates.json`,
`execution/screen-assessment.json`, `execution/result.json`,
`execution/raw-freeze.json` and `screen/summary.json`.
