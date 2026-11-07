# Trial 013 evaluation: normal allocation rejects depthwise fusion

The current depthwise convolution + bias + SiLU prototype is rejected for Haas on this Mac. Its direct Metal
checks pass, but it activates no fusion in the tested normal encoder allocation path. No primary latency screen,
full corpus, short-audio screen or public ASR quality comparison ran. There is no measured speed difference,
including no measured 0% result. The flag remains default OFF and no production or cumulative variant adopts it.

## Sources and execution boundary

| Item | Recorded source |
| --- | --- |
| Candidate speech.cpp | `1ab1d9c3eca96daa5f8d3ade37bae9b06cd01ea1` |
| Candidate ggml | `2b5ce8e9215177354519b42e84b0853de9196ed0` |
| Original speech.cpp control | `c1299723c958cd7f50d6273b3cdce769e215ffb2` |
| Original ggml with identical tests | `6e18078cfe8257378826cf16b07b378e6d530a57` |
| Frozen campaign runner, not executed | `54eed08312c70151ecf3cd872e83f09579be6605` |

The original ggml runtime is the campaign base `e1a9ac12edbb57402618fd6936598603f5ee6266`; the control adds the
candidate's tests without its kernel. Candidate OFF and ON use the same binary in separate processes with
`GGML_METAL_DW_SILU=0` and `1`. Each build was made from a clean recorded checkout on the M2 Ultra with native
Release Metal, explicit arm64 and the same Xcode SDK, under the campaign lock with `-j4`. All 74 speech and
28 standalone ggml `flags.make` files match after source/build path normalization.

The reviewed normal C++ helper was compiled against each build's exact static libraries. Its compile/link
commands, source copies, cache identity, flags, archives and executable hashes are retained. Existing CMake
commands and check executables performed the evaluation. The proposed Python execution tools were never
adopted or executed; their initial and revised development artifacts remain separate from the evidence manifest.
No model, audio or execution log was sent to the coding API.

The fixed campaign anchor's identity was verified, but it was not timed. Its implicit architecture/SDK settings
remain distinct from these explicit matched builds; this evaluation makes no performance equivalence claim
between those build configurations.

## Correctness checks

| Check | Actual result |
| --- | --- |
| Direct Metal original/OFF/ON | 61 cases and 2 poison guards pass in each configuration |
| Direct fusion coverage | 24 eligible cases fuse ON; 37 rejection controls do not; original/OFF fuse 0 |
| Direct arithmetic | Same-binary comparison and both complete original/OFF/ON file comparisons are byte-exact; no added allocation dependencies |
| CPU-reference backend comparison | 49 actual Metal-versus-CPU cases pass in each configuration; 0 unsupported |
| Related backend regressions | Per configuration: depthwise 16, SiLU 8 and ADD 84 pass |
| Unsupported regression cases | SiLU 4 and ADD 72 are identical across configurations and are not counted as passes |
| Normal projection output | 81 graph executions pass; all 6 original-versus-candidate full output comparisons are byte-exact |
| Separate graph-debug output | All 9 debug-level-1 output files and the extra debug-level-3 Q4 ON file match their corresponding normal output |

The direct set includes 11 mapped-buffer controls and one padded-input case that are deliberately outside the
CPU-reference set. The direct check verifies real fusion dispatches and poisoned outputs, rather than treating
an empty case list as success. Its separately allocated inputs do not establish normal allocator eligibility.

The normal helper uses only the final encoder-to-joint projection as graph output, one ordinary allocator,
explicit `MTL0` with registry `MTL`, and subsampled lengths `1,31,32,33,171,172,375,33,1`: seven unique
lengths and nine executions. Each process checks finite nonzero output, dimensions and exact repeats at 33 and 1. Three model types, Q4_K/Q8_0/F16, times three
configurations times nine shapes give the 81 executions. Synthetic features establish arithmetic and allocation
stability; they do not establish recognition quality.

## Why the normal graph cannot use this prototype

The separate log-sink launcher runs the same graph with the default command-buffer partitioning. It adds no
retained output, allocation change or runtime counter, and uses no fusion-debug flag. Debug level 1 covered all
81 graph executions and observed zero depthwise fusions, including every ON graph. The convolution, bias ADD
and SiLU remain separate dispatches.

The initial aggregation searched for five fused operations, matching the raw pattern that includes PERMUTE
and RESHAPE. The backend removes empty view operations before encoding, so an actual fusion would report three
operations. The original analysis is preserved; `graph-debug/corrected-analysis.json` records the source-based
correction and still finds zero depthwise fusions. The observed three-operation fusion messages belong to NORM.

One bounded `GGML_METAL_GRAPH_DEBUG=3` Q4 ON run covered the same nine shapes, with normal projected bytes
unchanged. Its 216 Conformer patterns have 120 final-output/input overlaps and 96 final-output/packed-kernel
overlaps, with a union of 216 and no bias overlap. Independent read-only review re-extracted every raw record
and matched these counts. Kernel extents equal `9*C*4` bytes; input, convolution output and SiLU output extents
equal `C*T*4`. F32 allocation sizes and zero-offset full-shape views show that these are real overlaps, not
padding or inflated base-view ranges.

Every pattern therefore satisfies an unconditional local FULL alias rejection condition. The ordinary graph
may reuse an input or packed-kernel region after convolution consumes it; a fused dispatch cannot safely use
that placement. The first condition that returned false was not instrumented, so the evidence establishes
sufficient rejection conditions, not the order in which guards rejected the pattern. No guard was weakened,
allocator changed or intermediate retained to make the optimization activate. This verdict concerns the current
prototype and observed allocation path, not every possible depthwise fusion design.

## Official reference scope

The existing check executables passed 99 stage executions against the verified official dumps:

| Stage and device | Executions | Displayed minimum SNR | Executable threshold |
| --- | ---: | ---: | ---: |
| Frontend, CPU F32, candidate ON | 11 | 103.3 dB | 90 dB |
| Encoder, CPU F32, original and candidate ON | 22 | 104.4 dB | 90 dB |
| Encoder, Metal F32, original/OFF/ON | 33 | 56.4 dB | 25 dB |
| Encoder, Metal F16, original/OFF/ON | 33 | 56.4 dB | 25 dB |

The minimum values are printed to one decimal place; pass/fail comes from the existing executable's unrounded
comparison. Official frontend, encoder and joint.enc dumps are available. This evaluation executed frontend
and encoder only; joint.enc was not independently checked against its official dump. A complete official
KDA/TDT reference is unavailable. Windows/Vulkan device execution was unavailable and is not claimed.

## Frozen evidence and skipped work

`outputs/haas-mac-campaign/013/evaluation.json`, `checks.json` and `check-summary.json` record the decision and
scope. `independent-review.json` records the source/range review. `evidence-manifest.json` freezes 746 local
files and 6 external files, 273,370,400 bytes in total, with 221 existing hash records verified. Its SHA-256 is
`9405b479f93900c36ce19bb2ec3e64ca06f0296af3068f87e31f91d22e34d9f5`.

The manifest retains raw outputs, initial and corrected analyses, source copies, command records, frozen
runner, build identities, flags, static libraries and executables. `freeze-scope.json` explicitly separates
unused execution-tool development and generated build bookkeeping. Pinned models and official arrays were
verified before the checks and were not rehashed during final freeze. No raw result was overwritten.

Because normal activation failed, the planned nine-input, three-round latency screen and fixed-anchor
comparison were skipped. Full 30-input timing, 650-utterance quality, short-audio evaluation, CER, public text,
tokens, timestamps, segments, load-time gains and memory gains were not measured. Diagnostic wall times are
not latency evidence. No further computation is required to reject this prototype; a different allocation or
fusion design would be a separate experiment with its own checks and measurements.
