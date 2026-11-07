# Trial 016 evaluation: attention head-store (write attention products in merged-head order)

**Decision: REJECTED — the prototype is rejected for insufficient full whole-request benefit, and the default stays off.** The selector `GGML_METAL_HEAD_STORE` remains `0` (unset behaves as `0`); nothing is adopted. The Q4_K screen result — +2.786% paired median with 9/9 input medians faster and all three screen rounds improving — justified only one follow-up: the full 30-input Q4_K run. That full run measured +0.316% paired median on 17/30 inputs with ON p90 and p99 above OFF, and those full-run observations govern the decision. Q8_0 and F16 remain screen-only, and short6 was not run. These results are never combined with the separate private quantization study and imply no adoption.

Scope: M2 SIMD-group F32 contiguous attention `MUL_MAT -> PERMUTE -> CONT`, shape `[T,128,8]`, `T>=64`, same Metal buffer and no destination overlap; the M5 tensor-API path is excluded by source. Model file type is distinct from the F32 intermediate operand type, so Q4_K, Q8_0 and F16 model files all exercise the same F32 intermediates. All numbers below come from the approved aggregates; full-precision values are in the companion JSON.

## Correctness and validation summary

**Initial invalid fixture and corrected case.** The original direct run aborted at graph construction of case 47: its `transpose_a` first operand is rejected by public `ggml_mul_mat` for every backend, so it tested no kernel. Before the abort, 46 direct rows were produced and 46/46 initial CPU-reference comparisons had passed per build; zero full direct gates completed and the candidate direct run never started (the SIGABRT record and partial dump were retained). The corrected case 47 is a legal `padded_head_a` with four floats of padding between heads, eligible for the CPU reference; every other case and runtime gate is unchanged.

**Corrected direct checks, per original/OFF/ON process:**
- 47 CPU-reference logical cases completed per original and candidate build, under the existing NMSE bound of 5e-4.
- 52 direct cases per process — 14 positive, 38 controls (5 mapped controls are Metal-direct-only) — plus two poison guards per process.
- Actual synthetic fusions observed: 14 with ON, 0 with original, 0 with OFF; zero introduced allocation dependencies.
- Complete result dumps matched across all three processes: 31,451,648 bytes each, identical sha256.
- 2,755 related MUL_MAT/CONT/allocator regression cases per build.

**Ordinary allocation eligibility (not a fusion count).** Across Q4_K/Q8_0/F16 with 14 inputs each: 42 model/input pairs, 1,008 observed ranges, 3,024 side records matched across original/OFF/ON. The 36 `T>=64` pairs had 864/864 eligible ranges — all in one underlying Metal buffer with zero overlap against either input. The six `T<64` control pairs had 0/144 eligible.

**Actual named normal encoder fusions and projected bytes.** Candidate ON performed 288 attention head-store fusions per model type (Q4_K, Q8_0, F16, F32; 14 inputs each; 1,152 total) across 168 ordinary executions and 56 compared model/input groups, with 0 fusions on `T<64` controls. Complete final projected F32 bytes were equal across original, OFF and ON for every group. First-process Q4 dispatch counts: 958 original, 958 OFF, 934 ON. Two local encoder index sequences were observed and matched on every input; runtime fusion counters were never initialized, so the command-buffer count is source-inferred only, not measured.

**Official F32 SNR.** 66 case runs across 6 processes (11 per process): CPU bound 90 dB with lowest printed encoder SNR 104.4 dB; GPU bound 25 dB with lowest printed 56.4 dB. This covers features through encoded output, not full KDA/TDT.

## Build and provenance

- Measured runtimes, one retained product per side: the original ordinary-encoder normal helper was linked to the retained original product — speech parent `114c514430c8aa1455c077819f8e72e674a48b75` with ggml `e1a9ac12edbb57402618fd6936598603f5ee6266` — and the candidate normal helpers were linked to the retained candidate product — speech parent `efa6a4ea491a576198a1f0dc2a7d8159f3646a48` with ggml `9066e6f2471eabed8632ded32d4514b6429d9790` (candidate product binary sha256 pinned in the JSON). Each ordinary helper is a distinct executable linked to the relevant product libraries, not one shared binary. Only the primary OFF/ON worker timings use the same retained candidate executable; the fixed campaign anchor used its own respective executable with implicit architecture/SDK flags.
- Later test-only commits — corrected check parent `d930bbf5185c41c85b8175834666a2532c4a157b`, corrected original cases `ccadee2edb3db66f37e840e921f0d2b2c5940932`, corrected candidate ggml `f9d1f765dcc119403206775a137d5a03448c7e13`, initial original cases `40830de1f9b83695cce372a387763ef3600b9951` — change only the linked ggml revision, the experiment note and the shared synthetic test header; they were used only for the corrected standalone checks and change no runtime.
- Only the original and candidate product builds have matched compile/link flags: 74/74 flags.make and link.txt files after normalizing source/build paths. The fixed anchor `f976ebd8e2fca1f15ddc2e6fc9f99e7090b0f9ea` uses implicit native architecture/SDK flags, so it is context only and outside that 74/74 control.
- Conditions: Apple M2 Ultra (24 cores, 192 GiB), macOS 26.7.2, Python 3.9.6, Release/native/Metal with explicit arm64 and macOS SDK, `-j4`; frozen runner `bc9198776ba66074d6fca55fa2c0a79be7690876` with per-file sha256 pins.

## Worker screen — all formats

Warm persistent worker, public FLEURS, Metal encoder, CPU decoding with four threads, alternating three rounds; graph-debug was disabled for worker timing. Metric per input: `100*(1 - median_ON/median_OFF)` from the three rounds, with per-round medians pairing the same input; the reported primary is the median across inputs. Each format used the same nine public inputs for both the three-round latency pass and a separate timestamp-enabled quality pass; there was no separate corpus of nine. The 9 quality comparisons per format represent the quality pass, and the 27 no-timestamp latency text comparisons per format represent the latency pass — different passes — and all recorded zero changes.

| Format | Paired median reduction | Inputs faster / slower (of 9) | Round medians |
|---|---|---|---|
| Q4_K | +2.7859% | 9 / 0 | +2.3517%, +3.5384%, +2.0645% |
| Q8_0 | +0.6783% | 5 / 4 | +0.0136%, +1.0171%, +0.6946% |
| F16 | +0.7788% | 6 / 3 | −1.2148%, +3.7310%, +0.1548% |

Only Q4_K combined a materially larger paired median with all 9 input medians faster and all three round medians improving, so only Q4_K was promoted to a full run. Q8_0 and F16 stayed screen-only.

## Q4_K full comparison — 30 latency inputs, three rounds

- **Paired median reduction: +0.3161%**, with 17/30 input medians faster and 13/30 slower. Strata (10 inputs each): 6–10 s +0.2846%, 10–20 s +0.3161%, 20–30 s +0.3249%.
- **Round medians:** +0.0519%, +0.2588%, +0.5293%; faster inputs per round 16 / 16 / 16 of 30.
- **Request latency and RTFx (n = 30 inputs):**

| | p50 (ms) | p90 (ms) | p99 (ms) | RTFx |
|---|---|---|---|---|
| OFF | 81.6570 | 129.6454 | 150.9833 | 168.7497 |
| ON | 81.3987 | 130.4375 | 151.1836 | 168.8482 |

ON improves p50 by about 0.26 ms but raises p90 by 0.79 ms and p99 by 0.20 ms; RTFx is essentially unchanged. With the whole-request median gain at only +0.316% and both tail percentiles worse with ON, the benefit is insufficient — these full-run observations, not the screen, govern the rejection.

- **Public quality and text identity:** 650 timestamp-enabled equality checks with 0 text changes, 0 public-token changes and 0 segment changes; 90 no-timestamp latency text comparisons with 0 changes. These comparison counts are separate from the 30 latency inputs. OFF and ON quality aggregates are identical (literal CER 0.074668, repository CER 0.046557; 2,608 literal edits over 34,928 characters; 1,520 repository edits over 32,648 characters).

## Scope limits

- M2 Ultra only; no Vulkan or other Apple-generation verification.
- The official F32 reference covers features through encoded output, not full KDA/TDT; the final joint.enc byte equality is an original-runtime differential.
- The private-recording quantization study is independent, contributes no additive gain here, and is never combined with these results.
- No production integration or permanent ADR change; the prototype is not adopted and the default is off.
- short6 was not run; Q8_0 and F16 have no full run.
- Raw TDT ids, frames and duration choices were not observed.
- Raw manifest retained under the evidence root: 641 artifacts (229,662,869 bytes), including 34 retained binaries (182,288,136 bytes).

## Evidence (local, repository-relative, under `outputs/haas-mac-campaign/016/`)

- `execution/gates.json`
- `checks/head-store/initial-failure-classification.json`
- `execution/screen-assessment.json`
- `execution/full-q4-assessment.json`
- `execution/corrected-product-reuse.json`
- `execution/raw-freeze.json`
- `execution/result.json`
- `allocator/result.json`

Companion machine-readable record: `experiments/haas-mac/016-attention-head-store-evaluation.json`. Design history and source boundary: `experiments/haas-mac/016-attention-head-store.md`.
