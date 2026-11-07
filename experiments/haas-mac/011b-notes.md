# 011b — Gate projection and GLU epilogue on Metal: final measured evaluation

**Status: REJECTED — measured slower.** The fusion was verified active (24 fusions on each of
the 48 eligible ON graphs, 1152 total), yet paired median request latency increased in all
three M2 Ultra screen profiles. Nine public inputs, three rounds each, Q4_K/Q8_0/F16. The
separate factor GEMM remains. The cause of the slowdown was not profiled. The speech gitlink
is unchanged.

## Screen result (ordinary worker timing, device MTL0)

Primary timing uses the ordinary worker without profiling or graph debug. The separate stage
timers are diagnostic. Paired latency increase is $100 \times (\mathrm{median}_f \, r_f - 1)$,
where $r_f$ is file $f$'s ON/OFF request-time ratio and each side's time is that file's median
over three rounds. Positive means slower. In words, take each file's median ON and OFF request
time over the rounds, form the ON/OFF ratio per file, take the median over the nine files,
subtract one and scale by 100. RTFx is total audio duration divided by total request time.
The paired median is not the ratio of the separate p50 columns. For Q8_0 the paired increase
is +7.61% while the separate p50 columns (ON p50 over OFF p50) give +11.1%.

| Weight | p50 OFF→ON ms | p90 OFF→ON ms | p99 OFF→ON ms | RTFx OFF→ON | Paired latency increase | Peak process memory OFF→ON MB |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_K | 79.68 → 84.72 | 127.10 → 131.35 | 151.90 → 155.26 | 170.2 → 162.8 | +6.32% | 465.8 → 457.5 |
| Q8_0 | 75.63 → 84.05 | 124.15 → 131.68 | 146.29 → 155.34 | 177.7 → 165.8 | +7.61% | 761.9 → 759.4 |
| F16 | 79.47 → 84.19 | 126.07 → 129.01 | 146.38 → 150.28 | 175.6 → 166.4 | +5.94% | 1349.3 → 1358.9 |

Memory is the `/usr/bin/time -l` peak process memory footprint, in decimal MB (1 MB =
1,000,000 bytes). Faster/slower inputs were Q4_K 1/8, Q8_0 0/9, F16 0/9. Per-round paired
increases spanned 5.5–7.0% (Q4_K), 2.4–14.8% (Q8_0) and 5.2–10.1% (F16). The diagnostic stage
timers show the ON encoder compute stage up 12–21% across short/medium/long inputs.
Peak-memory moves are small and mixed (−8.3, −2.5, +9.6 MB) and carry no bandwidth or
allocator-peak claim.

| Weight | Load OFF→ON ms | First request OFF→ON ms |
| --- | --- | --- |
| Q4_K | 150.1 → 148.1 | 170.1 → 192.2 |
| Q8_0 | 199.9 → 209.1 | 163.3 → 190.4 |
| F16 | 299.5 → 313.4 | 164.9 → 200.6 |

## Candidate retained for reproduction

The ordinary (non-tensor) simdgroup `kernel_mul_mm` in
`ggml/src/ggml-metal/kernels/mul_mm.metal` gains a `MUL_MAT+UNARY+MUL` store epilogue
(`FC_mul_mm_glu` / `FC_mul_mm_glu_swap` function constants, an extra `factor` argument). The
gate projection `W_gate·x` is sigmoid-ed as F32 and multiplied by the factor `a` inside the
gate mat-mul's store. The separate factor projection `a = Wa·x` and its materialization
remain, so this is not a two-projection or concatenated-weight kernel. The pattern and gate
are `GGML_METAL_FUSION_MUL_MAT_GLU` in `ggml/src/ggml-metal/ggml-metal-fusion.cpp`. The
speech site is `Encoder::convolution`, `mul(pw1_a(h), sigmoid(pw1_gate(h)))`.

- Selector `GGML_METAL_MM_GLU`. Unset or literal `0` disabled, literal `1` enabled, any other
  string (including empty) aborts. Read once at the first pattern check, each side in a fresh
  process. Invalid- and empty-name aborts were re-verified.
- Eligibility is a backend gate on shape/type, not model identity. Gate weight contiguous
  rank-2 F32/F16/Q8_0/Q4_K with K = M = 1024. x, factor and the MM/sigmoid/MUL outputs F32,
  contiguous, equal shape. T in 17…384 (T ≤ 16 and T ≥ 385 stay ordinary). src[1] precision
  hint UNDEFINED, so an explicit F32 hint does not fuse. Both projections accumulate in F32.
- Device gate is `has_simdgroup_mm` and no tensor API. The tensor API/M5 path is untouched.
  M5 and Vulkan are untested.
- Theoretical saving only. Two elementwise dispatches per eligible layer (48 across the 24
  layers) and 16 bytes of device traffic per output element (gate-MM and sigmoid write/read).
  Both GEMMs' FLOPs, the factor read and the final write remain. Arithmetic counts, not
  measured bandwidth or allocator-peak gains.

## Tested identities and exact scope

**Runtime sides.** Original runs speech `114c5144` with ggml `e1a9ac12`. Candidate OFF and ON
both run speech `c8b524ef` with ggml `18d581ee`, explicit OFF=0 and ON=1, each side in a fresh
process. The official, normal, API, CLI and worker headers and timing all originate from these
two source pairs. The measured speech gitlink is unchanged at `18d581ee`.

**Runner and binaries.** Runner `dfe7f1de`. The frozen digest `48844d0e…` is the SHA256 of the
runner's `worker.py` source. The speech worker binary SHA256 is `4fccde1a…` on the original
and `66dab2c9…` on the candidate.

**Direct-test identities.** The first direct runner build failed by reaching two opaque graph
fields. The two-line accessor commits `de13b1be`→`e9bf25fc` (cases) and `18d581ee`→`b499a30b`
(kernel) changed only testing lines, call the public APIs for the same positions and iteration
order, and preserve test semantics. The standalone direct tests alone use them.
`test-backend-ops` is separate: the original binary uses cases-only ggml `de13b1be` with the
selector unset; the candidate OFF and ON both use the `18d581ee` binary, in fresh processes
with `GGML_METAL_MM_GLU=0` and `=1` respectively.

**Hardware and scope.** Apple M2 Ultra (76-core GPU, 192 GB unified memory), macOS 26.7.2,
encoder device MTL0, CPU decoder with 4 worker threads, official CPU check at its default 16
threads. Release build, GGML_NATIVE=ON, GGML_METAL=ON, explicit arm64 and matching SDK, -j4.
Public FLEURS ja_jp only, nine screen inputs (three per duration stratum) × three rounds.
Quality runs carry timestamps, latency runs do not.

## Correctness coverage (measured, public aggregate)

1. Direct Metal fixtures. Original, OFF and ON each passed all 332 deterministic cases, and
   the three-side snapshots (100,518,496 bytes each) are byte-equal. ON shows 132 positive
   fusions and zero on the other 200 cases (raw and mapped alias negatives included), zero
   allocation dependencies, and two poison controls per side held.
2. Normal graphs. Four weight types × 16 lengths × 3 sides give 192 normal/retained projected
   equality pairs, bit-exact across all sides. The 48 eligible ordinary ON graphs fuse 24 each
   (1152 total). The 16 ineligible ordinary, the 64 retained-output ON and every original/OFF
   graph fuse zero. Weighted snapshots match per weight per side (873,563,392 bytes) across
   the three sides.
3. CPU cross-backend (`test-backend-ops`, NMSE 5e-4). Each of the three sides is exactly
   248 PASS / 20 FAIL of 268, with nine Q8_0 and eleven Q4_K failures, no new failure, no
   changed printed node, NMSE or limit, and the 20 remain strict
   failures. Eight transposed-X cases were excluded from the CPU-reference comparison. One F32
   case among them was directly reproduced as SIGABRT in the original unfiltered run and in an
   isolated invocation. All eight pass the separate Metal direct test, and no CPU pass is
   claimed for them. CPU activation quantization versus Metal half staging is plausible, not
   proved as the cause for all 20. Equal printed metrics do not prove bits for unsaved
   intermediates.
4. Official encoder. 11 datasets × 3 sides × 3 device/weight conditions give 99 complete runs
   with unmodified strict thresholds. Lowest final SNR 104.4 dB on CPU F32 (>90) and 56.4 dB
   on Metal F32 and F16 (>25). Within each precision/device condition the printed layer and
   final metrics were identical across sides. No raw official tensor bit-identity is claimed.
   CPU is an unchanged control.
5. Public contracts. Three public stage clips exercised the raw TDT paths and fields (27
   records of ids/frames/durations/text/tokens/graphs), plus F16 API cancellation and recovery
   and the worker protocol. The CLI made 36 process invocations, 12 weight/mode combinations
   across three process sides with three inputs each, giving 108 public recognition records in
   four modes, all identical across sides. On the nine public inputs, quality
   text/tokens/segments and latency text differ on zero pairs. CER values in prose are
   fractions, literal 0.073–0.075 and repository 0.043–0.045, same on both sides.

## Scope boundaries

The public 30-input latency set, the 650-input quality set, the 2 s/4 s short prefixes, M5,
Vulkan and any fixed-anchor measurement were not run. No official full-decoder verdict and no
full-corpus quality or adoption claim is made. Within the candidate, 27 OFF/ON quality pairs
and 81 OFF/ON latency text pairs showed zero differences. Against the original, 54
timestamp-quality and 162 latency-text record pairs across both sides showed zero differences.

## Artifacts

Local artifacts live under `outputs/haas-mac-campaign/011b/`, with key gates in
`execution/preperformance-gates.json`, `execution/cross-backend-three-side.json`,
`official-encoder/gates.json`, `public-contracts/gates.json`, `screen/summary.json`,
`screen-original-public-output-equality.json` and `execution/matched-flags-and-links.json`.
The raw manifest `evidence-raw.json` (SHA256
`97ce7f9a171e5ace5ae3469ef691de01b17ca0a47c701ae80549bbba452f7750`) covers 494 local files
with 10,959,268,798 logical bytes plus 38 external retained files with 32,692,597 bytes.
Logical sizes can include APFS-cloned snapshots, and `storage-clone-rpzgqHqc/result.json` is
storage-only. No storage-clone effect enters any speed metric.
