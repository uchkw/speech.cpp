# Trial 007b stopped evaluation, tile content attention with a dynamic relative-score mask

The candidate is rejected. On its first enabled Q4_K input, T=1 or 7 frontend feature frames, the enabled encoder
failed the preregistered requirement that its ordinary and retained projected outputs agree bit for bit. This is
a failed internal consistency gate, not a measured recognition or performance outcome. Neither was run and no
relaxed tolerance was applied.

The evaluation ran on Apple M2 Ultra, macOS 26.7.2, Release/arm64 with a matching SDK, GGML_NATIVE and
GGML_METAL enabled, 4 threads, Haas ja v1.0 Q4_K, with sides original unset, candidate OFF literal 0 and
candidate ON literal 1 under `SPEECH_HAAS_REL_FA`. Both enabled graphs ran with 24 FA graph nodes, graph node
counts and not GPU dispatch counts, and all 24 retained masks were finite with maximum absolute value 24.875 and
no zero or subnormal entries. The helper checked equality before saving its outputs, so the failed ON run left no
raw tensor and exited 1 without a successful terminal. Original and candidate OFF covered all 16 synthetic
lengths with 16 normal and 16 retained graphs per side, zero FA nodes and bit-exact ordinary versus retained
projected bytes, and their 448 F32 files per side matched in SHA256 and extents. This is not a completed
three-side raw comparison, because the ON side saved none.

## Failure-only diagnostic

The independently compiled failure-only helper `36e6ce8df02179d526b8352ca3c779abc116172c` used the exact same
runtime speech `566d2c5e141183daae681c0af29325d7f19a1bc7` and ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`,
the same flags, graph and tensor markings, and saved the two already computed host outputs in the same failure
branch without changing the original runtime or adding graphs, outputs or node markings. The rerun exited 1
without a successful terminal and produced two F32 outputs of 2,560 bytes each with actual shape one frame by
640 channels. 602 of 640 elements differ in raw representation, a count of changed F32 elements and not of
flipped bits. The first differing index is 0, the maximum absolute difference is 3.933906555175781e-6, and both
sides were finite.

The retained graph keeps the subsampling, layer, encoded and projected tensors plus 24 F16-to-F32 mask readouts,
so retention can change live ranges, allocation placement and fusion. The location and cause of the difference
and its effect on recognition are undetermined, and this report does not ascribe it to F16 rounding, tail
padding, a bad FA kernel, nondeterminism or model accuracy. No further localization or repeated attempt was made
beyond this single diagnostic.

## Variant, selector and checks

The dense query-dependent relative-position prepass remains. Each layer computes the projected positions and the
dense positional product $(q+v)P^T$ of shape $[2T-1, T, 8]$, where query $i$ and key $j$ read position index
$T-1-i+j$ through the relative shift, then copies to contiguous F32, scales by $1/\sqrt{128}$ and casts to a
per-layer contiguous F16 mask $[T, T, 8, 1]$. FA_EXT receives $(q+u)$, K and V as F32 tensors $[128, T, 8, 1]$
with that mask, scale $1/\sqrt{128}$, max_bias 0 and logit_softcap 0, and tiles QK, online softmax and PV. K and
V each take an explicit CONT into this canonical contiguous F32 layout, and the old content-score tensor,
separate softmax, transposed V copy and PV product are absent. The ggml e1a9 source is retained, the independent
noncontiguous K/V pad fix d27e2a6b was not applied, and at T=1 each canonical K/V row is 512 bytes.

The potential arithmetic differences were not localized as the failure cause. The original rounds
`(AC + BD) * scale` in F32, while FA_EXT rounds `AC * scale + F16(BD * scale)`, and distributing the scale
changes rounding even before the mask cast. The pinned Metal F32 FA kernels stage Q in half while reading K and
V in F32, whereas ordinary large-batch GEMMs stage both operands in half (`mul_mm.metal`). Online maxima,
exponential sums, rescaling and reduction order also differ from the old separate softmax and GEMM.

`SPEECH_HAAS_REL_FA=1` enables the variant for global Haas ja v1.0 on Metal only, with 24 unbiased layers, width
1024, eight heads of 128, KDA prediction and 1 to 384 subsampled frames. An unset switch or literal 0 keeps the
original graph, and any other value, another model, another backend or an out-of-range input throws a named
error, so enabling never silently selects the old graph. Actual invalid-value, empty, CPU ON and T385 negative
controls were rejected. The wrong-model guard is source only, no alternate public FastConformer model was
available, and unsupported-op denial was not observed on this supporting M2.

CPU and Metal each passed all 20 independent direct fixtures with per-head masks exact against the scalar
relative index. The worst NMSE was 1.11627716e-13 on CPU and 1.15624144e-13 on Metal, both strictly below the
pinned 5e-4 bound, while the shared-head control at 0.00875775384 and the reversed-position control at
0.983672817 both exceeded it. This direct CPU synthetic check is distinct from the family's intentional CPU ON
rejection.

None of the planned official 90 dB CPU and 25 dB Metal encoder checks were run. No public TDT, API, worker or
CLI recognition, 9x3 screen, 650-input quality set or short timing was run, and M5 and Vulkan are untested.
Speed, RTFx, memory effect and CER are unmeasured, with no 0 percent or other value claimed.

## Evidence

Tested revisions are candidate speech `566d2c5e141183daae681c0af29325d7f19a1bc7`, original speech
`114c514430c8aa1455c077819f8e72e674a48b75`, ggml `e1a9ac12edbb57402618fd6936598603f5ee6266` shared by both
sides, failure-only helper `36e6ce8df02179d526b8352ca3c779abc116172c`, offline triplet comparator
`82938006b1dd807d60c10db7a6c449679c25a03f` and prepared runner `dfe7f1ded3cfbadbe07d80560effba1c07a179bf`. The
failure-only result carries SHA256 `6ed40e8647280aee69f234c0dd00bb6961ca4a893aabfe088053ed395e95bd28`.

Local artifacts live under `outputs/haas-mac-campaign/007b/`. The first failure is recorded at
`execution/normal-retained-first-failure.json`, the failure-only run at `retained-diagnostic-t1/result.json`, the
raw manifest at `evidence-raw.json` and the selector checks at `selector-checks/result.json`. The raw manifest
covers 973 local files of 489,686,509 bytes plus 11 separately retained files of 9,875,822 bytes, with full
SHA256 `a8a0e5c2c2a84b90802c789337bbb5a6dc72141307d3c0555eba0632002d6b53`.
