# Trial 007a: relative-position ADD and softmax fusion

This prototype is rejected after the Mac screen for [issue 1](https://github.com/uchkw/speech.cpp/issues/1).
The checks establish equality on their tested inputs and a small encoder gain, but not a stable complete-request
latency gain. It is not an adopted runtime change.

## Source

- speech.cpp base: `114c514430c8aa1455c077819f8e72e674a48b75`, with the original campaign runtime.
- ggml base: `e1a9ac12edbb57402618fd6936598603f5ee6266`.
- Cases first: `c0118325707483b9a7a3b7935e981e7eac01e59b`.
- Deterministic original-kernel comparison: `15c05bd816a904b0cf001921e4ca147643ed1a41`.
- Kernel prototype: `c17f151a0915630c95b70ce3ced39cf2b842741d`.
- Alias cases and aligned controls: `4b448b7d` and `2925ae93`, before the alias runtime change.
- Exact-ac alias support: `f4832b21`.
- Parent branch: `codex/exp-haas-007a`; ggml branch: `codex/exp-haas-007a-metal`.
- Original-kernel branch: `codex/exp-haas-007a-original-cases`, at `0cea543c`, with identical check sources and no
  runtime changes. The original stricter prototype remains preserved by its earlier commit and parent `cbf59a9`.

No earlier runtime trial is included. The measured product is parent `bda92ef4` with ggml `f4832b21`. The adopted
fork's decision table is unchanged because this prototype is not adopted. The coordinator owns publication, with ggml published before its parent gitlink.

## Operation and bounds

FastConformer's global attention computes two separate products, `ac` and `bd`. The relative-position view is
`[T,T,heads]` F32, with `nb1=(2*T-2)*4`, `nb2=T*(2*T-1)*4` and byte offset `(T-1)*4`. The trial preserves both
products and fuses only their addition and the following unmasked, sink-free softmax. It does not use flash
attention or change attention scope, positions, scores, precision or decoder behavior.

`GGML_METAL_ADD_SOFTMAX=0|1` is an experimental switch; unset means `0` and another value aborts. It is read
when matching the pattern. Worker processes keep their recorded value fixed. The direct correctness check changes
it only after the parallel test workers have joined and each device execution has completed.

The new pattern requires M2 Ultra with the tensor API disabled. It rejects other devices, tensor-API devices,
non-F32 operands, broadcast or nonsquare shapes, batches, a view as the first addend, another relative-view
stride or offset, a mask, sinks and an in-place softmax. The generic matcher also rejects a protected ADD output
or a second consumer. Graph grouping and dispatch use the same environment and device gate. The raw allocation
dependency pass always skips this pattern, so it adds no input lifetimes; the full matcher rejects a softmax
output that partially overlaps `ac` or overlaps `bd` after allocation. Exact `ac` reuse is permitted only when
address, type, shape and all strides are identical. No Vulkan or matrix-product kernel is changed.

The graph allocator can reuse `ac` for ADD and then for softmax, since both operations permit in-place allocation.
The fused kernel supports that exact reuse: the maximum pass reads the complete row before `simd_max`, and any
additional SIMD groups synchronize through the existing threadgroup barriers. The exponential/sum pass assigns
each scalar or float4 element to one lane; that lane reads its input before replacing the same element, and no
other lane needs it. Normalization reads only the destination. Distinct rows occupy disjoint intervals because
`ac` and the output share the same contiguous layout. This argument does not permit shifted addresses or `bd`
aliasing, where a write could affect another row's pending input.

The fused load adds two F32 values under local `#pragma METAL fp math_mode(safe)` and `contract(off)` before
the existing scale expression. The rest of the softmax retains its compiler mode, exponent, scalar or float4
selection, thread count, maximum and sum reductions, barriers and normalization. The float4 path reads four
scalar values from `bd`, whose offset and row stride need only four-byte alignment. The ordinary specialization
has no added arithmetic, but equivalence to the independently built original kernel still requires measurement.

The local pragma follows [Apple's Metal Shading Language Specification, section 1.6.3](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf),
dated 2026-06-04. It prevents reassociation of the new sum without changing the softmax's math-function mode.
The downloaded source has SHA-256 `41538b30d2f1140a5b2a0c84ce0a9f7b67bf0c707e224cfea0bfe5a44aa26cf5`.
The Metal compiler accepted the pragma and every deterministic direct comparison passed; emitted instructions
were not disassembled. Non-finite inputs are not an equivalence claim: the existing fast-math path does not define their general behavior.

## Checks

`test-backend-ops -o ADD_SOFT_MAX_RELPOS` contains 44 cases. Fourteen use `T=1,31,32,33,171,172,375`, eight heads
and two asymmetric input sets, including logits near positive and negative 10000. Fourteen controls cover
contiguous or broadcast `bd`, another row stride, padded or viewed `ac`, exchanged operands, rectangular scores,
F16 addition followed by an F32 cast, masks, sinks, in-place softmax, output marking, a second consumer and a
following consumer. The following consumer remains eligible; the other controls must record no new fusion.
Fourteen additional cases place the output exactly over `ac`, at all seven sizes with both input ranges. Two
negative controls place it sixteen bytes into `ac` or over the start of `bd`; both retain valid float4 output
alignment and require zero new fusions. These allocation aliases are installed by the direct Metal check.

The normal whole-graph check compares with the CPU using the existing softmax NMSE bound of `1e-6`. The M2 Ultra
direct check runs two deterministic generations, env 0 then 1, for 176 executions. In separate allocations it
poisons every owned computed tensor with NaN before each execution. For aliased allocations it restores the same
input bytes after poisoning, so the input portion of the output cannot remain poisoned; it additionally requires
the final output to differ from the restored pre-execution bytes. The two roles are distinct. Every case checks
finite final and protected outputs, byte equality between paths and the exact `ADD+SOFT_MAX` count. The model
comparison must separately confirm that the normal allocator actually permits
the fusion, with 24 new fusions expected for an eligible 24-layer Haas encoder.

The same test-only sources are retained with the original ggml kernel. After build authorization, run the original
with `GGML_TEST_ADD_SOFTMAX_EXPECT=0`, and the candidate with `GGML_TEST_ADD_SOFTMAX_EXPECT=1`. Set
`GGML_TEST_ADD_SOFTMAX_DUMP` to a new file under `outputs/` for each run. The dump contains framed case names,
input bytes and output bytes for both switch values. Require the two complete files to match byte for byte and
save their SHA-256 hashes. An existing dump is refused. Leave the generic Metal-fusion-disable and tensor-enable
settings unset. CPU-only and Vulkan runs do not execute this Metal-specific direct comparison.

## Measured result

All builds use Release/native/Metal and four compile jobs. Every build or inference holds the shared performance
lock; diagnostic linking uses one compiler process. The worker uses `/usr/bin/python3` 3.9.6 and four CPU decoder
threads. The product build uses runner `0656426`; the screen uses frozen runner `b899922`, whose additional launch
validation rejects unrecorded `GGML_*` and `SPEECH_*` settings. The request timer is unchanged. No experiment variable
is inherited at launch, and graph logging is absent from every timing process.

- Original and candidate Metal each pass 44/44 CPU-reference cases with the switch unset/default OFF, followed by
  176 deterministic direct OFF/ON executions. The direct byte equality connects the fused path to that reference check.
  Candidate CPU also passes 44/44 cases. The complete original and candidate input/output dumps compare equal with
  `cmp`; both have SHA-256 `81cd2c19a0074a8c7f54b11747723be549ac3ebab8dfa491c48eb7c23297bce8`.
- Existing Metal ADD/SOFT_MAX regression cases pass 298/298. The new fusion's tested square shapes use 32 threads;
  the generic softmax's multiple-SIMD-group path is not a tested execution path of this new fusion.
- The ordinary encoder diagnostic runs three real inputs for each of Q4_K, Q8_0 and F16, with only the projected
  output marked. It uses `GGML_METAL_GRAPH_DEBUG=1`, not fusion debugging or statistics initialization. The default
  Metal command-buffer policy is retained (`n_cb=1` in this ggml version). OFF has the same dispatch counts as the
  fixed anchor. ON has 24 fewer SOFT_MAX dispatches and 24 additional two-operation fusions, with no other count
  difference. All nine projected tensors match the anchor and both switch values byte for byte. These log-enabled
  executions are correctness evidence, not timing evidence.
- The screen compares the same candidate binary with its switch OFF and ON, and the fixed original-runtime anchor,
  on nine inputs in three rotating rounds. Every type has 27/27 ordinary latency texts and 9/9 timestamp-enabled
  public outputs equal to both the disabled path and the anchor. These are a quality screen, not a 650-input result.

| Type | Screen request reduction, ON versus OFF | ON versus anchor | Alternating encoder reduction, short / medium / long |
|---|---:|---:|---:|
| Q4_K | -0.299% | 0.105% | 0.251% / 0.589% / 0.992% |
| Q8_0 | 0.360% | 0.291% | 0.305% / 0.800% / 1.332% |
| F16 | 5.358% | 2.520% | 0.388% / 0.746% / 1.252% |

Request figures are medians of paired per-file latency ratios after each file's median over three rounds. They are
observations, not accepted gains. Q4_K and Q8_0 do not establish a stable request improvement. F16's apparently larger
request change is not explained by the isolated encoder change. Its ON/OFF round ratios are 0.9279, 0.9946 and 0.9464,
while the unchanged OFF/anchor paths also vary. Do not present the 5.358% figure as a stable kernel speed-up.

A bounded stage diagnostic rotates OFF, ON and anchor for three rounds, with three real inputs and three measured
repeats per input in each process. It warms each input immediately before its measured repeats; the worker screen
instead warms all inputs before measuring requests. The helper keeps the existing `steady_clock` intervals and
prints raw IDs, emission frames and durations only after the final timer ends. It links the preserved product
libraries, not another model runtime. All 243 measured outputs agree in IDs, frames, durations, token and graph counts,
and text across conditions and repeats. This is separate evidence from the worker's public-output screen.

The F16 stage core decreases only 0.574%, 0.768% and 0.807% on the three inputs. Its OFF/anchor decoder ratios change
from the screen's sequential stage observations of 0.659/0.646/0.630 to 0.938/0.978/0.995 in the alternating diagnostic.
F16 ON/OFF decoder round medians are 0.980, 1.209 and 0.941. The changed order and rounds expose variation in an
unchanged stage; they do not identify a unique cause. Stage core time excludes the worker's outer request processing
and graph destruction, and is not interchangeable with warm request latency.

The compiler-flag audit found a literal difference: the candidate explicitly passes `-arch arm64` and the Xcode
`MacOSX.sdk` sysroot, while the fixed anchor leaves those implicit. The literal-equality assertion and both flag sets
are preserved. Other recorded optimization, CPU feature and preprocessor flags agree. Effective compiler defaults
have not been independently shown equal; this is not evidence that different SDKs were used. Same-binary OFF/ON has
identical flags and libraries and does not have this between-build limitation. The F16 model hash and CPU4 setting
are identical throughout. Load, first-request and peak-memory observations remain in the screen report without a
claim of a stable gain in those metrics.

The trial is rejected at the screen: preserve the 0.25% to 1.33% local encoder gain, but do not generalize it to an
end-to-end improvement. No full 30-input/650-input run or short-input supplement is performed. Official F32 stage
references, Windows Vulkan and M5 execution remain unavailable or untested, not passed. No runtime is integrated.

## Reproduction and evidence

The ignored artifact root is `outputs/haas-mac-campaign/007a`. Its `checkpoint.json`, `checks.json` and
`evidence-sha256.json` identify the measurements, diagnostics, source blobs, preserved binaries and hashes.
`screen/summary.json` is the original common-runner result; `screen-review.json` adds per-file, round and length
comparisons. `normal-encoder/summary.json` holds graph dispatch and tensor equality, and
`stage-alternating/summary.json` with `stage-review.json` holds the bounded path/timing diagnostic. None is an official
reference dump. The original PDF remains in `source-evidence/`.

The reusable diagnostic sources are `007a-encoder-trace.cpp` and `007a-stage-path.cpp` (checkpoint `1afbcb5`). Link each
with the compile flags and static-library list recorded by the corresponding preserved product's
`CMakeFiles/fastconformer-timing.dir/flags.make` and `link.txt`; the raw build manifests record the exact commands and
hashes. The source snapshot is distinct from the unchanged measured product commit. The ordinary trace helper is a
byte-identical copy of the helper used in trial 006a. Only its final projected output is marked.

`007a-stage-compare.py` (checkpoint `0129883`) runs the limited diagnostic from its assets and two helper build
manifests. It rejects inherited experiment variables, preserves the order, verifies binary/model/input hashes and
compares raw paths after timing. The runtime and every dump remain fixed after evaluation; any new experiment uses
a new output directory. Publication belongs to the coordinator, with ggml published before the parent gitlink.
