# Reduce TDT host copies and allocations

This prototype starts at speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`. The switch `SPEECH_HAAS_TDT_BUFFER_REUSE=0|1` defaults to
off and only selects the KDA TDT path. Other prediction networks do not read this switch. An invalid
value on KDA fails explicitly.

## Changed ownership boundaries

| Location | Original operation | Enabled operation |
| --- | --- | --- |
| TDT frame selection | Allocate and copy `joint.hidden()` floats into a vector before each prediction/joint evaluation. | Pass pointer and count into Graph's owned upload. Haas has 640 floats, or 2,560 bytes, per frame. |
| Graph float `input` and `set` | Check the element count and copy/convert into `Upload::bytes` immediately. | Keep the same checks and immediate ownership. Vector signatures remain and delegate to the pointer/count overloads. |
| Projected prediction readback | Allocate a new vector each outer TDT iteration. | Read all values into one request-local vector, retaining its capacity. |
| Logits readback | Allocate each outer iteration and again for each blank-joint result. | Read all token, blank and duration logits into one request-local vector, retaining its capacity. |

The off path retains its temporary frame vectors and fresh prediction/logits results. Every graph input
still owns its bytes before `compute`; no pointer survives an `input` or `set` call. The float input
validation, F16 conversion, and float32-only readback rules remain unchanged. Integer inputs are
untouched.

## Unchanged device and host transfer behavior

The prediction state still travels through the same host readback and upload. The encoder projection
still returns to the host, as CPU decoding needs it. The blank-joint graph is still rebuilt and computed
at the same point. Full logits are still read, the first of equal maxima still wins, and the existing
separate boosted-token vector is still overwritten for every selection. Progress/cancellation, blank
duration zero, maximum symbols, graph counts and the token/duration update order remain unchanged. This
does not combine resident state, cached joint graphs, device argmax, command reuse, or shared
plain/boosted prefixes.

## Checks

`fastconformer-tdt-buffer-check` has three modes:

```sh
fastconformer-tdt-buffer-check --graph-only cpu 4
fastconformer-tdt-buffer-check --synthetic "$ARTIFACTS/synthetic" cpu 4
fastconformer-tdt-buffer-check "$MODEL" cpu 4 "$DUMP/joint_encoded.npy"
```

- The graph-only cases compare both public float overloads, F32 and F16 conversion, source mutation
  before compute, source expiration after set, and the same exception category/message for wrong
  counts, types and foreign inputs.
- The step cases compare complete KDA states, projected predictions and every joint logit bit for bit
  over 16 changing labels and frames. Reusable predictions and logits are deliberately overwritten
  after each iteration, including token boosting and poisoned blank/duration entries, so the next read
  must replace every value. Their data pointers must remain stable after the first allocation.
- The deterministic synthetic F32 fixture and controlled frames cover plain/boosted paths, empty input,
  zero duration, blank chains, max symbols, cancellation before and during decoding, a final token
  extending past the input, and a plain request after a boosted or cancelled one. The same RNN-T greedy
  fixture retains its expected tokens/frames.
- Each optional real projection compares off/on plain and boosted raw token IDs, frame indices,
  durations, graph counts and exact progress sequences, with callback limits of zero, two and
  unlimited.

The fixture files are copied from the existing 004a differential check; they are not official checkpoint
weights or official KDA/TDT dumps. The optional `joint_encoded.npy` files may come from the pinned
official CPU encoder/joint projection. Decoding them with two C++ paths remains a differential decoder
check, not an official decoder check. The official full KDA/TDT reference is unavailable in this
environment and stays unverified; no bound is relaxed.

## Evaluation

**Verdict:** rejected

**Rationale:** The worker screen showed no stable warm-latency reduction across the three precisions. Q4_K had a small mixed-round gain, Q8_0 was slower in all three rounds, and F16 had a mixed-round net loss. Evaluation ended at the screen.

**Conditions:** Apple M2 Ultra, 24 CPU cores, 76 GPU cores and 192 GB unified memory. Release/native/Metal, Python 3.9.6, Metal encoder and an explicitly four-thread CPU decoder. Nine duration-stratified FLEURS Japanese inputs, 6.36 to 28.2 seconds, were measured in three alternating rounds. Each fresh worker warmed the inputs before timing. The primary comparison used the same matched binary with SPEECH_HAAS_TDT_BUFFER_REUSE=0 and 1.

**Source:** speech.cpp commit df540e59c4805571b5dcc46d308893d9fc1a0c0a; ggml commit e1a9ac12edbb57402618fd6936598603f5ee6266; runner 54eed08312c70151ecf3cd872e83f09579be6605.

Positive reductions mean faster. Per-side latency, throughput and memory figures are in the separate local artifact directory `outputs/haas-mac-campaign/017/evaluation.html`, which is not tracked alongside these notes.

### Profile reductions

| Profile | Paired reduction | Fixed-anchor reduction |
| --- | --- | --- |
| Q4_K | +0.265% | +0.429% |
| Q8_0 | -0.757% | -0.996% |
| F16 | -0.113% | -1.029% |

### Round distribution

| Profile | Round reductions in order |
| --- | --- |
| Q4_K | -0.111%, +1.467%, +0.265% |
| Q8_0 | -1.111%, -0.963%, -0.268% |
| F16 | -0.254%, +0.471%, -0.324% |

### Stage diagnostics (decoder only)

| Profile | Decoder-stage reduction |
| --- | --- |
| Q4_K | +22.772% |
| Q8_0 | +3.653% |
| F16 | +19.534% |

### Stage diagnostics (core only)

| Profile | Core-stage reduction |
| --- | --- |
| Q4_K | +3.385% |
| Q8_0 | +0.810% |
| F16 | +4.713% |

### Checks

| Name | Status | Detail |
| --- | --- | --- |
| Dedicated Graph and TDT checks | passed | 2478 of 2478 assertions passed in 14 configurations. The CPU1, CPU4 and Metal graph-only checks accounted for 39 assertions and the controlled synthetic cases for 423. The eight real-weight configurations accounted for 2,016 on Q4_K, Q8_0, F16 and F32 with CPU4 and Metal. |
| KDA state, predictions and full joint logits | exact | Every inspected value was finite and bit-identical between off and on across 16 successive inputs per real-weight configuration. Reused outputs retained their pointers after first allocation and replaced poisoned contents. The controlled fixture exercised cancel, blank, duration, maximum-symbol and recovery paths. |
| Raw TDT differential paths | exact | 528 off/on path pairs matched raw token IDs, frame indices, durations, decoder graph counts and callback sequences. Each of 11 official encoder projections was an input to the C++ decoder in eight model/backend configurations and six plain/boosted and callback-limit combinations. Controlled cases separately exercised a callback that stopped decoding. |
| Official F32 encoder reference | passed | The unchanged encoder check ran 11 distinct inputs on CPU and Metal each, for 22 executions. Final encoded output had minimum SNR 104.4 dB and maximum absolute error 0.000187 on CPU, and 56.4 dB and 0.0533 on Metal. The existing thresholds were 90 dB on CPU and 25 dB on Metal. |
| Original, candidate-off and candidate-on public output | exact | 108 CLI JSON records from 36 invocations produced identical files across all three products for each backend and option, with 9 model metadata records also matching across products. Three clips exercised plain timestamps and keyword guard enabled and disabled for Q4_K, Q8_0 and F16 with a four-thread CPU decoder, plus an F16 device decoder. The selected keywords gave the same text for both guard settings. |
| F16 API and worker contracts | passed | Both switch settings passed API cancellation from another thread, requested at 0.025 s and observed stopped at 0.051 s, then completed a fresh request. The worker passed collecting, waiting and running cancellation, one terminal response per request, keywords and recovery. API threads were automatic and worker threads were explicitly four. |
| Worker quality and stage records | exact | For each precision, both same-binary and fixed-anchor comparisons matched all 9 timestamped quality results including text and public token/segment text and times, and all 27 latency transcripts. The 9 stage rows per side matched clip, run, decode device, threads, token count, decoder graph count and text. |
| Matched build and input provenance | verified | The final candidate matched all 74 common generated compiler flag files and 74 link files of the original after normalizing absolute source and build paths. 96 preflight SHA-256 checks succeeded. The three WAV wrappers contained exactly the source PCM16 samples. |

### Notes

- Paired reduction uses the median of candidate/baseline input ratios after taking each input's median over three rounds. Warm p50, p90 and p99 summarize the nine input medians. RTFx divides total input duration by summed request time. Fresh-process load and first-request medians, and the maximum process peak memory footprint across measured workers, remain separate.
- The decoder and core columns report the median of nine paired stage reductions from three clips and three runs. The stage executable timed its own graph and decode paths separately from the worker's PCM-to-terminal response.
- The profile labels describe the whole GGUF. Q4_K and Q8_0 each contain 21 Q8_0 and 22 F32 decoder tensors. F16 contains 21 F16 and 22 F32 decoder tensors, and all 43 are F32 in the F32 check.
- Public behavioral comparisons reused the unchanged original built for 014 from base 114c514. The fixed timing anchor used a historical build with implicit SDK selection. The primary switch comparison used one binary throughout.
- The first candidate build omitted an explicit arm64 compiler flag. It was retained outside the measurements. The matched rebuild set CMAKE_OSX_ARCHITECTURES=arm64 and the same SDK as the original.
- The first ASR CLI command included the worker-only --no-warmup flag and returned usage status 2 before model inference. Its original files were retained. All validated CLI runs omitted that flag and used the same audio, settings and thresholds.

### Limitations

- The 9-clip screen gives a limited estimate of tail latency. The full corpus and additional timing rounds were not run.
- The changed KDA/TDT stage has no official end-to-end decoder tensor reference in this evaluation. Raw IDs, frames and durations were checked through the dedicated C++ differential path. The worker exposes public text, tokens and segment times. Callback-sequence equality on the 11 projections alone does not prove that every optional cutoff fired.
- The trial did not exercise Windows Vulkan or non-Haas models with matching stage dumps. Direct shared-Graph cases covered the exercised CPU and Metal paths.
- Process peak footprint includes the entire worker. The trial did not measure allocator-call counts, a full memory breakdown or isolated VRAM. Fresh workers retain the operating system's file and Metal caches.
