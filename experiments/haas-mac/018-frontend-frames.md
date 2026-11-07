# Parallelize the whole-input NeMo frontend by frame ranges

Based on speech.cpp `114c514430c8aa1455c077819f8e72e674a48b75` and ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`. The trial source is `09194919834fc486fdb5b461014933dcf37c70ce`.

Pre-emphasis runs sequentially over the whole input into one float32 buffer, and each worker computes a disjoint contiguous frame interval with the existing `log_mel`. One range runs on the caller and the remaining workers - 1 ranges run on `std::async(std::launch::async)` tasks. Futures are joined before the buffers are destroyed and `get()` propagates worker failures. Each range owns its windowed, spectrum and power scratch and shares only the read-only window, filterbank and FFT basis. Per-feature normalization stays sequential, reducing all frames in a fixed order through mean, unbiased variance, guard and float32 conversion.

`log_mels` selects `max(1, min(thread_limit, 4, frames / 128))` workers, capping at four. The 128 frames is a minimum per range only when two or more workers run; inputs below 128 frames still run on the single caller, and the split to multiple workers starts at 256 frames. Effective counts are source-derived without task tracing.

`SPEECH_HAAS_FRONTEND_PARALLEL=0|1` is read only at KDA FastConformer recognizer construction and defaults to off. `0` keeps it off, `1` passes the recognizer's existing thread preference capped at four, and any other value in that scope is rejected. Factory selector behavior is covered by separate runtime checks.

The common stage check compares candidate serial, two-worker and four-worker features, requiring bit-identical, finite raw float64 log-mel and final float32 values. Inputs include silence, edge impulses, low-amplitude alternating samples and deterministic noise. Sizes cover the minimum valid length, one sample on each side of hop and grain boundaries, frame counts not dividing evenly among workers, and 2, 4 and 30 s workloads. Short inputs must keep the serial error, fault and audio attribution. Concurrent shared-frontend calls must match isolated calls, and incremental `log_mel` over random pieces with discarded old samples must match the whole input. A dump root adds comparison of four-worker features to official features under the existing 90 dB bound, reported as unchecked when absent.

## Decision and measurement

Keep the switch off by default. F16 with a Metal encoder and four CPU decoder threads improved on all 30 timed inputs. Q4_K and Q8_0 each slowed in a duration stratum. The six short prefixes provide limited evidence for short requests.

Apple M2 Ultra, native arm64 Release, Metal encoder and greedy decoder on four CPU threads. A fresh worker warmed every measured input once. The same candidate binary handled the OFF and ON requests in alternating rounds. Positive percentages mean a shorter request. Each input contributes its median over three runs before the cohort percentiles and paired ratios are computed.

## Nine-clip screen

| Weights | Paired reduction | Three round reductions | Faster inputs |
| --- | --- | --- | --- |
| Q4_K | +2.22% | +0.97% / +2.28% / +1.54% | 9 / 9 |
| Q8_0 | -1.01% | +3.34% / +2.42% / -2.44% | 4 / 9 |
| F16 | +2.98% | +2.54% / +2.09% / +3.45% | 8 / 9 |

## Full worker results

| Weights | Paired reduction | Faster inputs | p50 ms OFF / ON | p90 ms OFF / ON | p99 ms OFF / ON | RTFx OFF / ON |
| --- | --- | --- | --- | --- | --- | --- |
| Q4_K | +0.91% | 18 / 30 | 84.53 / 83.00 | 132.22 / 128.53 | 154.84 / 151.35 | 164.24 / 167.60 |
| Q8_0 | +1.03% | 20 / 30 | 81.21 / 83.21 | 128.75 / 126.94 | 149.62 / 146.26 | 171.38 / 172.94 |
| F16 | +2.80% | 30 / 30 | 79.79 / 77.91 | 125.11 / 122.76 | 148.23 / 143.54 | 173.31 / 178.40 |

## Full rounds, duration strata and frontend

| Weights | Round reductions | 6–10 / 10–20 / 20–30 s | Frontend reductions for three stage clips | Direct campaign-anchor reduction |
| --- | --- | --- | --- | --- |
| Q4_K | +0.99% / +1.14% / +1.26% | -1.27% / +2.28% / +3.10% | +50.24% / +59.61% / +58.32% | +2.00% |
| Q8_0 | +2.36% / +0.08% / +1.89% | +2.50% / -1.25% / +1.49% | +54.73% / +57.91% / +58.13% | -0.00% |
| F16 | +1.99% / +2.94% / +2.03% | +3.23% / +2.81% / +1.89% | +54.27% / +57.27% / +58.19% | +2.68% |

## Load and process memory

| Weights | Ready ms OFF / ON | First warm request ms OFF / ON | Peak footprint MiB OFF / ON |
| --- | --- | --- | --- |
| Q4_K | 150.22 / 161.42 | 174.36 / 176.87 | 457.41 / 455.78 |
| Q8_0 | 198.46 / 194.73 | 173.17 / 162.34 | 753.86 / 749.94 |
| F16 | 294.53 / 297.90 | 172.53 / 167.82 | 1311.94 / 1309.66 |

## Public recognition quality

| Weights | Literal micro-CER | Repository micro-CER | Timestamped outputs changed | Timestamp-disabled texts changed |
| --- | --- | --- | --- | --- |
| Q4_K | 2608/34928 (7.4668%) | 1520/32648 (4.6557%) | 0 / 650 | 0 / 90 |
| Q8_0 | 2623/34928 (7.5097%) | 1494/32648 (4.5761%) | 0 / 650 | 0 / 90 |
| F16 | 2621/34928 (7.5040%) | 1498/32648 (4.5883%) | 0 / 650 | 0 / 90 |

## Separate short-prefix supplement

| Weights | 2 s paired reduction | 2 s rounds | 4 s paired reduction | 4 s rounds |
| --- | --- | --- | --- | --- |
| Q4_K | +1.72% | +0.01% / +1.28% / +3.77% | +2.32% | +0.47% / +2.32% / +3.69% |
| Q8_0 | -1.02% | -1.90% / -0.50% / +1.13% | +1.10% | -0.21% / +1.08% / +1.02% |
| F16 | +1.42% | +1.52% / +3.01% / +0.47% | +1.74% | +2.98% / +1.28% / +2.57% |

## Checks

- For each of F32, Q4_K, Q8_0 and F16, all 661 assertions passed. Raw F64 and final F32 features matched byte for byte across serial, two- and four-worker limits, including frame boundaries, shared FFT use and incremental sample origins. The selected implementation passed the existing greater-than-90-dB official feature bound for all eleven input tensors, with 103.3 dB minimum.
- Both legacy F32 serial frontend checks passed all eleven official feature cases. The original checker prints case/frame and SNR on separate lines. An initial parser missed those rows. The preserved successful log was re-read with its actual format, without repeating inference or changing a bound.
- 96 CLI launches covered 288 public requests across the original, unset, OFF and ON states, three weight types, CPU and Metal, and four option sets. All captured JSON fields matched within each condition. Twelve F16 C API and worker checks passed. The six worker runs observed collecting, waiting and running cancellations followed by recovery. The API used its automatic load-thread setting.
- On three public inputs per weight type, all final joint.enc F32 bytes and greedy text matched between the matched original, OFF and ON. The diagnostic was linked separately to each actual runtime.
- Each weight type matched on 650 timestamp-enabled text, public token and segment results and on all 90 timestamp-disabled latency texts. The same-binary OFF comparison and the fixed timing anchor were both checked. A separate prior run of the matched 114c514 original on the same public inputs reproduced the captured outputs. Its elapsed times were excluded from this trial.
- At both prefix lengths, text stayed stable in all three rounds and matched across OFF and ON. Each type had 18 paired text comparisons. Six unlabeled prefixes came from three recordings and have no valid CER.

## Interpreting the measurements

- Each full duration stratum contains ten inputs. The three representative stage clips last 7.98, 14.52 and 24.72 seconds. The host frontend timer covers task creation and joining, pre-emphasis and normalization. The full worker timer measures first PCM chunk preparation through the terminal reply and excludes file reading, model loading and warmup.
- For Q8_0 the cohort p50 increased while the median of matched input ratios improved. These summarize different quantities. The fixed campaign anchor used historical SDK selection. Its direct ratios are separate from the matched OFF/ON comparison.
- The candidate and the matched original have 74 matching compiler-flag files and 74 matching link files after normalization of their source/build directories. The original runtime source is 114c514. The fixed timing anchor is f976ebd8.
- Each short prefix is taken from the start of a longer recording. The frozen sample counts and frame formula imply one frontend worker for both modes at two seconds. At four seconds they imply one for OFF and three for ON. The timed worker exposed no task counter. Two-second variation cannot be attributed to frame parallelism.

## Coverage limits

- The full tests cover public FLEURS inputs from 6.36 to 28.2 seconds. The 30 input medians provide a limited estimate of production tail latency. Three source recordings cannot establish accuracy or typical timing for naturally completed short utterances.
- No official full KDA/TDT, Windows/Vulkan, real non-KDA/Sortformer, injected task-creation or allocation-failure test was run. The worker does not expose raw TDT IDs, frame indices or durations.
- Evidence for this F16 candidate covers a Metal encoder and CPU4 decoder. Default Voimem decoder placement and automatic thread counts require their own evaluation. Peak process footprint includes all worker allocations. The host FFT span does not provide a breakdown of kernel or device memory.
