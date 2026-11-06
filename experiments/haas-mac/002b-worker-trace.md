# Same-request worker timing diagnostic

This experiment locates the load-time FFN expansion gain within an actual Haas
worker request. It adds timers to source `f7b82da`; it is not a worker transport
optimization. The measured clean diagnostic source is
`6f46e4f15b9f65b68ca00ba21e074977c77d14d3`, with ggml
`e1a9ac12edbb57402618fd6936598603f5ee6266`.

Set `SPEECH_HAAS_WORKER_TRACE=1` to emit one JSON object on stderr after each
recognition request. The `haas_worker_trace` field identifies the record, `id`
matches the request, and `points_ms` holds monotonic elapsed milliseconds from
the worker entry point. The regular stdout protocol is unchanged. Set
`SPEECH_HAAS_FFN_F16=0` or `1` separately to choose the original or expanded
weights. The trace switch does not change this choice.

The trace spans PCM conversion, API setup and execution, resampling, engine
frontend/encode/decode/detokenization, encoder graph construction/compute/read/
destruction, terminal construction, locking, writing, flushing and cleanup.
Thread-local markers are verified at runtime: all 36 points appear in all 90
recorded requests, including the nested API and engine points.

## Boundaries

The Python request timer stays unchanged: after PCM file reading and hash
verification, before first base64 encoding, through the parsed and validated
terminal message. Every fresh worker warms all three inputs once, longest
first, before measuring them. The diagnostic uses three alternating effect
rounds for Q4_K and F16, plus three rotating overhead rounds comparing the plain
build, the diagnostic build with tracing off, and tracing on.

Use `finish_before` as the contained worker endpoint. It precedes terminal
writing and is within the Python interval. Producer-side `terminal_flush_after`
and `worker_exit` can occur after the client receives the terminal, so they are
reported separately and never subtracted from Python request time. Clock
epochs from different processes are never subtracted. Trace reservation occurs
before `worker_enter` and remains in the residual.

Nested intervals overlap and must not be summed. Median child intervals need
not sum to their median parent. For an additive decomposition, subtract the
contained interval from each matching request first and then average the
paired differences. The residual includes input preparation, transport,
parsing, queuing and terminal receipt; it is not a measurement of one I/O
operation. Trace serialization happens after the terminal and can affect a
following request, which is why overhead is measured separately.

## Replay

This directory retains the measured driver and interval map byte for byte.
`worker.py` also matches the measured snapshot. Their SHA-256 identities are:

| File | SHA-256 |
|---|---|
| `002b-worker-probe.py` | `08bf960c247bd89f69f92ae6b54ae5d0f97aea6ebddc2740a5c7b10686b6b6c1` |
| `002b-worker-probe-intervals.json` | `e0594adf757ae8f92d65c1bcae17bf6a8467a58b180097547ac6068f126236c0` |
| `worker.py` | `4b97435f62e153d3b4f17cdadf01691c367225b1bbcb0e0572e817997f2a3a7a` |

The external campaign artifact directory retains input and model hashes in
`002b-worker-probe/run.json`, binary/source identities in the two build
manifests, raw traces and analysis. Build manifests are
`builds/002b-ffn-f16/build.json` and `builds/002b-worker-trace/build.json`.
Set `diagnostic_source` to this branch's checkout and `campaign` to the prepared
campaign artifacts. Use a new output directory for a replay:

```sh
python3 -B "$diagnostic_source/experiments/haas-mac/002b-worker-probe.py" prepare \
  --runner "$diagnostic_source/experiments/haas-mac" \
  --assets "$campaign/assets/assets.json" \
  --plain-build "$campaign/builds/002b-ffn-f16/build.json" \
  --trace-build "$campaign/builds/002b-worker-trace/build.json" \
  --intervals "$diagnostic_source/experiments/haas-mac/002b-worker-probe-intervals.json" \
  --out "$new_output"
python3 -B "$new_output/runner-snapshot/002b-worker-probe.py" run \
  --manifest "$new_output/run.json"
python3 -B "$new_output/runner-snapshot/002b-worker-probe.py" analyze \
  --manifest "$new_output/run.json"
```

The driver holds `/private/tmp/speech-haas-mac-perf.lock` separately for each
worker. Experiment variables must not be inherited; the manifest supplies
each variant's values. Builds also hold the lock and use four build jobs.

## Observed result

On November 7, 2026, the M2 Ultra with four CPU decoder threads shows Q4_K
candidate/baseline paired median ratios of 0.90702 for encoder compute, 0.92628
for the engine span including cleanup, and 0.96064 for the request. F16 control
ratios are 0.99992, 1.00257 and 1.01300 respectively. Every paired text matches.
The measured diagnostic-disabled and diagnostic-enabled request ratios versus
the plain build are 1.00999 and 0.99591. This small comparison does not prove
zero instrumentation overhead.

The outside residual is about 22.7, 39.0 and 68.9 ms for the three inputs,
roughly 42-50% of request time. The PCM conversion, API audio copy and same-rate
resampler copy total about 0.05, 0.08 and 0.14 ms. The core gain therefore
survives in the worker and is diluted by a large outside span; no individual
transport operation is identified as its cause.

This three-input diagnostic is supplementary. The primary 30-input worker run
shows only 1.76% improvement versus 1.85% in the unchanged F16 control, with
768 MiB extra weights and a longer load. Trial 002b stays conditional and
unadopted. Official stage-reference and Windows Vulkan validation remain
unavailable. No production inference change is adopted by this checkpoint.
