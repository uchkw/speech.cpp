# CPU-only KDA constant preparation (default off)

**Verdict:** rejected

**Rationale:** Q4_K improved slightly in the paired worker screen, Q8_0 regressed, and F16 was close to unchanged. Round gains changed sign for each precision and no model type beat the fixed anchor in its aggregate paired comparison. Evaluation ended at the screen.

**Conditions:** Apple M2 Ultra, 24 CPU cores, 76 GPU cores, 192 GB unified memory, Python 3.9.6 and a Release/native/Metal arm64 build. Nine duration-stratified Japanese FLEURS recordings were measured in three alternating rounds per side. The encoder ran on Metal; the worker explicitly selected four CPU prediction threads for Q4_K, Q8_0 and F16. Each fresh worker warmed the nine inputs before timing.

**Source:** speech.cpp commit dd21a2f5b62ad32c31227ccfc7e214b7a45b457c; ggml commit e1a9ac12edbb57402618fd6936598603f5ee6266; runner 54eed08312c70151ecf3cd872e83f09579be6605.

Positive reductions mean faster. Per-side latency, throughput and memory figures are in the separate local artifact directory `outputs/haas-mac-campaign/020b/evaluation.html`, which is not tracked alongside these notes.

### Profile reductions

| Profile | Paired reduction | Fixed-anchor reduction |
| --- | --- | --- |
| Q4_K | +1.076% | -0.153% |
| Q8_0 | -0.589% | -0.562% |
| F16 | +0.257% | -1.705% |

### Round distribution

| Profile | Round reductions in order |
| --- | --- |
| Q4_K | +0.946%, +1.938%, -0.252% |
| Q8_0 | +3.732%, -5.499%, -1.054% |
| F16 | -9.021%, +0.706%, +0.065% |

### Stage diagnostics (decoder only)

| Profile | Decoder-stage reduction |
| --- | --- |
| Q4_K | +17.280% |
| Q8_0 | -3.140% |
| F16 | +9.066% |

### Stage diagnostics (core only)

| Profile | Core-stage reduction |
| --- | --- |
| Q4_K | +2.084% |
| Q8_0 | -1.708% |
| F16 | +1.686% |

### Checks

| Name | Status | Detail |
| --- | --- | --- |
| CPU and Metal differential | passed | All 1,911 same-backend assertions passed in 11 configurations. Synthetic F32 on CPU1, CPU4 and MTL0 contributed 186, 186 and 163 assertions. Each of Q4_K, Q8_0, F16 and F32 passed 180 CPU4 assertions and 164 MTL0 assertions. Each configuration kept both 17-step histories, giving 374 paired history steps across the matrix. |
| CPU cache and original Metal graph | passed | CPU reported one preparation, 40 F32 payload bytes and a 64-byte constant-cache allocation. Metal kept one EXP and one NEG per layer and reported zero preparations, zero payload and zero allocation. The real owner lifecycle tested Device/CPU/Device/CPU and both choices after an unset load. Dedicated generated-input TDT paths, progress, boosting and subsequent decode were checked. |
| Official F32 encoder | passed | Eleven distinct reference inputs ran on CPU and Metal (22 executions). The existing check reports subsampling and per-layer differences and gates the final encoded output. Minimum output SNR was 104.4 dB on CPU and 56.4 dB on Metal against thresholds of 90 and 25 dB. Maximum final absolute differences were 0.000187 and 0.0533. |
| Recorded public requests | passed | Across three distinct FLEURS recordings, 36 CLI invocations produced 108 outputs across original/off/on. Full public JSON matched with timestamps on plain requests and valid keywords with guard enabled and disabled on separate requests. The three file types used CPU4 decoding with a Metal encoder; F16 also exercised device decoding. Nine public metadata records matched. |
| Cancellation and recovery | passed | Both off/on F16 C API runs using automatic load threads stopped a live request without text and recovered on the next request. Both off/on F16 CPU4 worker runs cancelled a running request, recovered on the next request and kept one terminal message per request. Actual stdout contains the recovery message for all four runs. |
| Screen consistency | passed | Per model type and comparison, nine quality inputs matched in public text, token and segment arrays; all 27 timed text outputs matched. Each of nine stage rows matched on text, emitted token count and graph count. Same-binary off/on and fixed-anchor comparisons were checked independently. |
| Build and input identity | passed | All 110 preflight identities matched. Both actual arm64 compiles and links used the same absolute SDK. All 74 common compiler-flag files and 74 common link recipes matched the 014 original after only normalizing build and source directory prefixes. The two extra files belong to the dedicated check target. |

### Notes

- For each file, the candidate’s median latency over three rounds is divided by the baseline’s median. The paired warm reduction equals one minus the median of those nine ratios, expressed as a percentage. Round gains pair files within the same round. The main comparison uses one executable and changes the model-load selector from 0 to 1. RTFx divides summed audio duration by summed request time.
- Each worker starts with --no-warmup. The load figure covers process launch through ready and excludes any prediction setup. The first request is the same 28.2-second recording in every fresh worker, measured before the rest of warmup. Load and first request are medians of the three latency workers. Peak memory is the maximum process peak footprint across those three workers and their separate quality worker. The total process peak is distinct from the 40-byte CPU constant payload and its 64-byte allocation.
- The stage tool uses a separate normal Recognizer decoder call, a hand-built encoder graph and one unmeasured warm-up per recording. Its three inputs and three runs give nine keyed pairs. Stage reductions are medians of those nine individual ratios. They describe decoder/core time inside that harness; the primary warm metric includes public worker transport through its terminal response.
- Q4_K had a small net worker improvement and two positive rounds. Q8_0 had a net loss and two negative rounds. F16 had a slight net improvement with a mixed distribution. All three aggregate fixed-anchor comparisons were negative. The separate three-recording decoder stage improved for Q4_K and F16 and slowed for Q8_0.
- The retained original leaves its architecture cache variable blank, while its actual commands explicitly compile and link for arm64. The candidate supplies CMAKE_OSX_ARCHITECTURES=arm64 directly. The initial local validation expected both cache text values to be identical; its record is retained alongside the successful command-level comparison. The fixed campaign anchor is a separate source/build, with implicit SDK selection; it is reported separately from the same-binary comparison.

### Limitations

- Full-corpus timing, full-corpus recognition accuracy and a separate short-only repeat were not run. The measured quality comparison covers the nine public screen files. Keyword contract/output gates used three recordings; keyword-boost latency and live ASR latency were not measured.
- The ordinary worker does not expose raw TDT IDs, frame indices, duration decisions or individual KDA tensors. Its published token/segment arrays are checked separately from the dedicated C++ comparison. Official KDA, recurrent-state and complete TDT references were unavailable. The available official encoder reference does not independently validate the changed prediction stage.
- The M2 Ultra mixed Metal-encoder/CPU4 workload is the measured primary configuration. Other Macs, Windows Vulkan and end-to-end GPU-decoder performance were not evaluated. The selector stays off by default.

Identifiers: `uchkw/speech.cpp` source commit `dd21a2f`; `ggml` `e1a9ac12`.

Implementation: the default-off `SPEECH_HAAS_KDA_CONSTANTS` flag/model gate is captured once at model load. The `KdaLayers` constructor is the sole owner of the constant-cache decision for its prediction network's life: an enabled request with a null backend still throws the existing `logic_error`, and only the null-safe `ggml_backend_is_cpu()` of the pinned `ggml-cpu.h` may enable the cache, so the actual CPU prediction backend is the only site that retains the same F32 EXP-then-NEG output. Per-model ownership and the lazy CPU threadpool lifecycle are unchanged, and metadata-only and pure-load paths prepare nothing.

Non-CPU predictions keep the original per-layer EXP then NEG nodes, and the constant cache's own allocation stays at zero bytes there because no setup graph is ever built for it. On a GPU-loaded model the standard warmup differs by file default: quantized-default files additionally warm the CPU decoder and can therefore prepare once during that warmup, while F16-default files warm only the device, so the first requested CPU prediction can prepare later.

Check design: `fastconformer-constants-check` keeps the full original trajectory comparison on every backend and branches by backend. CPU runs carry the positive contract — one preparation, the expected F32 `[1,heads]` values and allocation, no per-layer EXP/NEG, the same derivation/retry and per-model ownership — while non-CPU runs are a negative control: one EXP and one NEG per layer like the original, exactly zero preparations, a zero-byte constant-cache allocation, a null snapshot backend and empty tensors after work.

Two independent 17-step histories cover prediction, state and logit equality. The lifecycle check decodes the selected owners in the order Device, CPU, Device, CPU, expecting 0/1/0/1 preparations on a device-loaded model and 1/1/1/1 on a CPU-loaded one, with plain/default-off owners always zero; the unset-switch control decodes both choices.
