# 023: worker standard-input settings

This is an unbuilt, unmeasured source checkpoint from `114c514430c8aa1455c077819f8e72e674a48b75`,
with ggml `e1a9ac12edbb57402618fd6936598603f5ee6266`. Neither setting is an established latency improvement.
The earlier 002b residual of about 42–50% leaves input transport unmeasured; it does not identify its cause.

`SPEECH_HAAS_WORKER_UNSYNC=1` calls `std::ios::sync_with_stdio(false)`.
`SPEECH_HAAS_WORKER_UNTIE=1` calls `std::cin.tie(nullptr)`.
Both accept only `0` or `1`, default to off when absent, and are read once before model loading and before
the worker starts its input thread. Invalid values produce the existing fatal error envelope before `ready`.
Both values are validated before either setting changes. The switches affect every model served by `worker`,
not only Haas. Library requests, `asr`, `tts`, and `serve` do not read them.

The command reads stdin only with `std::cin`. `tools/cli/main.cpp` still sets Windows stdin and stdout to
binary. `take_stdout()` still duplicates the protocol descriptor and redirects descriptor 1 to stderr.
`Protocol::line()` still writes and flushes its separate `FILE *`; no protocol output uses `std::cout`.
No parser, base64, PCM chunk, resampling, terminal-message, cancel, retry, default-option, or API code changes.
Windows behavior has only been inspected in source, not executed.

This branch has no added clocks or trace writes. The separate diagnostic branch
`codex/diag-haas-023-worker-transport` measures client encoding, pipe operations, reader CPU and wall time,
JSON parsing, base64 decoding, PCM append, and queue/worker boundaries. It writes traces after draining.
Do not use the diagnostic binary or its driver to accept a speed-up.

## Validation still required

All C++ builds, worker runs, model checks, Metal and Windows runs, and measurements remain unexecuted.
The source checkpoint is explicitly prepared before the campaign's compute slot becomes available.

1. Build the original and this branch separately under the shared performance lock, with `-j4`.
2. Run the small diagnostic first, separating reader CPU from time waiting for input. Compare original,
   diagnostic with tracing off, and diagnostic with tracing on before interpreting its durations.
3. Check all four settings (`00`, `01`, `10`, `11`) through the existing worker smoke scripts. Include invalid
   environment values, interleaved chunks, collecting/waiting/running cancels, immediate ID reuse,
   malformed JSON/base64, odd PCM bytes, wrong sequence numbers, EOF while collecting, ready/protocol 4,
   and exactly one terminal message. Cover recognition, synthesis, and enhancement, since stdin is shared.
4. Compare exact recognition text, tokens, times, and errors for Haas Q4_K, Q8_0, and F16 on CPU4 and Metal,
   plain and keyword boost. Exercise the other recognition family and the C API checks required by the project.
5. Use the frozen campaign runner with these two switches explicitly recorded. Screen a candidate suggested
   by the diagnosis for three alternating rounds, separate cold/load/warm latency and memory, and use the
   normal acceptance gates. Do not combine parser, base64, chunk-size, or resampler changes.
6. Execute the worker checks on Windows, including binary PCM transport and protocol stdout purity.

The two settings are hypotheses for an experiment, not a permanent design decision or a production recommendation.
