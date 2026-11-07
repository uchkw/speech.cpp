# Haas MLX resident timing pilot

A small, experiment-only pilot that times the public Japanese Haas MLX port
against the unmodified speech.cpp F16 resident worker on one Apple M2 Ultra
machine. Both underlying runtimes stay unchanged apart from one reviewed,
source-only change on the MLX side of this branch: the resident applies the
fixed FP16-parameter profile described below. The original BF16-parameter
resident is preserved unchanged on the separate BF16 branch; nothing here is
a benchmark framework, and the pilot makes no accuracy, CER or acceptance
claims. Raw string equality between the two products' texts is recorded,
nothing more.

Files:

- `resident.py` — persistent Haas MLX resident worker (JSON Lines protocol).
- `precision.py` — the one shared strict BF16→FP16 cast helper, the
  standalone untimed numeric-audit CLI, and the audit-admission checks.
- `pilot.py` — the paired timing driver (one user command).
- `check_pilot.py` — synthetic checks for the resident, driver and cast
  helper (standard library only; the one child they spawn is a local fixture,
  not an experiment run).
- `check_precision.py` — numerical and admission checks for `precision.py`
  (NumPy fixtures; no MLX, subprocess, model or file dataset).

## Resident worker

Usage: `python3.12 resident.py --model MODEL_DIR` inside the pinned
native-arm64 MLX venv (Python 3.12.11 with MLX 0.32.4 and mlx-metal 0.32.4,
numpy and SentencePiece, as installed for the upstream model).

Protocol, one request at a time, flushed JSON Lines on stdout, details and
failures on stderr:

    worker -> {"type": "ready", ...}
    driver -> {"type": "chunk", "id": "0", "seq": 0, "pcm": "<base64 S16LE>"}
    driver -> {"type": "transcribe", "id": "0", "sample_rate": 16000, "timestamps": false}
    worker -> {"type": "end", "id": "0", "stop": "complete", "text": "..."}

Behavior:

- The model (`haas_mlx.model.HaasModel.load`) and
  `SentencePieceProcessor(tokenizer.model)` load exactly once; after the load
  and before the ready message the reviewed one-time BF16→FP16 parameter cast
  is applied through the shared strict helper in `precision.py` (it calls
  `set_dtype(mx.float16)` and validates stable names, shapes, every leaf
  dtype and the pinned totals; no BF16 master copy is retained), and the cast
  parameters are evaluated (`mx.eval(model.parameters())`).
- Startup requires a real Metal GPU device, and performs no warmup request, no
  download, no compile pass, and no precision change other than that reviewed
  parameter cast.
- The ready message reports the actual identity: runtime name, the installed
  `mlx` and `mlx-metal` distribution versions (`importlib.metadata.version`),
  the `mx.default_device()` type, `mx.device_info()['device_name']`, and the
  Python executable, version and machine. It never claims speech.cpp protocol
  4 and never resolves CPU threads. `ready.model.precision` carries the
  honest cast profile with the pinned totals (731 parameter tensors,
  615871520 elements); the driver rejects an old BF16 ready without it and
  any forged profile (wrong dtype, counts or a master-copy claim).
- Each request converts the complete S16LE PCM
  (`numpy.frombuffer(..., dtype='<i2').astype(np.float32) / 32768.0`, then
  `mx.array`) and calls `haas_mlx.longform.transcribe_text(model, waveform,
  sp)` exactly once; that function's native quiet-point chunking, per-chunk
  cache clearing and final detokenization are preserved untouched. There is no
  token, timestamp, segment or second-pass path.
- Validation: exact request id, consecutive `seq` starting at 0, strict base64,
  nonempty even-length PCM, `sample_rate` 16000, `timestamps` false, and no
  unsupported fields. One request at a time; successive requests reuse the
  process; request buffers are released after each terminal result.
- Exit codes: 0 clean EOF while idle, 2 invalid traffic, 3 model load or
  transcription failure, 4 EOF with an open (abandoned) request.

`ResidentSession` is a pure standard-library protocol boundary around a
supplied callable, so it can be exercised without MLX, numpy or audio.

## FP16 parameter profile and precision audit

This branch runs exactly one reviewed precision profile. The resident casts
the pinned all-BF16 parameter tree to MLX float16 exactly once, after
`HaasModel.load` and before the ready message, through the one shared strict
helper in `precision.py` (`cast_parameters_to_float16`), then evaluates the
cast parameters. The helper flattens the tree by stable dot names with
`mlx.utils.tree_flatten`, requires every leaf to be `bfloat16` before and
`float16` after `set_dtype(mx.float16)`, requires unchanged names and shapes,
refuses anything but the pinned model totals (731 parameter tensors,
615871520 elements) and reports an honest profile (`master_dtype: null`, no
BF16 master copy is retained; operator structure, the native quiet-point
chunking, per-chunk cache policy and decode policy are untouched, while the
numerical precision of operator arithmetic follows the actual dtypes of its
inputs — never a claim that all arithmetic is unchanged or that all
arithmetic is float16). There is no alternate code path when validation
fails:
the resident exits 3 and the pilot never starts. Quantization, compilation
and cache behavior are untouched.

The separate untimed numeric audit (never run inside the pilot, never on a
timed startup, and isolated from its request, first-request and load
measurements) compares the stored values of every parameter tensor before
and after the shared cast, offline and one tensor at a time:

    . $HOME/Dropbox/bin/conda-init.sh
    . $HOME/venv/py3.11/mlx/bin/activate
    /path/to/mlx-py3.12-venv/bin/python3.12 -B precision.py \
        --mlx-source PORT_CHECKOUT \
        --mlx-model-dir MODEL_DIR \
        --mlx-model-manifest MODEL_MANIFEST.JSON \
        --out FRESH_AUDIT_DIRECTORY

It writes `FRESH_AUDIT_DIRECTORY/audit.json` (fresh output only, exit 0)
after re-verifying the port checkout and the three-file model manifest with
the pilot's own checks, verifying that the imported `haas_mlx` really comes
from the requested checkout `src`, and confirming the pinned MLX/Python/device
identity. Source words are widened exactly (`array.view(mx.uint16)`, NumPy
uint32 `<< 16`, then `.view(np.float32)`); destinations are widened with
`np.asarray(tensor).astype(np.float32)`, which preserves observed FP16
subnormals. The artifact records per-tensor and total counts of numerically
unequal values (signed zeros equal), underflow to signed or unsigned zero,
kept FP16 subnormals below 2^-14, finite-to-infinity overflow, nonfinite
values on either side, and the maximum finite absolute difference computed in
float64 — names, shapes and counts only, never values, tensor data or
personal paths. Any nonfinite source/destination value or any overflow
records a diagnostic and exits nonzero (3): such a cast is never eligible.
Underflow, kept subnormals and changed-but-finite values are reported as
counts and never affect eligibility; the audit is numeric evidence only,
independent of ASR, and claims no accuracy, CER or timing result.

The driver admits the artifact once, before any child process starts, via
`--mlx-precision-audit FRESH_AUDIT_DIRECTORY/audit.json`: the schema, the
completed eligible status, the profile, the pinned totals and dtype claims,
the per-tensor accounting (uniquely named well-formed rows whose totals
reproduce every declared count and max), the exact source/model identities
the pilot just verified itself, the pinned
runtime, and the SHA-256 of the current `resident.py`/`precision.py` must all
match, so a bad, partial, stale or missing audit is refused (and the
artifact's SHA-256 is registered in `run.json`).

## Pilot driver

One shell, one sequence: source the two MLX activation scripts, then launch
the wrapper (the controller is `/usr/bin/python3`, system Python 3.9 with the
standard library plus the parent campaign helpers `worker.py` / `compare.py`).
The MLX child runs the separate pinned native-arm64 MLX 3.12.11 venv passed
as `--mlx-python`; its executable spelling, typically a venv symlink, is
recorded verbatim and never resolved:

    . $HOME/Dropbox/bin/conda-init.sh
    . $HOME/venv/py3.11/mlx/bin/activate
    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock /usr/bin/python3 -B pilot.py \
        --native-build BUILD.JSON \
        --assets ASSETS.JSON \
        --mlx-python /path/to/mlx-py3.12-venv/bin/python3.12 \
        --mlx-source PORT_CHECKOUT \
        --mlx-model-dir MODEL_DIR \
        --mlx-model-manifest MODEL_MANIFEST.JSON \
        --mlx-precision-audit FRESH_AUDIT_DIRECTORY/audit.json \
        --out FRESH_DIRECTORY

Locking: run the driver under the existing shared campaign lock
(`/usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock ...`). The driver
itself never takes and never nests that lock; if a wrapper already holds it,
the driver simply runs inside it.

Verification, all once and before any timing or process startup:

- Native side: the build manifest must pass `worker.require_evaluation_build`
  (a PGO training-only build is refused), be source-verified, pin the accepted
  baseline commits (source `114c514430c8aa1455c077819f8e72e674a48b75`, ggml
  `e1a9ac12edbb57402618fd6936598603f5ee6266`), and its registered speech
  binary must match its recorded SHA-256 (the campaign's unused stage tool is
  not verified here). The assets must register exactly one f16
  model row carrying the accepted verified SHA
  `f752e02f68c1dfff795900e40aa82e5302976fb5360080fcbd330d8b3c992cd8`, and the
  file must match it.
- MLX side: the port checkout must be clean (no model code additions) and
  pinned at `cde96d263f56aad75c1de70282f71a91562092dd`; the model manifest must
  carry revision `15ef4f11618c6a1ee50e0eafaae13beded8ecdf9` and exactly the
  three files `model.safetensors` (published SHA
  `7f207ce23fee14a9c113381777391f52bdaa1b401a8131149b39727c2a278ebb`),
  `tokenizer.model`
  (`9bd7bcdff62e43200001cd4c258d22e475b7b3b1e0f64f8be8e6e046f402f38c`) and
  `config.json`, each with size and verified SHA-256 matching the files on
  disk. `config.json` is additionally rejected unless its actual Git blob
  SHA-1 equals the public pin `638a6704459b3041d8122c08ae872f90464edd49`, so
  a manifest rewritten to agree with a modified config does not pass. The
  supplied `--mlx-precision-audit` artifact is admitted against exactly these
  verified identities, the pinned runtime, the pinned 731/615871520 totals
  and the current `resident.py`/`precision.py` digests, before any child
  process starts; a bad, partial, stale or missing audit is refused.
- Inputs: the fixed nine of the thirty stratified subset rows at indexes
  `[0, 4, 9, 10, 14, 19, 20, 24, 29]` (exactly three per original stratum).
  Each row must carry a positive integer `samples`, finite `seconds` equal
  to `samples/16000`, and a PCM file of exactly `2 * samples` even
  bytes matching its stored SHA-256; duplicate names and incomplete rows are
  refused. Only those nine PCM files are read or hashed; the wider corpus is
  never touched, and model bytes are never hashed inside a request.
- Environment: `worker.execution_environment` with no recorded knobs (any
  inherited, unrecorded `GGML_`/`SPEECH_` variable is a refusal) and no `MLX_`
  variables at all; the effective allowed knob set is empty and recorded.

Measurement:

- Native child argv:
  `speech worker F16 --device gpu --threads 4 --no-warmup --decode-on-cpu`;
  its ready must report `protocol` 4, `model.device` `MTL0`, `model.threads`
  integer 4 and `defaults.decode_on_cpu` true — the requested F16 Metal
  encoder / four-CPU-thread TDT decode placement. The MLX child is
  `python3.12 resident.py --model MODEL_DIR` with `PYTHONPATH` set only to the
  pinned port `src`, `HF_HUB_OFFLINE=1` and no `.pyc`; its ready must report
  the pilot-private `haas-mlx-pilot-1` protocol (never 4 or threads), MLX
  0.32.4 with mlx-metal 0.32.4, Python 3.12.11 arm64, a GPU default device
  and `device_name` `Apple M2 Ultra`, plus the fixed FP16 parameter profile
  with the pinned totals in `ready.model.precision` (an old BF16 ready or a
  forged profile is rejected). Unexpected placement or versions are
  rejected before the first warm request.
- A fresh process per side and round. Each child is wrapped in
  `/usr/bin/time -l -o memory.log` (isolated from stderr), its stdout carries
  only the JSON Lines protocol, and stderr goes to `stderr.log`. Pipe reading
  uses a queue plus a thread with timeouts, and children are closed, waited
  for and killed safely on every failure path. One wall-clock watchdog bounds
  each whole group (children start in their own session, wrapper included): a
  worker that stops reading its stdin is killed by it, the blocked write is
  released, and the round is marked timed out even when the surfaced error is
  a BrokenPipeError.
- One shared outer per-request function for both products: read and hash-check
  the PCM before the clock starts; start `time.perf_counter()` before the
  first base64/JSON encoding and send; send chunks of up to exactly 32768
  bytes, each flushed; send `transcribe` with `sample_rate` 16000 and
  `timestamps: false`; ignore only valid native progress; verify the terminal
  `end`/id/`stop: complete`/text; stop the clock after the terminal read,
  including all caller JSON parsing. Nothing else happens in the timed path.
- Three rounds in AB/BA/AB side order (`mlx` is A), both sides measuring the
  same clip order within a round, rotated by `round * 7` as the campaign does.
  Nine warmups (longest first) then nine measured requests per side and round:
  27 results per side, 54 timed rows. Spawn-to-ready (`load_seconds`) and the
  first request (the first warmup) are recorded separately from the warm and
  measured populations.
- Per-side aggregation (recorded with a `latency_scope` label): the primary
  `latency` summary runs the shared `latency()` helper over nine distinct
  per-recording medians — each the median of that input's three measured
  repetitions, which the summary requires exactly — with warm interpolation
  across those nine values and RTFx as the sum of the nine distinct audio
  durations divided by the sum of the nine median latencies. `measured_count`
  stays 27 and the former pooled 27-request summary is kept unchanged as the
  separate `pooled_measured_latency`; the immutable old BF16-branch summary
  pooled a different 27-row population, so its numbers are not comparable
  with either field. This is aggregation of already-timed rows only; no
  inference or upstream path changes.
- The output directory must not exist; it is never overwritten. On a side
  failure a `partial.json` keeps the ready message, the active request id,
  every completed warm and measured row, the error and the timeout state next
  to `stderr.log` and `memory.log`; a `failure.json` with the error and
  traceback is kept alongside all worker diagnostics, and no summary is
  written, so partial runs never look successful.

Output layout:

    OUT/run.json                     manifest: pins, digests, conditions, argv/env,
                                      the admitted precision audit (SHA-256 + summary)
    OUT/mlx/round-N/                 result.json, stderr.log, memory.log
    OUT/native/round-N/              result.json, stderr.log, memory.log
    OUT/*/round-N/partial.json       only when that side/round failed
    OUT/summary.json                 per-side primary latency over nine per-recording
                                     medians (p50/p90/p99/RTFx) plus the separate
                                     pooled 27-request latency, per-input medians,
                                     spawn-to-ready, first vs warm, peak memory,
                                     paired raw-text difference counts and median
                                     ratios (mlx/native)
    OUT/paired-text-changes.json     the raw text pairs that differ, per request
    OUT/failure.json                 only on failure

Each `result.json` stores the raw warmup and measured rows, the actual argv
and environment notes, the full ready message, artifact identities and the
measurement conditions.

## Checks

`python3 check_pilot.py` runs a concise synthetic suite under the system
Python 3.9 with no downloads, MLX package, real audio or large data: resident
protocol acceptance and rejections (bad seq/id/PCM/options, abandoned and
clean EOF, successive requests), a fake-module resident identity check (the
dummy `mlx.metal` has no `get_device_name`, so a wrong API name fails), the
strict shared cast helper on a small symbolic BF16 tree (honest counts and
profile, refusal of non-BF16 leaves, partial/reshaping/renaming casts, empty
trees and trees that are not the pinned model), ready placement,
pinned-version and FP16-profile rejection for both products, timer boundaries
around the shared request function, exact 32768-byte chunking, input-row
  coherence, the enforced config Git blob pin, timeout/EOF/error/nonzero-exit
  cleanup with partial-row retention, AB/BA/AB round order, the 27-per-side
  paired counts and ratios, and the per-side primary latency over nine
  per-recording medians with the pooled 27 kept separate (an outlier fixture
  shows a raw-27 primary would be wrong, and a 27-row maldistribution is
  refused). The only subprocess is a tiny stdlib-only Python
child, spawned as a fixture, that emits a synthetic ready and then refuses to
read a longer-than-pipe PCM so the watchdog kill is exercised for real.

`/path/to/mlx-py3.12-venv/bin/python3.12 -B check_precision.py` runs the
numerical and admission checks (the numerical ones skip cleanly where NumPy
is absent): signed zeros and exact widened values (65280 is exact in both
BF16 and float16), the kept FP16 subnormal 2^-24, the BF16 2^-25 underflow to
signed/unsigned zero, overflow at 65536 — which intentionally reports both
`overflow_to_infinity` and `nonfinite_after` — and nonfinite values stopping
the audit with diagnostics and ineligibility, the float64 finite maximum
difference, and the audit-admission rejections (bad/partial/missing
artifacts, wrong or absent dtype and count claims, disqualifying numeric
evidence marked eligible, corrupted per-tensor rows — an empty row, a
duplicate name, a local count outside its tensor, declared totals
inconsistent with the rows — stale runner SHA or stale source/model/runtime
identities, unverified module path).

This documentation contains no measurements and claims no results; no run
exists yet on either branch. The pilot never runs the full 650-input corpus
automatically — the fixed nine-input screen above is all it does — and a
full 650 would be considered only if this FP16-parameter profile later shows
a speed gain; none has been run or measured for this profile, and there is
no accuracy or time evidence for it.
