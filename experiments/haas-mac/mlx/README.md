# Haas MLX resident timing pilot

A small, experiment-only pilot that times the public Japanese Haas MLX port
against the unmodified speech.cpp F16 resident worker on one Apple M2 Ultra
machine. Both runtimes stay unchanged; nothing here is a benchmark framework,
and the pilot makes no accuracy, CER or acceptance claims. Raw string equality
between the two products' texts is recorded, nothing more.

Files:

- `resident.py` — persistent Haas MLX resident worker (JSON Lines protocol).
- `pilot.py` — the paired timing driver (one user command).
- `check_pilot.py` — synthetic checks for both (standard library only; the one
  child they spawn is a local fixture, not an experiment run).

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
  `SentencePieceProcessor(tokenizer.model)` load exactly once; parameters are
  evaluated (`mx.eval(model.parameters())`) before the ready message.
- Startup requires a real Metal GPU device, and performs no warmup request, no
  download, no compile pass, and no weight or precision changes.
- The ready message reports the actual identity: runtime name, the installed
  `mlx` and `mlx-metal` distribution versions (`importlib.metadata.version`),
  the `mx.default_device()` type, `mx.device_info()['device_name']`, and the
  Python executable, version and machine. It never claims speech.cpp protocol
  4 and never resolves CPU threads.
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
  a manifest rewritten to agree with a modified config does not pass.
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
  and `device_name` `Apple M2 Ultra`. Unexpected placement or versions are
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
- The output directory must not exist; it is never overwritten. On a side
  failure a `partial.json` keeps the ready message, the active request id,
  every completed warm and measured row, the error and the timeout state next
  to `stderr.log` and `memory.log`; a `failure.json` with the error and
  traceback is kept alongside all worker diagnostics, and no summary is
  written, so partial runs never look successful.

Output layout:

    OUT/run.json                     manifest: pins, digests, conditions, argv/env
    OUT/mlx/round-N/                 result.json, stderr.log, memory.log
    OUT/native/round-N/              result.json, stderr.log, memory.log
    OUT/*/round-N/partial.json       only when that side/round failed
    OUT/summary.json                 per-side latency (p50/p90/p99/RTFx), per-input
                                     medians, spawn-to-ready, first vs warm, peak
                                     memory, paired raw-text difference counts and
                                     median ratios (mlx/native)
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
dummy `mlx.metal` has no `get_device_name`, so a wrong API name fails), ready
placement and pinned-version rejection for both products, timer boundaries
around the shared request function, exact 32768-byte chunking, input-row
coherence, the enforced config Git blob pin, timeout/EOF/error/nonzero-exit
cleanup with partial-row retention, AB/BA/AB round order and the 27-per-side
paired counts and ratios. The only subprocess is a tiny stdlib-only Python
child, spawned as a fixture, that emits a synthetic ready and then refuses to
read a longer-than-pipe PCM so the watchdog kill is exercised for real.

This documentation contains no measurements and claims no results; no run
exists yet.
