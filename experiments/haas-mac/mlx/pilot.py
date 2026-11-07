#!/usr/bin/env python3
"""Paired timing pilot: the public Haas MLX port against the unmodified speech.cpp F16 resident.

Both runtimes stay unchanged apart from one reviewed source-only change on
the MLX side: the resident applies the fixed FP16-parameter profile (the one
shared strict cast helper in precision.py, once after load and before ready).
This driver only verifies artifacts, spawns and times them; it also admits
exactly one finished precision-audit artifact (--mlx-precision-audit) against
the source/model identities it verified itself and the current runner
digests, before any child process starts, and never reruns the audit on a
timed startup. Everything is verified once, before any process starts or
clock runs. The timed path is one shared per-request function per product
that only verifies the prepared PCM (before its clock starts), base64-encodes,
sends, reads and parses protocol messages. Ready messages are checked for the
exact placement, pinned versions and the FP16 parameter profile before the
first warm request, and each worker group runs under one wall-clock watchdog
that also releases a blocked write.

Three rounds run in AB/BA/AB side order over the same fixed nine stratified
inputs, with nine warmups (longest first) and then nine measured requests per
side and round: 27 results per side, 54 timed rows. Output goes to a fresh
directory only; failures keep their diagnostics and never look successful.

Run under the shared campaign wrapper lock; this driver never takes or nests
the lock itself:

    /usr/bin/lockf -k /private/tmp/speech-haas-mac-perf.lock /usr/bin/python3 -B pilot.py \
        --native-build BUILD.json --assets ASSETS.json --mlx-python PYTHON3.12 \
        --mlx-source PORT_CHECKOUT --mlx-model-dir MODEL_DIR \
        --mlx-model-manifest MODEL_MANIFEST.json \
        --mlx-precision-audit AUDIT_DIRECTORY/audit.json --out FRESH_DIRECTORY
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import json
import math
import os
import platform
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The campaign helpers are checked in one level above this experiment directory.
PARENT = HERE.parent

sys.dont_write_bytecode = True
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(HERE))
from worker import execution_environment, require_evaluation_build
from compare import latency, sha

import precision  # noqa: E402  (shared FP16 cast pins and audit admission; stdlib only)

NATIVE_SOURCE_COMMIT = '114c514430c8aa1455c077819f8e72e674a48b75'
NATIVE_GGML_COMMIT = 'e1a9ac12edbb57402618fd6936598603f5ee6266'
NATIVE_F16_SHA256 = 'f752e02f68c1dfff795900e40aa82e5302976fb5360080fcbd330d8b3c992cd8'
MLX_SOURCE_COMMIT = 'cde96d263f56aad75c1de70282f71a91562092dd'
MLX_MODEL_REVISION = '15ef4f11618c6a1ee50e0eafaae13beded8ecdf9'
MLX_WEIGHTS_SHA256 = '7f207ce23fee14a9c113381777391f52bdaa1b401a8131149b39727c2a278ebb'
MLX_TOKENIZER_SHA256 = '9bd7bcdff62e43200001cd4c258d22e475b7b3b1e0f64f8be8e6e046f402f38c'
MLX_CONFIG_GIT_BLOB = '638a6704459b3041d8122c08ae872f90464edd49'
MLX_PROTOCOL = 'haas-mlx-pilot-1'
MLX_VERSION = '0.32.4'
MLX_METAL_VERSION = '0.32.4'
MLX_PYTHON_VERSION = '3.12.11'
MLX_MACHINE = 'arm64'
MLX_DEVICE_NAME = 'Apple M2 Ultra'
MLX_MODEL_FILES = ('model.safetensors', 'tokenizer.model', 'config.json')
SUBSET_INDEXES = [0, 4, 9, 10, 14, 19, 20, 24, 29]
SUBSET_SIZE = 30
MEASURED_PER_ROUND = 9
WARMUPS_PER_ROUND = 9
STRATA = ('6-10s', '10-20s', '20-30s')
SIDES = ('mlx', 'native')
ROUNDS = 3
SHIFT_PER_ROUND = 7
CHUNK_BYTES = 32768
RECEIVE_TIMEOUT = 300.0
EXIT_TIMEOUT = 30.0
# One generous bound on a whole worker group: model load plus 18 requests.
WATCHDOG_SECONDS = 5400.0
# Recorded beside the per-side latency numbers so the population cannot be
# misread: the primary summary aggregates nine per-recording medians, and the
# pooled 27-request summary is a separate, explicitly secondary population.
LATENCY_SCOPE_NOTE = (
    'primary latency: the latency() helper over nine distinct per-recording medians, each the '
    'median of that input\'s three measured repetitions; percentiles use warm interpolation '
    'across the nine values and RTFx is the sum of the nine distinct audio durations divided '
    'by the sum of the nine median latencies; pooled_measured_latency is the unchanged pooled '
    'summary over all 27 raw measured requests (count 27), and the immutable old BF16-branch '
    'summary pooled a different 27-row population, so its numbers are not comparable')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def round_order(round_index):
    """Side order per round: AB/BA/AB with SIDES = ('mlx', 'native')."""
    shift = round_index % len(SIDES)
    return SIDES[shift:] + SIDES[:shift]


def shifted(records, round_index):
    """The campaign clip order: rotate the measured list by round * 7."""
    shift = (round_index * SHIFT_PER_ROUND) % len(records)
    return records[shift:] + records[:shift]


def verify_native(build, assets):
    """Accept only the pinned, source-verified baseline build and f16 model row."""
    require_evaluation_build(build)
    if not build.get('source_verified'):
        raise ValueError('The native build manifest is not source verified.')
    if build.get('source_commit') != NATIVE_SOURCE_COMMIT or build.get('ggml_commit') != NATIVE_GGML_COMMIT:
        raise ValueError('The native build must pin the accepted baseline source and ggml commits.')
    rows = [m for m in assets.get('models', []) if isinstance(m, dict) and m.get('type') == 'f16']
    if len(rows) != 1:
        raise ValueError('The assets must register exactly one f16 model row.')
    model = rows[0]
    if model.get('sha256') != NATIVE_F16_SHA256 or model.get('verified_sha256') != NATIVE_F16_SHA256:
        raise ValueError('The f16 model row must carry the accepted verified SHA.')
    entry = build.get('speech')
    if not isinstance(entry, dict) or 'path' not in entry or 'sha256' not in entry:
        raise ValueError('The registered speech binary entry is incomplete.')
    if sha(entry['path']) != entry['sha256']:
        raise ValueError('The registered speech binary changed after the build manifest was written.')
    if sha(model['path']) != model['sha256']:
        raise ValueError('The f16 model changed after the assets manifest was written.')
    return model


def verify_mlx_source(source):
    """The port checkout must be clean and pinned; no model code additions."""
    if git(source, 'status', '--porcelain'):
        raise ValueError('The MLX port worktree must be clean; no model code additions are allowed.')
    commit = git(source, 'rev-parse', 'HEAD')
    if commit != MLX_SOURCE_COMMIT:
        raise ValueError(f'The MLX port must be checked out at {MLX_SOURCE_COMMIT}; found {commit}.')
    return {'commit': commit, 'tree': git(source, 'rev-parse', 'HEAD^{tree}'), 'worktree_clean': True}


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b'blob %d\x00' % len(data) + data).hexdigest()


def verify_mlx_model(manifest_path, model_dir):
    """Verify the pinned three-file MLX model manifest once, before any startup."""
    manifest = read(manifest_path)
    if manifest.get('revision') != MLX_MODEL_REVISION:
        raise ValueError(f'The MLX model manifest revision must be {MLX_MODEL_REVISION}.')
    files = manifest.get('files')
    if not isinstance(files, dict) or set(files) != set(MLX_MODEL_FILES):
        raise ValueError(f'The MLX model manifest must list exactly {sorted(MLX_MODEL_FILES)}.')
    for name in MLX_MODEL_FILES:
        entry = files[name]
        if (not isinstance(entry, dict) or not isinstance(entry.get('size'), int) or entry['size'] <= 0
                or not isinstance(entry.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', entry['sha256'])
                or entry.get('verified') is not True):
            raise ValueError(f'The manifest entry for {name} is incomplete or unverified.')
    if files['model.safetensors']['sha256'] != MLX_WEIGHTS_SHA256:
        raise ValueError('The model weights must match the published pinned SHA.')
    if files['tokenizer.model']['sha256'] != MLX_TOKENIZER_SHA256:
        raise ValueError('The tokenizer must match the published pinned SHA.')
    digests = {}
    for name in MLX_MODEL_FILES:
        path = Path(model_dir) / name
        if path.stat().st_size != files[name]['size']:
            raise ValueError(f'{name} does not have the manifest size.')
        actual = sha(path)
        if actual != files[name]['sha256']:
            raise ValueError(f'{name} does not match the manifest sha256.')
        digests[name] = {'sha256': actual, 'size': files[name]['size']}
    # The manifest SHA is rewriteable evidence; the public Git blob is not, so a
    # config modified after pinning is rejected even when the manifest agrees.
    blob = git_blob_sha1(Path(model_dir) / 'config.json')
    if blob != MLX_CONFIG_GIT_BLOB:
        raise ValueError(f'config.json Git blob is {blob}; the pinned config is {MLX_CONFIG_GIT_BLOB}.')
    digests['config.json']['git_blob_sha1'] = blob
    return {'revision': manifest['revision'], 'files': digests, 'manifest_sha256': sha(manifest_path)}


def select_inputs(assets):
    """The fixed nine of the thirty stratified rows; three per original stratum."""
    subset = assets.get('subset')
    if not isinstance(subset, list) or len(subset) != SUBSET_SIZE:
        raise ValueError(f'The assets subset must supply all {SUBSET_SIZE} stratified rows.')
    chosen = [subset[index] for index in SUBSET_INDEXES]
    if len({row['file'] for row in chosen}) != MEASURED_PER_ROUND:
        raise ValueError('The selected inputs contain duplicate file names.')
    required = ['file', 'stratum', 'samples', 'seconds', 'pcm_path', 'pcm_sha256']
    for row in chosen:
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"Incomplete subset row {row.get('file')!r}: missing {missing}.")
    strata = {}
    for row in chosen:
        strata.setdefault(row['stratum'], []).append(row['file'])
    if set(strata) != set(STRATA) or sorted(len(names) for names in strata.values()) != [3, 3, 3]:
        raise ValueError('The fixed indexes must select exactly three rows of each original stratum.')
    for row in chosen:
        validate_input_row(row)
        if sha(row['pcm_path']) != row['pcm_sha256']:
            raise ValueError(f"PCM hash mismatch: {row['file']}")
    return chosen


def validate_input_row(row):
    """Reject a timing row whose metadata disagrees with its PCM bytes."""
    samples = row['samples']
    if type(samples) is not int or samples <= 0:
        raise ValueError(f"Row {row['file']!r} needs a positive integer sample count.")
    seconds = row['seconds']
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
            or not math.isfinite(seconds)):
        raise ValueError(f"Row {row['file']!r} needs finite duration seconds.")
    if seconds != samples / 16000.0:
        raise ValueError(f"Row {row['file']!r} seconds disagree with samples/16000.")
    size = Path(row['pcm_path']).stat().st_size
    if size % 2 or size != 2 * samples:
        raise ValueError(f"Row {row['file']!r} PCM must be exactly two bytes per sample.")


def native_environment():
    """Full child environment; refuses inherited GGML_/SPEECH_ knobs unrecorded.

    The pilot records no knobs, so the effective allowed set is empty on both
    sides and any inherited experiment variable is a hard refusal."""
    return execution_environment({})


def require_no_mlx_tweaks():
    leaks = sorted(key for key in os.environ if key.startswith('MLX_'))
    if leaks:
        raise ValueError(f'Unrecorded MLX_ environment tweaks must be unset: {leaks}')


def mlx_environment(source):
    env = {key: value for key, value in os.environ.items() if not key.startswith('MLX_')}
    env['PYTHONPATH'] = str(Path(source) / 'src')  # only the pinned port src
    env['HF_HUB_OFFLINE'] = '1'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def native_command(build, model):
    return [build['speech']['path'], 'worker', model['path'],
            '--device', 'gpu', '--threads', '4', '--no-warmup', '--decode-on-cpu']


def mlx_command(mlx_python, model_dir):
    # Keep the operator-supplied executable spelling (often a venv symlink)
    # unexpanded in the launched argv and every recorded manifest.
    return [str(mlx_python), str(HERE / 'resident.py'), '--model', str(model_dir)]


def verify_native_ready(ready):
    """The speech.cpp worker must confirm the requested placement before warmup.

    The pilot requests the F16 model with the encoder on the Metal device and
    the TDT decode on four CPU threads, so MTL0, threads 4 and decode_on_cpu
    true are all load-bearing."""
    if ready.get('type') != 'ready' or ready.get('protocol') != 4:
        raise RuntimeError(f'Native worker did not become ready with protocol 4: {ready}')
    model = ready.get('model')
    if not isinstance(model, dict) or model.get('device') != 'MTL0':
        raise RuntimeError(f'Native worker ready.model.device must be MTL0: {ready}')
    resolved = model.get('threads')
    if type(resolved) is not int or resolved != 4:
        raise RuntimeError(f'Native worker ready.model.threads is {resolved!r}; expected 4.')
    if not isinstance(ready.get('defaults'), dict) \
            or ready['defaults'].get('decode_on_cpu') is not True:
        raise RuntimeError(f'Native worker ready.defaults.decode_on_cpu must be true: {ready}')


def verify_mlx_runtime(ready):
    """The MLX side must report the pinned pilot identity, never the native one."""
    if (ready.get('type') != 'ready' or ready.get('runtime') != 'haas-mlx'
            or ready.get('protocol') != MLX_PROTOCOL):
        raise RuntimeError(f'MLX resident did not report the pilot protocol and runtime: {ready}')
    if ready.get('model', {}).get('threads') is not None:
        raise RuntimeError(f'MLX resident must not claim the speech.cpp thread resolution: {ready}')
    mlx = ready.get('mlx')
    if not isinstance(mlx, dict):
        raise RuntimeError(f'MLX resident ready lacks its MLX identity: {ready}')
    if mlx.get('version') != MLX_VERSION or mlx.get('mlx_metal_version') != MLX_METAL_VERSION:
        raise RuntimeError(
            f'MLX resident must run MLX {MLX_VERSION} and mlx-metal {MLX_METAL_VERSION}: {ready}')
    if 'gpu' not in str(mlx.get('device', '')).lower():
        raise RuntimeError(f'MLX resident default device is not a GPU: {ready}')
    if mlx.get('device_name') != MLX_DEVICE_NAME:
        raise RuntimeError(f'MLX resident device_name must be {MLX_DEVICE_NAME!r}: {ready}')
    python_identity = ready.get('python')
    if (not isinstance(python_identity, dict) or not python_identity.get('executable')
            or python_identity.get('version') != MLX_PYTHON_VERSION
            or python_identity.get('machine') != MLX_MACHINE):
        raise RuntimeError(
            f'MLX resident must run Python {MLX_PYTHON_VERSION} on {MLX_MACHINE}: {ready}')


def expected_mlx_precision():
    """The one fixed FP16 parameter profile every MLX ready message must carry."""
    return {'profile': precision.PRECISION_PROFILE, 'cast': True,
            'source_dtype': precision.SOURCE_PARAMETER_DTYPE,
            'parameter_dtype': precision.CAST_PARAMETER_DTYPE, 'master_dtype': None,
            'arithmetic': precision.ARITHMETIC_NOTE,
            'tensor_count': precision.PINNED_PARAMETER_TENSORS,
            'element_count': precision.PINNED_PARAMETER_ELEMENTS}


def verify_mlx_ready(ready):
    """verify_mlx_runtime plus the fixed FP16 parameter profile.

    An old BF16 resident (no model.precision) and a stale or forged profile
    (wrong dtype, master-copy claim or pinned totals) are both rejected
    before the first warm request; there is no alternate acceptance path."""
    verify_mlx_runtime(ready)
    reported = ready.get('model', {}).get('precision')
    if not isinstance(reported, dict):
        raise RuntimeError(f'MLX resident ready must carry model.precision: {ready}')
    for key, value in expected_mlx_precision().items():
        if reported.get(key) != value:
            raise RuntimeError(f'MLX resident precision.{key} must be {value!r}, '
                               f'got {reported.get(key)!r}: {ready}')
    sample = reported.get('shape_sample')
    if (not isinstance(sample, list) or not sample or len(sample) > precision.SHAPE_SAMPLE_SIZE
            or not all(isinstance(row, list) and len(row) == 2 and isinstance(row[0], str)
                       and isinstance(row[1], list)
                       and all(type(dim) is int and dim > 0 for dim in row[1])
                       for row in sample)):
        raise RuntimeError(f'MLX resident precision.shape_sample must be small honest '
                           f'shape metadata: {sample!r}')


def default_spawn(command, env, memory_log, stderr_log):
    """Launch a worker with Darwin /usr/bin/time -l isolated into memory_log."""
    return subprocess.Popen(['/usr/bin/time', '-l', '-o', str(memory_log), *command],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr_log,
                            text=True, encoding='utf-8', start_new_session=True, env=env)


def shutdown_process(process):
    """Terminate a live child process group, escalating to SIGKILL."""
    if process.poll() is not None:
        return
    with contextlib.suppress(OSError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def parse_memory_log(text):
    memory = {label.replace(' ', '_'): int(value) for value, label in re.findall(
        r'^\s*(\d+)\s+(maximum resident set size|peak memory footprint)\s*$', text, re.MULTILINE)}
    if 'peak_memory_footprint' not in memory:
        raise RuntimeError(f'Missing peak memory footprint in {memory}.')
    return memory


def read_prepared_pcm(record):
    """Read and hash-check one input; called before that request's clock starts."""
    pcm = Path(record['pcm_path']).read_bytes()
    if hashlib.sha256(pcm).hexdigest() != record['pcm_sha256']:
        raise ValueError(f"Input changed: {record['file']}")
    return pcm


def run_side_round(side, round_index, directory, command, env, verify_ready, identities,
                   records, warmup, conditions=None, environment_note=None, *,
                   spawn=default_spawn, clock=time.perf_counter, shutdown=shutdown_process,
                   read_pcm=read_prepared_pcm, receive_timeout=RECEIVE_TIMEOUT,
                   watchdog_seconds=WATCHDOG_SECONDS, chunk_bytes=CHUNK_BYTES):
    """One fresh worker process: ready, warmups longest first, then measurements.

    recognize() below is the one shared outer per-request function used for
    both products; only verify_ready and the spawned command differ. One
    wall-clock watchdog bounds the whole group, because a worker that stops
    reading its stdin blocks write/flush in a way message timeouts cannot
    release; any failure keeps its completed rows in partial.json."""
    directory.mkdir(parents=True, exist_ok=False)
    memory_log = directory / 'memory.log'
    messages = queue.Queue()
    started = datetime.now(timezone.utc).isoformat()
    with (directory / 'stderr.log').open('w') as log:
        start = clock()
        process = spawn(command, env, memory_log, log)
        expired = threading.Event()

        # killpg reaches only this child's own session group, which includes
        # the /usr/bin/time wrapper; closing the group releases a blocked write.
        def expire():
            expired.set()
            with contextlib.suppress(Exception):
                shutdown(process)

        timer = threading.Timer(watchdog_seconds, expire)
        timer.daemon = True
        timer.start()

        def read_stdout():
            try:
                for line in process.stdout:
                    messages.put(json.loads(line))
                messages.put(RuntimeError('Worker stdout closed before the next message.'))
            except Exception as error:
                messages.put(error)
            finally:
                # Reached only after the iteration ended (EOF or shutdown), so
                # this thread is no longer reading the pipe it closes here.
                with contextlib.suppress(OSError, ValueError):
                    process.stdout.close()

        reader = threading.Thread(target=read_stdout, daemon=True)
        reader.start()

        def receive():
            try:
                message = messages.get(timeout=receive_timeout)
            except queue.Empty:
                raise RuntimeError('Timed out waiting for a worker message.')
            if isinstance(message, Exception):
                raise message
            if not isinstance(message, dict):
                raise RuntimeError(f'Malformed worker message: {message!r}')
            return message

        def send(message):
            process.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
            process.stdin.flush()

        def recognize(record, request_id):
            pcm = read_pcm(record)  # verified before the clock starts
            begin = clock()         # before the first base64/JSON encoding and send
            for seq, offset in enumerate(range(0, len(pcm), chunk_bytes)):
                send({'type': 'chunk', 'id': request_id, 'seq': seq,
                      'pcm': base64.b64encode(pcm[offset:offset + chunk_bytes]).decode()})
            send({'type': 'transcribe', 'id': request_id, 'sample_rate': 16000, 'timestamps': False})
            while True:
                result = receive()
                if result.get('id') != request_id:
                    raise RuntimeError(f'Unexpected request id: {result}')
                if result.get('type') == 'progress':
                    continue  # valid native progress only; the MLX resident never sends it
                if (result.get('type') != 'end' or result.get('stop') != 'complete'
                        or not isinstance(result.get('text'), str)):
                    raise RuntimeError(f'Recognition failed: {result}')
                text = result['text']
                break
            return {'file': record['file'], 'stratum': record['stratum'],
                    'audio_seconds': record['seconds'], 'seconds': clock() - begin,
                    'text': text, 'request_id': request_id}

        ready, load_seconds = None, None
        warmed, measured = [], []
        active_request_id = None
        try:
            ready = receive()
            load_seconds = clock() - start
            verify_ready(ready)
            ordered = sorted(warmup, key=lambda record: (-record['seconds'], record['file']))
            for index, record in enumerate(ordered):
                active_request_id = f'warm-{index}'
                warmed.append(recognize(record, active_request_id))
            for index, record in enumerate(shifted(records, round_index)):
                active_request_id = str(index)
                measured.append(recognize(record, active_request_id))
            active_request_id = None
            process.stdin.close()
            if process.wait(timeout=EXIT_TIMEOUT) != 0:
                raise RuntimeError('Worker exited with failure.')
        except Exception as failure:
            if expired.is_set():
                failure = RuntimeError(f'The worker group outlived the {watchdog_seconds:g}s '
                                       'round watchdog and was killed by it.')
            write(directory / 'partial.json',
                  {'status': 'failed', 'side': side, 'round': round_index,
                   'error': f'{type(failure).__name__}: {failure}', 'timed_out': expired.is_set(),
                   'ready': ready, 'load_seconds': load_seconds,
                   'active_request_id': active_request_id,
                   'warmups_completed': warmed, 'measured_completed': measured,
                   'diagnostics': {'stderr_log': 'stderr.log', 'memory_log': 'memory.log'}})
            raise failure
        finally:
            timer.cancel()
            # Kill and wait first: the dead group delivers the stdout EOF that
            # ends the reader and unblocks a write stuck in the pipe.
            shutdown(process)
            timer.join()
            reader.join()  # the reader saw that EOF and closed stdout itself
            if not process.stdin.closed:
                with contextlib.suppress(OSError, ValueError):
                    process.stdin.close()
    result = {'started_utc': started, 'side': side, 'round': round_index, 'conditions': conditions,
              'command': command, 'spawn_prefix': ['/usr/bin/time', '-l', '-o', str(memory_log)],
              'environment': environment_note, 'ready': ready, 'identities': identities,
              'load_seconds': load_seconds, 'first_request': warmed[0],
              'warmups': warmed, 'warmup_count': len(warmed),
              'measured': measured, 'memory': parse_memory_log(memory_log.read_text())}
    write(directory / 'result.json', result)
    return result


def execute_rounds(run):
    """run(side, round_index) -> result dict; sides alternate AB/BA/AB."""
    runs = {side: [] for side in SIDES}
    for round_index in range(ROUNDS):
        for side in round_order(round_index):
            runs[side].append(run(side, round_index))
    return runs


def side_summary(runs):
    rows = [row for run in runs for row in run['measured']]
    if len(runs) != ROUNDS or len(rows) != ROUNDS * MEASURED_PER_ROUND:
        raise ValueError(f'Each side needs {ROUNDS} runs of {MEASURED_PER_ROUND} measured requests.')
    by_file = {}
    for row in rows:
        by_file.setdefault(row['file'], []).append(row)
    if len(by_file) != MEASURED_PER_ROUND or any(len(group) != ROUNDS for group in by_file.values()):
        raise ValueError(f'The primary latency needs each of the {MEASURED_PER_ROUND} inputs '
                         f'measured in exactly {ROUNDS} repetitions.')
    per_input = [{'file': file, 'stratum': group[0]['stratum'],
                  'audio_seconds': group[0]['audio_seconds'],
                  'median_seconds': statistics.median(row['seconds'] for row in group)}
                 for file, group in sorted(by_file.items())]
    primary_rows = [{'audio_seconds': row['audio_seconds'], 'seconds': row['median_seconds']}
                    for row in per_input]
    warm_seconds = [row['seconds'] for run in runs for row in run['warmups']]
    return {'latency': latency(primary_rows),
            'latency_scope': LATENCY_SCOPE_NOTE,
            'measured_count': len(rows),
            'pooled_measured_latency': latency(rows),
            'per_input_medians': per_input,
            'rounds': [latency(run['measured']) for run in runs],
            'first_request_ms': statistics.median(run['first_request']['seconds'] for run in runs) * 1000,
            'warmup_median_ms': statistics.median(warm_seconds) * 1000,
            'load_ms': statistics.median(run['load_seconds'] for run in runs) * 1000,
            'peak_memory_bytes': max(run['memory']['peak_memory_footprint'] for run in runs),
            'repeated_text_changes': sum(len({row['text'] for row in group}) > 1
                                         for group in by_file.values())}


def paired_summary(runs):
    """Pair every request by round and file; raw text equality only, never CER."""
    measured = {side: {run['round']: run['measured'] for run in runs[side]} for side in SIDES}
    if set(measured['mlx']) != set(measured['native']) or set(measured['mlx']) != set(range(ROUNDS)):
        raise ValueError('Paired comparison requires every round on both sides.')
    counts = {'requests': 0, 'text_equal': 0, 'text_changed': 0}
    changes, request_ratios, per_file = [], [], {}
    for round_index in sorted(measured['mlx']):
        left, right = measured['mlx'][round_index], measured['native'][round_index]
        if len(left) != len(right):
            raise ValueError('Paired rounds must contain the same number of requests.')
        for mlx_row, native_row in zip(left, right):
            if mlx_row['file'] != native_row['file']:
                raise ValueError('Paired requests must name the same input file.')
            if mlx_row['seconds'] <= 0 or native_row['seconds'] <= 0:
                raise ValueError('Nonpositive request timing cannot be paired.')
            counts['requests'] += 1
            if mlx_row['text'] == native_row['text']:
                counts['text_equal'] += 1
            else:
                counts['text_changed'] += 1
                changes.append({'round': round_index, 'file': mlx_row['file'],
                                'request_id': mlx_row['request_id'],
                                'mlx_text': mlx_row['text'], 'native_text': native_row['text']})
            request_ratios.append(mlx_row['seconds'] / native_row['seconds'])
            per_file.setdefault(mlx_row['file'], {'mlx': [], 'native': []})
            per_file[mlx_row['file']]['mlx'].append(mlx_row['seconds'])
            per_file[mlx_row['file']]['native'].append(native_row['seconds'])
    per_input_ratio = {file: statistics.median(values['mlx']) / statistics.median(values['native'])
                       for file, values in per_file.items()}
    paired = {**counts,
              'median_latency_ratio': statistics.median(per_input_ratio.values()),
              'per_request_ratio_median': statistics.median(request_ratios),
              'per_input_median_ratio': per_input_ratio,
              'ratio_direction': 'mlx/native; below 1.0 means the MLX port was faster',
              'text_comparison': 'raw string equality only; this pilot claims no CER or semantic equivalence'}
    return paired, changes


def summarize(directory, manifest, runs):
    """Write the paired summary; only called after every run succeeded."""
    sides = {side: side_summary(runs[side]) for side in SIDES}
    paired, changes = paired_summary(runs)
    summary = {'status': 'measured',
               'claim': 'Raw paired timing pilot only; no accuracy, CER or acceptance claim.',
               'conditions': manifest['conditions'], 'pins': manifest['pins'],
               'sides': sides, 'paired': paired,
               'raw_results': [str(Path(side) / ('round-%d' % round_index) / 'result.json')
                               for side in SIDES for round_index in range(ROUNDS)]}
    write(directory / 'paired-text-changes.json', changes)
    write(directory / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Paired timing pilot: Haas MLX port vs the speech.cpp F16 resident.')
    parser.add_argument('--native-build', required=True, help='campaign build.json of the speech binary')
    parser.add_argument('--assets', required=True, help='campaign assets.json with the subset and models')
    parser.add_argument('--mlx-python', required=True, help='Python with MLX for the resident child')
    parser.add_argument('--mlx-source', required=True, help='pinned Haas MLX port Git checkout')
    parser.add_argument('--mlx-model-dir', required=True, help='local converted MLX model directory')
    parser.add_argument('--mlx-model-manifest', required=True, help='verified pinned model file manifest')
    parser.add_argument('--mlx-precision-audit', required=True,
                        help='admitted precision.py audit artifact for the FP16 parameter profile')
    parser.add_argument('--out', required=True, help='fresh output directory (never overwritten)')
    args = parser.parse_args(argv)

    for label, value, kind in [('native-build', args.native_build, 'file'),
                               ('assets', args.assets, 'file'),
                               ('mlx-python', args.mlx_python, 'file'),
                               ('mlx-source', args.mlx_source, 'directory'),
                               ('mlx-model-dir', args.mlx_model_dir, 'directory'),
                               ('mlx-model-manifest', args.mlx_model_manifest, 'file'),
                               ('mlx-precision-audit', args.mlx_precision_audit, 'file')]:
        path = Path(value)
        if not (path.is_file() if kind == 'file' else path.is_dir()):
            raise SystemExit(f'pilot: --{label} must be an existing {kind}: {value}')

    build = read(args.native_build)
    assets = read(args.assets)
    native_model = verify_native(build, assets)
    mlx_source_id = verify_mlx_source(args.mlx_source)
    mlx_model_id = verify_mlx_model(args.mlx_model_manifest, args.mlx_model_dir)
    # The audit is admitted once, against the identities just verified above
    # and the current resident/precision sources, before any child starts.
    audit_admission = precision.verify_audit_artifact(
        read(args.mlx_precision_audit),
        {'port': mlx_source_id, 'model': mlx_model_id,
         'runner': {'resident_sha256': sha(HERE / 'resident.py'),
                    'precision_sha256': sha(HERE / 'precision.py')},
         'runtime': {'mlx': {'version': MLX_VERSION, 'mlx_metal_version': MLX_METAL_VERSION,
                             'device_name': MLX_DEVICE_NAME},
                     'python': {'version': MLX_PYTHON_VERSION, 'machine': MLX_MACHINE}}})
    audit_sha = sha(args.mlx_precision_audit)
    inputs = select_inputs(assets)
    native_env = native_environment()
    require_no_mlx_tweaks()
    mlx_env = mlx_environment(args.mlx_source)
    commands = {'native': native_command(build, native_model),
                'mlx': mlx_command(args.mlx_python, args.mlx_model_dir)}
    verify_ready = {'native': verify_native_ready, 'mlx': verify_mlx_ready}
    identities = {
        'native': {'build': {key: build[key] for key in
                             ['source_commit', 'source_tree', 'ggml_commit'] if key in build},
                   'speech_sha256': build['speech']['sha256'],
                   'model': {'type': 'f16', 'sha256': native_model['sha256']}},
        'mlx': {'port': mlx_source_id, 'model': mlx_model_id,
                'resident_sha256': sha(HERE / 'resident.py'),
                'precision': audit_admission},
    }
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'paired timing pilot: nine inputs, three rounds; the native speech.cpp F16 '
                 'resident is unmodified, while the MLX resident applies the reviewed one-time '
                 'BF16->FP16 parameter cast after load (operator structure, chunking and decode '
                 'policy unchanged)',
        'conditions': {
            'rounds': ROUNDS,
            'side_order_per_round': [list(round_order(r)) for r in range(ROUNDS)],
            'warmups_per_side_round': WARMUPS_PER_ROUND, 'warmup_policy': 'longest first',
            'measured_per_side_round': MEASURED_PER_ROUND,
            'measured_per_side': ROUNDS * MEASURED_PER_ROUND,
            'timed_rows_total': ROUNDS * MEASURED_PER_ROUND * len(SIDES),
            'clip_order_shift_per_round': SHIFT_PER_ROUND,
            'chunk_bytes': CHUNK_BYTES, 'sample_rate': 16000, 'timestamps': False,
            'round_watchdog_seconds': WATCHDOG_SECONDS,
            'native_threads': 4, 'native_extra_args': ['--no-warmup', '--decode-on-cpu'],
            'mlx_precision': {
                'profile': precision.PRECISION_PROFILE,
                'source_dtype': precision.SOURCE_PARAMETER_DTYPE,
                'parameter_dtype': precision.CAST_PARAMETER_DTYPE, 'master_dtype': None,
                'tensor_count': precision.PINNED_PARAMETER_TENSORS,
                'element_count': precision.PINNED_PARAMETER_ELEMENTS,
                'cast': 'once per resident, after load and before ready; never inside requests',
                'audit': 'admitted artifact only; never rerun on timed startup',
            },
            'lock': 'run under the shared lockf wrapper lock; the pilot takes no lock itself',
        },
        'subset_indexes': SUBSET_INDEXES, 'subset_size': SUBSET_SIZE,
        'inputs': [{key: row[key] for key in
                    ['file', 'stratum', 'samples', 'seconds', 'pcm_path', 'pcm_sha256']}
                   for row in inputs],
        'pins': {'native_source_commit': NATIVE_SOURCE_COMMIT, 'native_ggml_commit': NATIVE_GGML_COMMIT,
                 'native_f16_sha256': NATIVE_F16_SHA256, 'mlx_source_commit': MLX_SOURCE_COMMIT,
                 'mlx_model_revision': MLX_MODEL_REVISION, 'mlx_weights_sha256': MLX_WEIGHTS_SHA256,
                 'mlx_tokenizer_sha256': MLX_TOKENIZER_SHA256, 'mlx_config_git_blob': MLX_CONFIG_GIT_BLOB,
                 'mlx_precision_profile': precision.PRECISION_PROFILE,
                 'mlx_precision_audit_schema': precision.AUDIT_SCHEMA,
                 'mlx_parameter_tensors': precision.PINNED_PARAMETER_TENSORS,
                 'mlx_parameter_elements': precision.PINNED_PARAMETER_ELEMENTS},
        'identities': identities,
        'precision_audit': {'file': Path(args.mlx_precision_audit).name, 'sha256': audit_sha,
                            'admission': audit_admission},
        'assets': {'path': str(Path(args.assets).resolve()), 'sha256': sha(args.assets)},
        'environment': {'effective_allowed_knobs': {'native': {}, 'mlx': {}},
                        'native_child': 'os.environ via campaign worker.execution_environment, no recorded knobs',
                        'mlx_child': {'base': 'os.environ without MLX_* keys',
                                      'set': {'PYTHONPATH': str(Path(args.mlx_source) / 'src'),
                                              'HF_HUB_OFFLINE': '1',
                                              'PYTHONDONTWRITEBYTECODE': '1'}}},
        'python': {'controller': platform.python_version(), 'executable': sys.executable,
                   'mlx_child': str(args.mlx_python)},
        'runner_sha256': {'pilot.py': sha(HERE / 'pilot.py'), 'resident.py': sha(HERE / 'resident.py'),
                          'precision.py': sha(HERE / 'precision.py'),
                          'worker.py': sha(PARENT / 'worker.py'), 'compare.py': sha(PARENT / 'compare.py')},
    }

    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=False)  # fresh output only, no overwrites
    write(directory / 'run.json', manifest)
    print(f'verified artifacts and inputs; timing into {directory}', flush=True)
    try:
        def run(side, round_index):
            result = run_side_round(
                side, round_index, directory / side / ('round-%d' % round_index), commands[side],
                native_env if side == 'native' else mlx_env, verify_ready[side],
                {'side': side, **identities[side]}, inputs, inputs,
                conditions=manifest['conditions'], environment_note=manifest['environment'])
            print(f"{side} round {round_index}: {len(result['measured'])} measured requests", flush=True)
            return result

        runs = execute_rounds(run)
        summarize(directory, manifest, runs)
    except Exception as error:
        write(directory / 'failure.json', {'status': 'failed', 'error': str(error),
                                           'traceback': traceback.format_exc(),
                                           'created_utc': datetime.now(timezone.utc).isoformat(),
                                           'partial_results': sorted(str(path.relative_to(directory))
                                                                     for name in ('result.json', 'partial.json')
                                                                     for path in directory.glob(f'*/*/{name}'))})
        raise
    print(directory / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
