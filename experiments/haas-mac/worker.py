"""Measure one serialized Haas worker run; used by compare.py under the campaign lock."""
import base64
from datetime import datetime, timezone
import json
import hashlib
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import threading
import time

BINARY_EXPERIMENT_ENVIRONMENT = {'SPEECH_HAAS_CPU_REPACK', 'SPEECH_HAAS_FFN_F16', 'SPEECH_HAAS_TDT_RESIDENT_STATE',
                                 'SPEECH_HAAS_TDT_JOINT_REUSE', 'SPEECH_HAAS_TDT_CPU_PLAN', 'SPEECH_HAAS_ENCODER_QKV',
                                 'GGML_METAL_MM_SILU', 'GGML_METAL_ADD_SOFTMAX', 'SPEECH_HAAS_TDT_BLANK_BATCH',
                                 'SPEECH_HAAS_KDA_QKV', 'SPEECH_HAAS_POSITION_CACHE', 'GGML_METAL_GLU',
                                 'SPEECH_HAAS_ENCODER_GRAPH_CACHE', 'SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE',
                                 'GGML_METAL_DW_SILU', 'SPEECH_HAAS_DEPTHWISE_PREPACK', 'GGML_METAL_RESIDUAL_NORM',
                                 'GGML_METAL_HEAD_STORE', 'SPEECH_HAAS_TDT_BUFFER_REUSE', 'SPEECH_HAAS_FRONTEND_PARALLEL'}
EXPERIMENT_ENVIRONMENT = BINARY_EXPERIMENT_ENVIRONMENT | {'GGML_METAL_MM_TILE_N'}


def execution_environment(recorded):
    """Return the launch environment only when every backend override is explicitly recorded and allowed."""
    for key, value in recorded.items():
        if key not in EXPERIMENT_ENVIRONMENT:
            raise ValueError('Only explicitly allowed experiment environment variables can be recorded.')
        if key in BINARY_EXPERIMENT_ENVIRONMENT and value not in {'0', '1'}:
            raise ValueError(f'{key} must be explicitly set to 0 or 1.')
    for key in os.environ:
        if key.startswith(('GGML_', 'SPEECH_')) and key not in recorded:
            raise ValueError(f'{key} is inherited but not explicitly recorded; unset it or record an allowed switch explicitly.')
    return {**os.environ, **recorded}


def run(spec):
    child_environment = execution_environment(spec['environment'])
    for item in [spec['build']['speech'], spec['model']]:
        digest = hashlib.sha256()
        with Path(item['path']).open('rb') as f:
            while block := f.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('The model or binary changed after the manifest was written.')
    directory = Path(spec['directory'])
    directory.mkdir(parents=True, exist_ok=False)
    command = [spec['build']['speech']['path'], 'worker', spec['model']['path'], '--device', 'gpu',
               '--threads', str(spec['threads']), '--no-warmup', '--decode-on-cpu']
    memory_log = directory / 'memory.log'
    messages = queue.Queue()
    with (directory / 'stderr.log').open('w') as log:
        started = datetime.now(timezone.utc).isoformat()
        start = time.perf_counter()
        process = subprocess.Popen(['/usr/bin/time', '-l', '-o', str(memory_log), *command],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, encoding='utf-8', start_new_session=True,
                                   env=child_environment)

        def read():
            try:
                for line in process.stdout:
                    messages.put(json.loads(line))
                messages.put(RuntimeError('Worker stdout closed before the next message.'))
            except Exception as error:
                messages.put(error)

        threading.Thread(target=read, daemon=True).start()

        def receive():
            message = messages.get(timeout=120)
            if isinstance(message, Exception):
                raise message
            return message

        def send(message):
            process.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
            process.stdin.flush()

        def recognize(record, request_id, timestamps=False):
            pcm = Path(record['pcm_path']).read_bytes()
            if hashlib.sha256(pcm).hexdigest() != record['pcm_sha256']:
                raise ValueError(f"Input changed: {record['file']}")
            start = time.perf_counter()
            for seq, offset in enumerate(range(0, len(pcm), 32768)):
                send({'type': 'chunk', 'id': request_id, 'seq': seq,
                      'pcm': base64.b64encode(pcm[offset:offset + 32768]).decode()})
            send({'type': 'transcribe', 'id': request_id, 'sample_rate': 16000, 'timestamps': timestamps})
            while True:
                result = receive()
                if result.get('id') != request_id:
                    raise RuntimeError(f'Unexpected request id: {result}')
                if result.get('type') == 'progress':
                    continue
                if result.get('type') != 'end' or result.get('stop') != 'complete':
                    raise RuntimeError(f'Recognition failed: {result}')
                if timestamps and any(not isinstance(result.get(key), list) for key in ['tokens', 'segments']):
                    raise RuntimeError('Timestamp evidence requires public token and segment arrays.')
                break
            elapsed = time.perf_counter() - start
            return {'file': record['file'], 'stratum': record['stratum'], 'audio_seconds': record['seconds'],
                    'seconds': elapsed, 'reference': record['reference'], 'text': result['text'],
                    'tokens': result.get('tokens'), 'segments': result.get('segments')}

        try:
            ready = receive()
            load_seconds = time.perf_counter() - start
            if ready.get('type') != 'ready' or ready.get('protocol') != 4:
                raise RuntimeError(f'Worker did not become ready: {ready}')
            warm = sorted(spec['warmup'], key=lambda r: (-r['seconds'], r['file']))
            warmed = [recognize(record, f'warm-{i}') for i, record in enumerate(warm)]
            measured = []
            records = spec['records']
            shift = (spec['round'] * 7) % len(records)
            for i, record in enumerate(records[shift:] + records[:shift]):
                measured.append(recognize(record, str(i), spec['phase'] == 'quality'))
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError('Worker exited with failure.')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    memory = {label.replace(' ', '_'): int(value) for value, label in re.findall(
        r'^\s*(\d+)\s+(maximum resident set size|peak memory footprint)\s*$', memory_log.read_text(), re.MULTILINE)}
    if 'peak_memory_footprint' not in memory:
        raise RuntimeError(f'Missing peak memory footprint in {memory_log}.')
    report = {'started_utc': started, 'phase': spec['phase'], 'round': spec['round'], 'side': spec['side'],
              'type': spec['model']['type'], 'command': command, 'environment': spec['environment'], 'build': spec['build'], 'ready': ready,
              'load_seconds': load_seconds, 'first_request': warmed[0], 'warmup_count': len(warmed),
              'memory': memory, 'records': measured}
    report['worker_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"{spec['model']['type']} {spec['side']} {spec['phase']} round {spec['round']}: {len(measured)} requests", flush=True)
