"""Diagnose a serialized worker run without writing audio or transcript payloads.

Usage: python3 transport_worker.py SPEC.json
Run under the shared campaign performance lock. SPEC is one frozen runner worker spec.
The request timer, PCM chunks, JSON encoding, text pipes, and flushes match the pinned driver.
"""
import argparse
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

DRIVER_BASE_COMMIT = '09cc03857b4af626fd537ee2be85f78c00d31135'
DRIVER_BASE_SHA256 = '9049921d06e74580db1000e27dc5fbb7e4c07a4c0ed1559b4a6cf87a00b7cb88'

BINARY_EXPERIMENT_ENVIRONMENT = {'SPEECH_HAAS_CPU_REPACK', 'SPEECH_HAAS_FFN_F16', 'SPEECH_HAAS_TDT_RESIDENT_STATE',
                                 'SPEECH_HAAS_TDT_JOINT_REUSE', 'SPEECH_HAAS_TDT_CPU_PLAN', 'SPEECH_HAAS_ENCODER_QKV',
                                 'GGML_METAL_MM_SILU', 'GGML_METAL_ADD_SOFTMAX', 'SPEECH_HAAS_TDT_BLANK_BATCH',
                                 'SPEECH_HAAS_KDA_QKV', 'SPEECH_HAAS_POSITION_CACHE', 'GGML_METAL_GLU',
                                 'SPEECH_HAAS_ENCODER_GRAPH_CACHE', 'SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE',
                                 'GGML_METAL_DW_SILU', 'SPEECH_HAAS_DEPTHWISE_PREPACK',
                                 'SPEECH_HAAS_WORKER_UNSYNC', 'SPEECH_HAAS_WORKER_UNTIE',
                                 'SPEECH_HAAS_WORKER_TRANSPORT_TRACE'}
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


def stamp():
    return [time.perf_counter(), time.thread_time()]


def result_digest(result):
    values = {key: result.get(key) for key in ['text', 'tokens', 'segments', 'stop']}
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def run(spec):
    child_environment = execution_environment(spec['environment'])
    trace = {'schema': 1, 'driver_base_commit': DRIVER_BASE_COMMIT, 'driver_base_sha256': DRIVER_BASE_SHA256,
             'clock': 'perf_counter_seconds/thread_time_seconds', 'lines': [], 'received': [], 'waits': [], 'requests': []}
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
                while True:
                    row = {'response_line': len(trace['received']) + 1, 'read_begin': stamp()}
                    line = process.stdout.readline()
                    row['read_end'] = stamp()
                    if not line:
                        break
                    row['parse_begin'] = stamp()
                    value = json.loads(line)
                    row['parse_end'] = stamp()
                    row['enqueue_begin'] = stamp()
                    messages.put((value, row['response_line']))
                    row['enqueue_end'] = stamp()
                    trace['received'].append(row)
                messages.put(RuntimeError('Worker stdout closed before the next message.'))
            except Exception as error:
                messages.put(error)

        reader = threading.Thread(target=read, daemon=True)
        reader.start()

        def receive():
            row = {'begin': stamp()}
            message = messages.get(timeout=120)
            row['end'] = stamp()
            if isinstance(message, Exception):
                raise message
            value, row['response_line'] = message
            trace['waits'].append(row)
            return value, row['response_line']

        def send(message, request_index, encoding=None):
            row = {'line': len(trace['lines']) + 1, 'request': request_index,
                   'kind': message['type'], 'seq': message.get('seq', -1), 'json_begin': stamp()}
            line = json.dumps(message, ensure_ascii=False) + '\n'
            row['json_end'] = stamp()
            row['bytes'] = len(line) - 1
            row['write_begin'] = stamp()
            process.stdin.write(line)
            row['write_end'] = stamp()
            process.stdin.flush()
            row['flush_end'] = stamp()
            if encoding is not None:
                row.update(encoding)
            trace['lines'].append(row)

        def recognize(record, request_id, timestamps=False):
            pcm = Path(record['pcm_path']).read_bytes()
            if hashlib.sha256(pcm).hexdigest() != record['pcm_sha256']:
                raise ValueError(f"Input changed: {record['file']}")
            request_index = len(trace['requests']) + 1
            first_line = len(trace['lines']) + 1
            start = time.perf_counter()
            for seq, offset in enumerate(range(0, len(pcm), 32768)):
                encoding = {'base64_begin': stamp()}
                encoded = base64.b64encode(pcm[offset:offset + 32768]).decode()
                encoding['base64_end'] = stamp()
                send({'type': 'chunk', 'id': request_id, 'seq': seq, 'pcm': encoded}, request_index, encoding)
            send({'type': 'transcribe', 'id': request_id, 'sample_rate': 16000, 'timestamps': timestamps}, request_index)
            while True:
                result, response_line = receive()
                validate_begin = stamp()
                if result.get('id') != request_id:
                    raise RuntimeError('Unexpected request id in protocol.')
                if result.get('type') == 'progress':
                    continue
                if result.get('type') != 'end' or result.get('stop') != 'complete':
                    raise RuntimeError('Recognition did not end with complete status.')
                if timestamps and any(not isinstance(result.get(key), list) for key in ['tokens', 'segments']):
                    raise RuntimeError('Timestamp evidence requires public token and segment arrays.')
                break
            end = time.perf_counter()
            elapsed = end - start
            trace['requests'].append({'request': request_index, 'first_line': first_line, 'last_line': len(trace['lines']),
                                      'response_line': response_line, 'start': start, 'end': end, 'elapsed': elapsed,
                                      'validate_begin': validate_begin, 'warmup': request_id.startswith('warm-')})
            return {'file': record['file'], 'stratum': record['stratum'], 'audio_seconds': record['seconds'],
                    'seconds': elapsed, 'terminal_sha256': result_digest(result)}

        try:
            ready, _ = receive()
            load_seconds = time.perf_counter() - start
            if ready.get('type') != 'ready' or ready.get('protocol') != 4:
                raise RuntimeError('Worker did not become ready with protocol 4.')
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
            reader.join(timeout=10)
            if reader.is_alive():
                raise RuntimeError('Worker stdout reader did not drain.')
            tail = messages.get_nowait()
            if not isinstance(tail, RuntimeError) or str(tail) != 'Worker stdout closed before the next message.' or not messages.empty():
                raise RuntimeError('Unexpected trailing stdout message or reader error.')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    (directory / 'client-trace.json').write_text(json.dumps(trace, indent=2) + '\n')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('spec', type=Path)
    args = parser.parse_args()
    run(json.loads(args.spec.read_text()))
