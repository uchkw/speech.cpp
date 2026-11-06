#!/usr/bin/env python3
"""Runs one serialized worker probe with startup and request markers outside its timers.

Usage: run.py --build build.json --assets assets.json --out NEW_DIRECTORY
              --warmup standard|none --trace 0|1 [--round N] [--type q4_k]
"""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
from pipeline_trace import analyze

LOCK = '/private/tmp/speech-haas-mac-perf.lock'
SWITCH = 'GGML_METAL_PIPELINE_TRACE'
SCREEN = (0, 4, 9, 10, 14, 19, 20, 24, 29)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(args):
    inherited = sorted(key for key in os.environ if key.startswith(('GGML_', 'SPEECH_')))
    if inherited:
        raise ValueError('Unset unrecorded inherited experiment variables: ' + ', '.join(inherited))
    build, assets = read(args.build), read(args.assets)
    if not build.get('source_verified'):
        raise ValueError('The probe requires a source-verified build manifest.')
    model = next(item for item in assets['models'] if item['type'] == args.type)
    if len(assets['subset']) != 30:
        raise ValueError('The campaign subset must contain 30 records.')
    records = [assets['subset'][i] for i in SCREEN]
    if len({record['file'] for record in records}) != len(records):
        raise ValueError('The screen contains duplicate file keys.')
    for item in (build['speech'], model):
        if sha256(item['path']) != item['sha256']:
            raise ValueError('A binary or model differs from its manifest: ' + item['path'])
    destination = Path(args.out).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    snapshot = destination / 'runner-snapshot'
    snapshot.mkdir()
    sources = {}
    for name in ('run.py', 'pipeline_trace.py'):
        source = Path(__file__).resolve().with_name(name)
        shutil.copyfile(source, snapshot / name)
        sources[name] = sha256(source)
    environment = {SWITCH: args.trace}
    command = [build['speech']['path'], 'worker', model['path'], '--device', 'gpu',
               '--threads', '4', '--decode-on-cpu']
    if args.warmup == 'none':
        command.append('--no-warmup')
    manifest = {
        'schema': 1, 'purpose': 'startup and pipeline diagnostics, not an optimization trial',
        'build': build, 'model': model, 'records': records, 'threads': 4, 'round': args.round,
        'warmup': args.warmup, 'trace': args.trace, 'environment': environment, 'command': command,
        'source_sha256': sources, 'assets_sha256': sha256(args.assets),
        'build_manifest_sha256': sha256(args.build), 'python': sys.version,
        'lock': LOCK, 'counter_time': 'Metal encode completion, not GPU completion',
        'latency_definition': 'After PCM read and hash, before base64/JSON/chunk writes, through validated terminal queue receipt.',
        'marker_definition': 'Python markers are outside request timers. No Python/C++ absolute clocks are compared.',
        'cold_definition': 'Fresh worker process; filesystem and opaque Apple caches are not purged.',
    }
    write(destination / 'run.json', manifest)
    messages = queue.Queue()
    memory_log = destination / 'memory.log'
    started = datetime.now(timezone.utc).isoformat()
    with (destination / 'stderr.log').open('ab', buffering=0) as log, \
            (destination / 'stdout.jsonl').open('x', encoding='utf-8') as stdout:
        def marker(event, **fields):
            payload = ('HAAS_PIPELINE_PHASE ' + json.dumps({'event': event, **fields}, ensure_ascii=False) + '\n').encode()
            if log.write(payload) != len(payload):
                raise RuntimeError('Cannot write a complete diagnostic marker.')

        marker('launch_begin')
        launch = time.perf_counter()
        process = subprocess.Popen(['/usr/bin/time', '-l', '-o', str(memory_log), *command],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                                   text=True, encoding='utf-8', start_new_session=True,
                                   env={**os.environ, **environment})

        def reader():
            try:
                for line in process.stdout:
                    stdout.write(line)
                    messages.put(json.loads(line))
                messages.put(RuntimeError('Worker stdout closed before the next message.'))
            except Exception as error:
                messages.put(error)

        reader_thread = threading.Thread(target=reader, daemon=True)
        reader_thread.start()

        def receive():
            message = messages.get(timeout=120)
            if isinstance(message, Exception):
                raise message
            return message

        def send(message):
            process.stdin.write(json.dumps(message, ensure_ascii=False) + '\n')
            process.stdin.flush()

        def recognize(record, phase, index):
            pcm = Path(record['pcm_path']).read_bytes()
            if len(pcm) != record['samples'] * 2 or hashlib.sha256(pcm).hexdigest() != record['pcm_sha256']:
                raise ValueError('PCM input differs from the assets manifest: ' + record['file'])
            request_id = f'{phase}-{index}'
            timestamps = phase == 'quality'
            marker('request_begin', phase=phase, id=request_id, file=record['file'], timestamps=timestamps)
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
                if timestamps and any(not isinstance(result.get(key), list) for key in ('tokens', 'segments')):
                    raise RuntimeError('The quality phase requires public tokens and segments.')
                break
            elapsed = time.perf_counter() - start
            since_launch = time.perf_counter() - launch
            marker('request_end', phase=phase, id=request_id, file=record['file'])
            return {'phase': phase, 'id': request_id, 'file': record['file'], 'stratum': record['stratum'],
                    'pcm_sha256': record['pcm_sha256'], 'audio_seconds': record['seconds'], 'seconds': elapsed,
                    'launch_to_result_seconds': since_launch, 'terminal': result}

        try:
            ready = receive()
            ready_seconds = time.perf_counter() - launch
            if ready.get('type') != 'ready' or ready.get('protocol') != 4:
                raise RuntimeError(f'Worker did not become ready: {ready}')
            marker('ready', warmup=args.warmup)
            rows = [recognize(record, 'first', i) for i, record in enumerate(
                sorted(records, key=lambda item: (-item['seconds'], item['file'])))]
            shift = (args.round * 7) % len(records)
            rows += [recognize(record, 'repeat', i) for i, record in enumerate(records[shift:] + records[:shift])]
            rows += [recognize(record, 'quality', i) for i, record in enumerate(records)]
            process.stdin.close()
            if process.wait(timeout=30) != 0:
                raise RuntimeError('Worker exited with failure.')
            reader_thread.join(timeout=5)
            if reader_thread.is_alive():
                raise RuntimeError('Worker stdout reader did not finish.')
            marker('process_end')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            reader_thread.join(timeout=5)
    memory = {label.replace(' ', '_'): int(value) for value, label in re.findall(
        r'^\s*(\d+)\s+(maximum resident set size|peak memory footprint)\s*$', memory_log.read_text(), re.MULTILINE)}
    if 'peak_memory_footprint' not in memory:
        raise RuntimeError('The worker did not report peak memory footprint.')
    write(destination / 'result.json', {'started_utc': started, 'ready_seconds': ready_seconds, 'ready': ready,
                                      'memory': memory, 'records': rows, 'manifest_sha256': sha256(destination / 'run.json')})
    write(destination / 'trace-summary.json', analyze(destination))
    print(str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', required=True)
    parser.add_argument('--assets', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--warmup', choices=('standard', 'none'), required=True)
    parser.add_argument('--trace', choices=('0', '1'), required=True)
    parser.add_argument('--type', choices=('q4_k', 'q8_0', 'f16'), default='q4_k')
    parser.add_argument('--round', type=int, default=0)
    parser.add_argument('--locked', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.round < 0:
        parser.error('--round must be nonnegative')
    if not args.locked:
        os.execv('/usr/bin/lockf', ['/usr/bin/lockf', '-k', LOCK, '/usr/bin/python3', '-B',
                                  str(Path(__file__).resolve()), *sys.argv[1:], '--locked'])
    run(args)


if __name__ == '__main__':
    main()
