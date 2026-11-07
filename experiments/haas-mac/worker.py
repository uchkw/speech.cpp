"""Measure one serialized Haas worker run; used by compare.py under the campaign lock."""
import base64
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import queue
import re
import signal
import struct
import subprocess
import threading
import time

DIAGNOSTIC_EXPERIMENT_ENVIRONMENT = {'GGML_CPU_NORM_MUL_ADD_FUSION_TRACE', 'SPEECH_HAAS_WORKER_TRANSPORT_TRACE'}
BINARY_EXPERIMENT_ENVIRONMENT = {'SPEECH_HAAS_CPU_REPACK', 'SPEECH_HAAS_FFN_F16', 'SPEECH_HAAS_TDT_RESIDENT_STATE',
                                 'SPEECH_HAAS_TDT_JOINT_REUSE', 'SPEECH_HAAS_TDT_CPU_PLAN', 'SPEECH_HAAS_ENCODER_QKV',
                                 'GGML_METAL_MM_SILU', 'GGML_METAL_ADD_SOFTMAX', 'SPEECH_HAAS_TDT_BLANK_BATCH',
                                 'SPEECH_HAAS_KDA_QKV', 'SPEECH_HAAS_POSITION_CACHE', 'GGML_METAL_GLU',
                                 'SPEECH_HAAS_ENCODER_GRAPH_CACHE', 'SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE',
                                 'GGML_METAL_DW_SILU', 'SPEECH_HAAS_DEPTHWISE_PREPACK', 'GGML_METAL_RESIDUAL_NORM',
                                 'GGML_METAL_HEAD_STORE', 'SPEECH_HAAS_TDT_BUFFER_REUSE', 'SPEECH_HAAS_FRONTEND_PARALLEL',
                                 'SPEECH_HAAS_FRONTEND_VDSP', 'SPEECH_HAAS_AUTO_THREADS', 'SPEECH_HAAS_KDA_CONSTANTS',
                                 'GGML_CPU_NORM_MUL_ADD_FUSION', 'SPEECH_HAAS_WORKER_UNSYNC', 'SPEECH_HAAS_WORKER_UNTIE',
                                 'SPEECH_HAAS_TDT_DEVICE_ARGMAX', 'SPEECH_HAAS_REL_FA', 'SPEECH_HAAS_KEYWORD_CACHE',
                                 'SPEECH_HAAS_RAW_MEL_PREFIX', 'GGML_METAL_BIND_CACHE',
                                 'SPEECH_HAAS_KDA_PREFIX_CACHE',
                                 'GGML_METAL_MM_GLU', 'SPEECH_HAAS_KEYWORD_PREFIX'} | DIAGNOSTIC_EXPERIMENT_ENVIRONMENT
EXPERIMENT_ENVIRONMENT = BINARY_EXPERIMENT_ENVIRONMENT | {'GGML_METAL_MM_TILE_N'}

# The decoder placement shared by every side of one run: the worker's --device value and its decode-on-CPU argument.
# A Metal run turns the CPU-decoder flag explicitly false, in the =form the production command line accepts.
DECODER_DEVICES = {'cpu': ('gpu', '--decode-on-cpu'), 'metal': ('MTL0', '--decode-on-cpu=false')}


def execution_environment(recorded, *, diagnostic=False):
    """Return the launch environment only when every backend override is explicitly recorded and allowed."""
    for key, value in recorded.items():
        if key not in EXPERIMENT_ENVIRONMENT:
            raise ValueError('Only explicitly allowed experiment environment variables can be recorded.')
        if key in BINARY_EXPERIMENT_ENVIRONMENT and value not in {'0', '1'}:
            raise ValueError(f'{key} must be explicitly set to 0 or 1.')
        if key in DIAGNOSTIC_EXPERIMENT_ENVIRONMENT and value == '1' and not diagnostic:
            raise ValueError(f'{key}=1 requires the dedicated diagnostic phase; do not enable it in performance runs.')
    for key in os.environ:
        if key.startswith(('GGML_', 'SPEECH_')) and key not in recorded:
            raise ValueError(f'{key} is inherited but not explicitly recorded; unset it or record an allowed switch explicitly.')
    return {**os.environ, **recorded}


def require_evaluation_build(build):
    """Reject a build explicitly reserved for profile training before any evaluation launch."""
    if build.get('performance_evaluation_eligible') is False:
        raise ValueError('A build marked performance_evaluation_eligible=false cannot run an evaluation; use the dedicated pgo_train driver.')


# The opt-in synthetic keyword experiment: request_options on individual warmup and measured records only. The flat
# wire fields below are strictly allowlisted and validated pre-launch; raw keyword text is never coerced or logged.
KEYWORD_REQUEST_OPTIONS = ('keywords', 'keywords_alpha', 'keywords_guard')


def plan_requests(spec):
    """Validate every warmup and measured record pre-launch; precompute the exact execution associations."""
    for key in KEYWORD_REQUEST_OPTIONS:
        if key in spec:
            raise ValueError(f'spec.{key} is misplaced; keyword options belong in record.request_options.')
    if 'request_options' in spec:
        raise ValueError('spec.request_options is misplaced; keyword options belong in record.request_options.')
    warm = sorted(spec['warmup'], key=lambda r: (-r['seconds'], r['file']))
    records = spec['records']
    shift = (spec['round'] * 7) % len(records)
    ordered = ([(f'warm-{i}', record, False) for i, record in enumerate(warm)]
               + [(str(i), record, spec['phase'] == 'quality')
                  for i, record in enumerate(records[shift:] + records[:shift])])
    opted_in = any('request_options' in record for record in (*spec['warmup'], *records))
    for _, record, timestamps in ordered:
        for key in KEYWORD_REQUEST_OPTIONS:
            if key in record:
                raise ValueError(f'record.{key} is misplaced; keyword options belong in record.request_options.')
        if opted_in and 'timestamps' in record:
            raise ValueError('record.timestamps is phase-controlled and would be silently ignored.')
        if 'request_options' not in record:
            continue
        options = record['request_options']
        if type(options) is not dict:
            raise ValueError('record.request_options must be a JSON object.')
        for key in options:
            if key not in KEYWORD_REQUEST_OPTIONS:
                raise ValueError(f'request_options.{key} is not an allowed keyword request option.')
        if 'keywords' in options and type(options['keywords']) is not str:
            raise ValueError('request_options.keywords must be a JSON string.')
        keywords = options.get('keywords', '')
        if keywords:
            if not keywords.strip():
                raise ValueError('request_options.keywords must not be only whitespace and blank lines.')
            # The worker takes keywords as a c_str: a NUL would truncate it, and lone surrogates cannot cross the UTF-8 wire.
            if '\x00' in keywords:
                raise ValueError('request_options.keywords must not contain a NUL character.')
            try:
                keywords.encode('utf-8')
            except UnicodeEncodeError:
                raise ValueError('request_options.keywords must be strict UTF-8 text.') from None
            if timestamps:
                raise ValueError('request_options.keywords is not allowed on a measured quality request; '
                                 'warmup requests may carry them.')
        for key in ('keywords_alpha', 'keywords_guard'):
            if key in options and not keywords:
                raise ValueError(f'request_options.{key} requires a nonempty request_options.keywords.')
        if 'keywords_alpha' in options:
            alpha = options['keywords_alpha']
            if type(alpha) not in (int, float) or alpha <= 0 or (type(alpha) is float and not math.isfinite(alpha)):
                raise ValueError('request_options.keywords_alpha must be a finite number greater than 0.')
            try:
                alpha32 = struct.unpack('<f', struct.pack('<f', alpha))[0]
            except (struct.error, OverflowError):
                alpha32 = math.inf
            if not math.isfinite(alpha32) or alpha32 <= 0:
                raise ValueError('request_options.keywords_alpha must remain finite and greater than 0 '
                                 'after the float32 cast.')
        if 'keywords_guard' in options and type(options['keywords_guard']) is not bool:
            raise ValueError('request_options.keywords_guard must be a JSON boolean.')
    provenance = None
    if opted_in:
        provenance = [{'id': request_id, 'file': record['file'],
                       'request_options': dict(record['request_options']) if 'request_options' in record else {},
                       'timestamps': timestamps}
                      for request_id, record, timestamps in ordered]
    return ordered[:len(warm)], ordered[len(warm):], provenance


def run(spec):
    decoder_device = spec.get('decoder_device', 'cpu')
    if decoder_device not in DECODER_DEVICES:
        raise ValueError(f'decoder_device must be cpu or metal, not {decoder_device!r}.')
    decode_on_cpu = decoder_device == 'cpu'
    if 'decode_on_cpu' in spec and spec['decode_on_cpu'] is not decode_on_cpu:
        raise ValueError(f"decode_on_cpu {spec['decode_on_cpu']!r} disagrees with the requested {decoder_device!r} decoder.")
    child_environment = execution_environment(spec['environment'], diagnostic=spec.get('phase') == 'diagnostic')
    if spec['phase'] != 'pgo_train':
        require_evaluation_build(spec['build'])
    threads = spec['threads']
    if type(threads) is not int or threads < 0:
        raise ValueError('threads must be a nonnegative integer; use 0 to omit the CLI option.')
    expected = spec.get('expected_threads', threads)
    if type(expected) is not int or expected <= 0:
        raise ValueError('expected_threads must be a positive integer when automatic threads are requested.')
    if threads and expected != threads:
        raise ValueError('expected_threads must equal the explicit thread count.')
    thread_selection = {'requested_mode': 'explicit' if threads else 'auto',
                        'requested_threads': threads, 'expected_threads': expected}
    warm_order, measured_order, provenance = plan_requests(spec)
    for item in [spec['build']['speech'], spec['model']]:
        digest = hashlib.sha256()
        with Path(item['path']).open('rb') as f:
            while block := f.read(8 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() != item['sha256']:
            raise ValueError('The model or binary changed after the manifest was written.')
    directory = Path(spec['directory'])
    directory.mkdir(parents=True, exist_ok=False)
    device, decode_argument = DECODER_DEVICES[decoder_device]
    command = [spec['build']['speech']['path'], 'worker', spec['model']['path'], '--device', device]
    if threads:
        command += ['--threads', str(threads)]
    command += ['--no-warmup', decode_argument]
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

        if provenance:
            # Opt-in adapter only: plain controls keep their original bytes; the terminal is taken by reference, not copied.
            entries = {entry['id']: entry for entry in provenance}
            plain_send, plain_receive = send, receive

            def keyword_send(message):
                if message.get('type') == 'transcribe':
                    options = entries[message['id']]['request_options']
                    if options:
                        message = {**message, **options}
                plain_send(message)

            def keyword_receive():
                message = plain_receive()
                if isinstance(message, dict) and message.get('type') == 'end':
                    entry = entries.get(message.get('id'))
                    if entry is not None:
                        entry['terminal'] = message
                return message

            send, receive = keyword_send, keyword_receive

        try:
            ready = receive()
            load_seconds = time.perf_counter() - start
            if ready.get('type') != 'ready' or ready.get('protocol') != 4:
                raise RuntimeError(f'Worker did not become ready: {ready}')
            resolved = ready.get('model', {}).get('threads')
            if type(resolved) is not int or resolved != expected:
                raise ValueError(f'Worker ready.model.threads is {resolved!r}; expected {expected}.')
            thread_selection['resolved_threads'] = resolved
            # The ready message is request-condition evidence of the launch, not an observation of kernel dispatch.
            model_options = ready.get('model', {}).get('options')
            option_default = None
            if isinstance(model_options, list):
                option_default = next((option.get('default') for option in model_options
                                       if isinstance(option, dict) and option.get('name') == 'decode_on_cpu'), None)
            if decoder_device == 'metal':
                ready_device = ready.get('model', {}).get('device')
                if ready_device != 'MTL0':
                    raise ValueError(f'Worker ready.model.device is {ready_device!r}; a metal decoder run requires MTL0.')
                requested_decode = ready.get('defaults', {}).get('decode_on_cpu')
                if type(requested_decode) is not bool or requested_decode:
                    raise ValueError(f'Worker ready.defaults.decode_on_cpu is {requested_decode!r}; '
                                     'a metal decoder run requires the JSON boolean false.')
            warmed = [recognize(record, request_id, timestamps) for request_id, record, timestamps in warm_order]
            measured = []
            for request_id, record, timestamps in measured_order:
                measured.append(recognize(record, request_id, timestamps))
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
              'decoder_device': decoder_device, 'decode_on_cpu': decode_on_cpu,
              'decoder_evidence': {'ready_model_device': ready.get('model', {}).get('device'),
                                   'ready_defaults_decode_on_cpu': ready.get('defaults', {}).get('decode_on_cpu'),
                                   'model_option_default_decode_on_cpu': option_default},
              'load_seconds': load_seconds, 'first_request': warmed[0], 'warmup_count': len(warmed),
              'memory': memory, 'records': measured, 'thread_selection': thread_selection}
    if provenance:
        # Keyword launch defaults stay unobserved unless the worker reports them in ready, which the report already keeps.
        report['request_provenance'] = provenance
    report['worker_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"{spec['model']['type']} {spec['side']} {spec['phase']} round {spec['round']}: {len(measured)} requests", flush=True)
