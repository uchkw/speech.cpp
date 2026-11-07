"""Check worker thread selection, decoder-device selection and ready validation without a model or subprocess."""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import worker


def check(root):
    root.mkdir()
    artifact = root / 'fixture'; artifact.write_bytes(b'fixture')
    identity = {'path': str(artifact), 'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest()}
    # Recognition PCM crossing the protocol's 32768-byte chunk boundary: two chunks of 32768 and 64 bytes.
    pcm = root / 'boundary.pcm'; pcm.write_bytes(bytes(range(256)) * 128 + bytes(range(64)))
    record = {'file': 'boundary.wav', 'pcm_path': str(pcm), 'pcm_sha256': hashlib.sha256(pcm.read_bytes()).hexdigest(),
              'stratum': 'fixture', 'seconds': 1., 'reference': 'same'}
    base = {'environment': {}, 'build': {'speech': identity}, 'model': {**identity, 'type': 'fixture'},
            'phase': 'latency', 'round': 0, 'side': 'fixture', 'warmup': [record], 'records': [record]}

    def launch(name, requested, expected, resolved, accepted, phase='latency', build_flags=None, environment=None,
               decoder_device=None, ready_model=None, ready_defaults=None, error=None):
        spec = {**base, 'threads': requested, 'directory': str(root / name), 'phase': phase,
                'build': {**base['build'], **(build_flags or {})}, 'environment': environment or {}}
        if expected is not None:
            spec['expected_threads'] = expected
        if decoder_device is not None:
            spec['decoder_device'] = decoder_device
            spec['decode_on_cpu'] = decoder_device == 'cpu'
        sent, flushes = [], []

        class Input(io.StringIO):
            def write(self, message):
                sent.append(message); return super().write(message)

            def flush(self):
                flushes.append(len(sent)); super().flush()

        def start(command, **options):
            suffix = ['--no-warmup', '--decode-on-cpu=false' if decoder_device == 'metal' else '--decode-on-cpu']
            if requested:
                suffix = ['--threads', str(requested), *suffix]
            assert command[4:] == [str(artifact), 'worker', str(artifact), '--device',
                                   'MTL0' if decoder_device == 'metal' else 'gpu', *suffix]
            assert options['env'] == worker.execution_environment(spec['environment'], diagnostic=phase == 'diagnostic')
            Path(command[3]).write_text('100 peak memory footprint\n')
            ready = {'type': 'ready', 'protocol': 4, 'model': {'threads': resolved, **(ready_model or {})}}
            if ready_defaults is not None:
                ready['defaults'] = ready_defaults
            messages = [ready]
            messages.extend({'type': 'end', 'id': request_id, 'stop': 'complete', 'text': 'same'}
                            for request_id in ['warm-0', '0'])
            return SimpleNamespace(stdout=io.StringIO(''.join(json.dumps(m) + '\n' for m in messages)),
                                   stdin=Input(), wait=lambda timeout: 0, poll=lambda: 0)

        with patch.object(worker.subprocess, 'Popen', start), patch('builtins.print'):
            if accepted:
                worker.run(spec)
                report = json.loads((Path(spec['directory']) / 'result.json').read_text())
                assert report['phase'] == phase and report['build'] == spec['build']
                assert report['environment'] == spec['environment']
                assert report['thread_selection'] == {
                    'requested_mode': 'explicit' if requested else 'auto', 'requested_threads': requested,
                    'expected_threads': expected if expected is not None else requested, 'resolved_threads': resolved}
                assert report['decoder_device'] == (decoder_device or 'cpu')
                assert report['decode_on_cpu'] is ((decoder_device or 'cpu') == 'cpu')
                expected_ready = {'type': 'ready', 'protocol': 4, 'model': {'threads': resolved, **(ready_model or {})}}
                if ready_defaults is not None:
                    expected_ready['defaults'] = ready_defaults
                assert report['ready'] == expected_ready
                option_default = next((option.get('default') for option in (ready_model or {}).get('options', [])
                                       if option.get('name') == 'decode_on_cpu'), None)
                assert report['decoder_evidence'] == {'ready_model_device': (ready_model or {}).get('device'),
                                                      'ready_defaults_decode_on_cpu': (ready_defaults or {}).get('decode_on_cpu'),
                                                      'model_option_default_decode_on_cpu': option_default}
                messages = [json.loads(line) for line in sent]
                assert [m['id'] for m in messages if m['type'] == 'transcribe'] == ['warm-0', '0']
                return sent, flushes, messages
            try:
                worker.run(spec)
            except ValueError as failure:
                assert (error or 'ready.model.threads') in str(failure), failure
            else:
                raise AssertionError('An unexpected worker run was accepted.')
            assert not sent and not (Path(spec['directory']) / 'result.json').exists()

    for requested, expected, resolved in [(0, 16, 16), (0, 4, 4), (16, None, 16), (4, None, 4)]:
        launch(f'pass-{requested}-{resolved}', requested, expected, resolved, True)
    for requested, expected, resolved in [(0, 4, 16), (16, None, 4), (4, None, None), (4, None, True)]:
        launch(f'refuse-{requested}-{resolved}', requested, expected, resolved, False)
    launch('pass-explicit-cpu', 4, None, 4, True, decoder_device='cpu')
    sent, flushes, messages = launch('pass-wire-bytes', 4, None, 4, True)
    assert flushes == list(range(1, len(sent) + 1))
    for line, message in zip(sent, messages):
        assert line == json.dumps(message, ensure_ascii=False) + '\n'
    assert sent[2] == json.dumps({'type': 'transcribe', 'id': 'warm-0', 'sample_rate': 16000, 'timestamps': False},
                                 ensure_ascii=False) + '\n'
    assert sent[5] == json.dumps({'type': 'transcribe', 'id': '0', 'sample_rate': 16000, 'timestamps': False},
                                 ensure_ascii=False) + '\n'
    for request_id in ['warm-0', '0']:
        round_messages = [m for m in messages if m.get('id') == request_id]
        assert [m['type'] for m in round_messages] == ['chunk', 'chunk', 'transcribe']
        chunks = [m for m in round_messages if m['type'] == 'chunk']
        assert [m['seq'] for m in chunks] == [0, 1]
        assert [len(base64.b64decode(m['pcm'])) for m in chunks] == [32768, 64]
        assert b''.join(base64.b64decode(m['pcm']) for m in chunks) == pcm.read_bytes()

    metal_model = {'device': 'MTL0', 'options': [{'name': 'decode_on_cpu', 'type': 'bool', 'default': True}]}
    for name, option_default in [('q4', True), ('q8', True), ('f16', False)]:
        launch(f'pass-metal-{name}', 4, None, 4, True, decoder_device='metal',
               ready_model={'device': 'MTL0', 'options': [{'name': 'decode_on_cpu', 'type': 'bool', 'default': option_default}]},
               ready_defaults={'decode_on_cpu': False})
    launch('pass-metal-auto-threads', 0, 16, 16, True, decoder_device='metal',
           ready_model=metal_model, ready_defaults={'decode_on_cpu': False})
    launch('refuse-metal-device', 4, None, 4, False, decoder_device='metal',
           ready_model={**metal_model, 'device': 'Metal'}, ready_defaults={'decode_on_cpu': False},
           error='ready.model.device')
    launch('refuse-metal-device-missing', 4, None, 4, False, decoder_device='metal',
           ready_model={'options': metal_model['options']}, ready_defaults={'decode_on_cpu': False},
           error='ready.model.device')
    for label, value in [('true', True), ('string', 'false'), ('numeric', 0)]:
        launch(f'refuse-metal-default-{label}', 4, None, 4, False, decoder_device='metal',
               ready_model=metal_model, ready_defaults={'decode_on_cpu': value}, error='ready.defaults.decode_on_cpu')
    launch('refuse-metal-default-missing', 4, None, 4, False, decoder_device='metal',
           ready_model=metal_model, ready_defaults={}, error='ready.defaults.decode_on_cpu')
    launch('refuse-metal-threads', 4, None, 16, False, decoder_device='metal',
           ready_model=metal_model, ready_defaults={'decode_on_cpu': False}, error='ready.model.threads')

    for phase in ['latency', 'quality', 'diagnostic']:
        with patch.object(worker.subprocess, 'Popen') as start:
            try:
                worker.run({**base, 'phase': phase, 'threads': 4, 'directory': str(root / 'ineligible'),
                            'build': {**base['build'], 'performance_evaluation_eligible': False}})
            except ValueError as error:
                assert 'performance_evaluation_eligible=false' in str(error), error
            else:
                raise AssertionError('An ineligible build reached evaluation.')
            start.assert_not_called()
            assert not (root / 'ineligible').exists()
    launch('training-only', 4, None, 4, True, 'pgo_train',
           {'instrumented': True, 'performance_evaluation_eligible': False})
    launch('eligible', 4, None, 4, True, build_flags={'performance_evaluation_eligible': True})
    for key in ['GGML_CPU_NORM_MUL_ADD_FUSION_TRACE', 'SPEECH_HAAS_WORKER_TRANSPORT_TRACE']:
        launch(key, 4, None, 4, True, phase='diagnostic', environment={key: '1'})
    for values, reason in [({'threads': 0}, 'expected_threads'), ({'threads': -1}, 'threads'),
                           ({'threads': True}, 'threads'), ({'threads': 0, 'expected_threads': 0}, 'expected_threads'),
                           ({'threads': 4, 'expected_threads': 16}, 'explicit thread')]:
        with patch.object(worker.subprocess, 'Popen') as start:
            try:
                worker.run({**base, **values, 'directory': str(root / 'invalid')})
            except ValueError as error:
                assert reason in str(error), error
            else:
                raise AssertionError('An invalid thread expectation was accepted.')
            start.assert_not_called()
    for values, reason in [({'decoder_device': 'cuda'}, 'decoder_device must be cpu or metal'),
                           ({'decoder_device': 'metal', 'decode_on_cpu': True}, 'disagrees')]:
        with patch.object(worker.subprocess, 'Popen') as start:
            try:
                worker.run({**base, 'threads': 4, 'directory': str(root / 'invalid-decoder'), **values})
            except ValueError as error:
                assert reason in str(error), error
            else:
                raise AssertionError('An invalid decoder selection was accepted.')
            start.assert_not_called()
            assert not (root / 'invalid-decoder').exists()
