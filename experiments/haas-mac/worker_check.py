"""Check worker thread selection and ready validation without a model or subprocess."""
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
    record = {'file': 'fixture.wav', 'pcm_path': str(artifact), 'pcm_sha256': identity['sha256'],
              'stratum': 'fixture', 'seconds': 1., 'reference': 'same'}
    base = {'environment': {}, 'build': {'speech': identity}, 'model': {**identity, 'type': 'fixture'},
            'phase': 'latency', 'round': 0, 'side': 'fixture', 'warmup': [record], 'records': [record]}

    def launch(name, requested, expected, resolved, accepted, phase='latency', build_flags=None):
        spec = {**base, 'threads': requested, 'directory': str(root / name), 'phase': phase,
                'build': {**base['build'], **(build_flags or {})}}
        if expected is not None:
            spec['expected_threads'] = expected
        sent = []

        class Input(io.StringIO):
            def write(self, message):
                sent.append(json.loads(message)); return super().write(message)

        def start(command, **options):
            suffix = ['--no-warmup', '--decode-on-cpu']
            if requested:
                suffix = ['--threads', str(requested), *suffix]
            assert command[4:] == [str(artifact), 'worker', str(artifact), '--device', 'gpu', *suffix]
            assert options['env'] == worker.execution_environment(spec['environment'])
            Path(command[3]).write_text('100 peak memory footprint\n')
            messages = [{'type': 'ready', 'protocol': 4, 'model': {'threads': resolved}}]
            messages.extend({'type': 'end', 'id': request_id, 'stop': 'complete', 'text': 'same'}
                            for request_id in ['warm-0', '0'])
            return SimpleNamespace(stdout=io.StringIO(''.join(json.dumps(m) + '\n' for m in messages)),
                                   stdin=Input(), wait=lambda timeout: 0, poll=lambda: 0)

        with patch.object(worker.subprocess, 'Popen', start), patch('builtins.print'):
            if accepted:
                worker.run(spec)
                report = json.loads((Path(spec['directory']) / 'result.json').read_text())
                assert report['phase'] == phase and report['build'] == spec['build']
                assert report['thread_selection'] == {
                    'requested_mode': 'explicit' if requested else 'auto', 'requested_threads': requested,
                    'expected_threads': expected if expected is not None else requested, 'resolved_threads': resolved}
                assert [m['id'] for m in sent if m['type'] == 'transcribe'] == ['warm-0', '0']
            else:
                try:
                    worker.run(spec)
                except ValueError as error:
                    assert 'ready.model.threads' in str(error), error
                else:
                    raise AssertionError('An unexpected resolved thread count was accepted.')
                assert not sent and not (Path(spec['directory']) / 'result.json').exists()

    for requested, expected, resolved in [(0, 16, 16), (0, 4, 4), (16, None, 16), (4, None, 4)]:
        launch(f'pass-{requested}-{resolved}', requested, expected, resolved, True)
    for requested, expected, resolved in [(0, 4, 16), (16, None, 4), (4, None, None), (4, None, True)]:
        launch(f'refuse-{requested}-{resolved}', requested, expected, resolved, False)
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
