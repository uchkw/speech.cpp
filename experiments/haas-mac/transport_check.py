"""Check transport trace accounting with synthetic data; this never starts a worker or loads a model.

Usage: python3 -B experiments/haas-mac/transport_check.py
"""
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
from transport_summary import summarize
from transport_worker import execution_environment, result_digest
import transport_worker


def fixture():
    client = {'schema': 1, 'clock': 'perf_counter_seconds/thread_time_seconds', 'lines': [],
              'received': [], 'waits': [], 'requests': []}
    events = [{'schema': 1, 'event': 'drained', 'lines': 6, 'jobs': 2, 'eof': [500000000, 9000000]}]
    for request, serial in [(1, 4), (2, 9)]:
        py_origin, cpp_origin = request * 10.0, request * 100000000
        first = len(client['lines']) + 1
        for index, kind in enumerate(['chunk', 'chunk', 'transcribe']):
            ordinal = len(client['lines']) + 1
            py = py_origin + index * .001
            native = cpp_origin + index * 1000000
            row = {'line': ordinal, 'request': request, 'kind': kind, 'seq': index if kind == 'chunk' else -1,
                   'bytes': 100 if kind == 'chunk' else 80}
            for name, offset in [('base64_begin', 0), ('base64_end', .0001), ('json_begin', .0002),
                                 ('json_end', .0003), ('write_begin', .0004), ('write_end', .0005), ('flush_end', .0006)]:
                row[name] = [py + offset, py / 100 + offset / 10]
            client['lines'].append(row)
            cpp = {'schema': 1, 'event': 'line', **{key: row[key] for key in ['line', 'kind', 'seq', 'bytes']}, 'serial': serial}
            # A first getline waits well before this request starts; CPU consumption remains small.
            for name, offset in [('read_begin', -90000000 if index == 0 else -100000), ('read_end', 0), ('locked', 1000),
                                 ('parse_begin', 2000), ('parse_end', 10000), ('decode_begin', 11000),
                                 ('decode_end', 20000), ('append_begin', 21000), ('append_end', 25000),
                                 ('queued', 30000), ('dispatch_end', 31000)]:
                cpp[name] = [native + offset, native // 100 + (0 if offset < 0 else offset // 10 + 1000)]
            events.append(cpp)
        job = {'schema': 1, 'event': 'job', 'serial': serial, 'recognition': True}
        for name, offset in [('taken', 2100000), ('entered', 2200000), ('api_begin', 2300000), ('api_end', 17000000),
                             ('terminal_ready', 18000000), ('finish_returned', 50000000)]:
            job[name] = [cpp_origin + offset, (cpp_origin + offset) // 100]
        events.append(job)
        response = {'response_line': request}
        for name, offset in [('read_begin', -1), ('read_end', .019), ('parse_begin', .0191), ('parse_end', .0192),
                             ('enqueue_begin', .0193), ('enqueue_end', .0195)]:
            response[name] = [py_origin + offset, (py_origin + offset) / 100]
        client['received'].append(response)
        client['waits'].append({'response_line': request, 'begin': [py_origin + .003, py_origin / 100],
                                'end': [py_origin + .0194, py_origin / 100 + .0001]})
        client['requests'].append({'request': request, 'first_line': first, 'last_line': len(client['lines']),
                                   'response_line': request, 'start': py_origin, 'end': py_origin + .02,
                                   'elapsed': (py_origin + .02) - py_origin,
                                   'validate_begin': [py_origin + .0196, py_origin / 100 + .0002], 'warmup': request == 1})
    return client, events


def refused(client, events, text):
    try:
        summarize(client, events)
    except ValueError as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError(f'Accepted invalid trace: {text}')


def driver_wire_check():
    """Compare both drivers against an in-memory protocol peer, with no process or model execution."""
    source = subprocess.check_output(['git', 'show', transport_worker.DRIVER_BASE_COMMIT + ':experiments/haas-mac/worker.py'],
                                     cwd=Path(__file__).resolve().parents[2])
    assert hashlib.sha256(source).hexdigest() == transport_worker.DRIVER_BASE_SHA256
    baseline = {'__file__': __file__}
    exec(compile(source, 'pinned_worker.py', 'exec'), baseline)

    class Peer:
        def __init__(self, command, **kwargs):
            self.sent, self.flushes, self.closed = [], 0, False
            self.answers = queue.Queue()
            self.answers.put(json.dumps({'type': 'ready', 'protocol': 4}) + '\n')
            self.stdin = SimpleNamespace(write=self.write, flush=self.flush, close=self.close)
            assert kwargs['text'] is True and kwargs['encoding'] == 'utf-8'
            # The pinned driver iterates the stream, while the diagnostic driver times readline explicitly.
            class Output:
                def readline(inner):
                    return self.answers.get(timeout=5)

                def __iter__(inner):
                    return iter(inner.readline, '')
            self.stdout = Output()
            Path(command[3]).write_text('1 peak memory footprint\n1 maximum resident set size\n')

        def write(self, line):
            self.sent.append(line)
            message = json.loads(line)
            if message['type'] == 'transcribe':
                self.answers.put(json.dumps({'type': 'end', 'id': message['id'], 'stop': 'complete',
                                             'text': 'private transcript fixture', 'tokens': [], 'segments': []}) + '\n')
            return len(line)

        def flush(self):
            self.flushes += 1

        def close(self):
            self.closed = True
            self.answers.put('')

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0 if self.closed else None

    with tempfile.TemporaryDirectory(prefix='haas-transport-fixture-') as directory:
        root = Path(directory)
        (root / 'model').write_bytes(b'synthetic model identity, never loaded')
        identity = {'path': str(root / 'model'), 'sha256': hashlib.sha256((root / 'model').read_bytes()).hexdigest(), 'type': 'fixture'}
        pcm = bytes(range(256)) * 128 + b'\x01\x02'
        (root / 'audio').write_bytes(pcm)
        record = {'pcm_path': str(root / 'audio'), 'pcm_sha256': hashlib.sha256(pcm).hexdigest(),
                  'file': 'fixture', 'stratum': 'fixture', 'seconds': len(pcm) / 32000, 'reference': 'private transcript fixture'}
        peers = []
        for name, run in [('original', baseline['run']), ('diagnostic', transport_worker.run)]:
            def start(command, **kwargs):
                peer = Peer(command, **kwargs)
                peers.append(peer)
                return peer
            spec = {'build': {'speech': identity}, 'model': identity, 'environment': {}, 'directory': str(root / name),
                    'threads': 4, 'warmup': [record], 'records': [record], 'round': 0, 'phase': 'quality', 'side': name}
            with patch.dict(os.environ, {}, clear=True), patch('subprocess.Popen', side_effect=start):
                run(spec)
        assert peers[0].sent == peers[1].sent and peers[0].flushes == peers[1].flushes == 6
        for name in ['client-trace.json', 'result.json']:
            text = (root / 'diagnostic' / name).read_text()
            assert 'private transcript fixture' not in text and '"pcm"' not in text
        assert json.loads((root / 'diagnostic/client-trace.json').read_text())['requests'][1]['warmup'] is False


def main():
    client, events = fixture()
    report = summarize(client, events)
    assert len(report['rows']) == 2 and [row['serial'] for row in report['rows']] == [4, 9]
    row = report['rows'][0]
    assert math.isclose(row['contained_first_read_end_to_terminal_ready_seconds'], .018)
    assert math.isclose(row['unattributed_outside_contained_seconds'], .002)
    assert row['worker']['first_getline_including_idle']['wall'] > row['client_request_seconds']
    assert 0 < row['worker']['first_getline_including_idle']['thread_cpu'] < .00001
    assert row['worker']['finish_lock_write_flush']['wall'] > row['client_request_seconds']
    assert row['client']['terminal_stdout_readline_including_idle']['wall'] > row['client_request_seconds']
    assert not report['syscalls_measured'] and 'total' not in row['worker'] and 'total' not in row['client']

    missing_cpu = deepcopy(events)
    for event in missing_cpu:
        for value in event.values():
            if isinstance(value, list):
                value[1] = -1
    assert summarize(client, missing_cpu)['rows'][0]['worker']['parse_json']['thread_cpu'] is None

    bad = deepcopy(events)
    bad.append(deepcopy(events[-1]))
    refused(client, bad, 'Duplicate serial')
    bad = deepcopy(events)
    del bad[1]['decode_end']
    refused(client, bad, 'Missing point')
    bad = deepcopy(events)
    bad[2]['serial'] = 10
    refused(client, bad, 'Request/serial mismatch')
    bad = deepcopy(events)
    bad[1]['parse_end'][0] = bad[1]['parse_begin'][0] - 1
    refused(client, bad, 'Negative wall interval')
    bad = deepcopy(events)
    bad[1]['decode_begin'][1] = -1
    bad[1]['decode_end'][1] = -2
    refused(client, bad, 'Unmeasured point')
    bad = deepcopy(client)
    bad['requests'][0]['end'] = bad['requests'][0]['start'] + .01
    bad['requests'][0]['elapsed'] = .01
    refused(bad, events, 'Terminal response boundaries differ')
    bad = deepcopy(events)
    bad[0]['lines'] -= 1
    refused(client, bad, 'Incomplete worker trace')
    bad = deepcopy(client)
    bad['lines'][0]['bytes'] += 1
    refused(bad, events, 'Input line metadata differs')
    refused(client, [], 'Missing or unknown worker trace schema')

    names = ['SPEECH_HAAS_WORKER_UNSYNC', 'SPEECH_HAAS_WORKER_UNTIE', 'SPEECH_HAAS_WORKER_TRANSPORT_TRACE']
    with patch.dict(os.environ, {}, clear=True):
        assert execution_environment({}) == {}
        for name in names:
            for value in ['0', '1']:
                assert execution_environment({name: value}) == {name: value}
            try:
                execution_environment({name: 'yes'})
            except ValueError:
                pass
            else:
                raise AssertionError('Non-binary experiment flag passed.')
    with patch.dict(os.environ, {names[0]: '1'}, clear=True):
        try:
            execution_environment({})
        except ValueError:
            pass
        else:
            raise AssertionError('Unrecorded inherited experiment flag passed.')
    original = {'id': 'first', 'text': 'fixture', 'tokens': [], 'segments': [], 'stop': 'complete'}
    assert result_digest(original) == result_digest({**original, 'id': 'reused'})
    assert result_digest(original) != result_digest({**original, 'text': 'different'})
    driver_wire_check()
    print('PASS: independent clocks, overlaps, idle/wait separation, serial pairing, incomplete traces, payload digest, environment guards, pinned-driver wire bytes')


if __name__ == '__main__':
    main()
