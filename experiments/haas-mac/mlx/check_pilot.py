#!/usr/bin/env python3
"""Synthetic checks for resident.py and pilot.py.

Standard library only: no downloads, no MLX package, no real audio and no
large data. The resident protocol boundary is driven through a fake transcribe
callable; the driver is driven through fake children, a fake clock and
injected PCM reads. The load-surface fakes carry a small symbolic BF16
parameter tree (dtype and shape metadata only), so the reviewed strict
cast helper and the truthful FP16 ready profile run for real here. The only
subprocess is one tiny stdlib Python child, a local fixture that blocks its
stdin so the blocked-pipe watchdog is exercised for real. This file checks
behavior only; it asserts no measurement results and makes no pass claims
beyond these tests.
"""

from __future__ import annotations

import base64
import enum
import hashlib
import io
import json
import os
import platform
import queue
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pilot    # noqa: E402  (imports the parent campaign helpers)
import precision  # noqa: E402
import resident  # noqa: E402

FP16_PROFILE = {'profile': 'fp16-parameters', 'cast': True, 'source_dtype': 'bfloat16',
                'parameter_dtype': 'float16', 'master_dtype': None,
                'arithmetic': precision.ARITHMETIC_NOTE,
                'tensor_count': 731, 'element_count': 615871520,
                'shape_sample': [['embed.weight', [4609, 1024]], ['fb', [128, 257]]]}
MLX_READY = {'type': 'ready', 'runtime': 'haas-mlx', 'protocol': 'haas-mlx-pilot-1',
             'mlx': {'version': '0.32.4', 'mlx_metal_version': '0.32.4', 'device': 'gpu',
                     'device_name': 'Apple M2 Ultra'},
             'python': {'executable': '/fake/bin/python3.12', 'version': '3.12.11',
                        'machine': 'arm64'},
             'model': {'precision': FP16_PROFILE}}
NATIVE_READY = {'type': 'ready', 'protocol': 4, 'model': {'device': 'MTL0', 'threads': 4},
                'defaults': {'decode_on_cpu': True}}


def chunk(request_id, seq, pcm):
    return {'type': 'chunk', 'id': request_id, 'seq': seq,
            'pcm': base64.b64encode(pcm).decode()}


def terminal(request_id, **overrides):
    message = {'type': 'transcribe', 'id': request_id, 'sample_rate': 16000, 'timestamps': False}
    message.update(overrides)
    return message


class FakeClock:
    """Deterministic clock that logs every read so boundaries can be checked."""

    def __init__(self, events):
        self.events = events
        self.now = 1000.0

    def __call__(self):
        self.events.append(('clock', self.now))
        return self.now

    def advance(self, seconds):
        self.now += seconds


class FakeStdout:
    """A line queue the pilot reader thread consumes; close() means EOF."""

    def __init__(self):
        self.lines = queue.Queue()
        self.closed = False       # close() is idempotent, like a dead child's pipe
        self.read_to_eof = False  # set by the consumer when iteration ends

    def emit(self, message):
        self.lines.put(json.dumps(message, ensure_ascii=False))

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.lines.put(None)

    def __iter__(self):
        while True:
            line = self.lines.get()
            if line is None:
                self.read_to_eof = True
                return
            yield line


class FakeProcess:
    """Minimal Popen stand-in: never alive once waited or shut down."""

    def __init__(self, stdout, stdin, exit_code=0, memory_log=None):
        self.stdout, self.stdin = stdout, stdin
        self.exit_code, self.memory_log = exit_code, memory_log
        self.done, self.shutdowns = False, 0
        self.pid = -424242

    def die(self):
        """A dead child's stdout means EOF; repeated close stays a single EOF."""
        self.done = True
        self.stdout.close()

    def poll(self):
        return None if not self.done else self.exit_code

    def wait(self, timeout=None):
        self.die()
        if self.memory_log is not None:
            self.memory_log.write_text('  4242  peak memory footprint\n')
        return self.exit_code


def recording_shutdown(process):
    process.shutdowns += 1
    process.die()


class FakeChild:
    """In-process stand-in for a resident worker subprocess."""

    def __init__(self, events, ready, *, progress=False, error=False, silent=False,
                 exit_code=0, clock=None, close_after_ready=False, fail_after=None):
        self.events, self.ready = events, ready
        self.progress, self.error, self.silent = progress, error, silent
        self.exit_code, self.clock = exit_code, clock
        self.close_after_ready = close_after_ready
        self.fail_after = fail_after
        self.stdout = FakeStdout()
        self.requests, self.terminals = [], []
        self.command = self.env = self.process = None
        self._current = None

    def spawn(self, command, env, memory_log, stderr_log):
        self.command, self.env = list(command), dict(env)
        self.stdout.emit(self.ready)
        if self.close_after_ready:
            self.stdout.close()
        self.process = FakeProcess(self.stdout, self._stdin(),
                                   exit_code=self.exit_code, memory_log=memory_log)
        return self.process

    def _stdin(self):
        child = self

        class _Stdin:
            def __init__(self):
                self.closed = False

            def write(self, text):
                for line in text.splitlines():
                    child.handle(json.loads(line))

            def flush(self):
                pass

            def close(self):
                if self.closed:  # one EOF, like a real pipe wrapper
                    return
                self.closed = True
                child.events.append(('stdin_closed',))

        return _Stdin()

    def handle(self, message):
        self.events.append(('send', message.get('type'), message.get('id'), message.get('seq')))
        if message.get('type') == 'chunk':
            pcm = base64.b64decode(message['pcm'], validate=True)
            if self._current is None or self._current['id'] != message['id']:
                self._current = {'id': message['id'], 'seqs': [], 'sizes': [], 'parts': []}
            self._current['seqs'].append(message['seq'])
            self._current['sizes'].append(len(pcm))
            self._current['parts'].append(pcm)
        elif message.get('type') == 'transcribe':
            request = self._current
            self._current = None
            request['pcm'] = b''.join(request['parts'])
            self.requests.append(request)
            self.terminals.append(message)
            if self.silent:
                return
            if self.error or (self.fail_after is not None and len(self.requests) > self.fail_after):
                self.stdout.emit({'type': 'error', 'id': message['id'], 'stop': 'error',
                                  'message': 'synthetic failure'})
                return
            if self.progress:
                self.stdout.emit({'type': 'progress', 'id': message['id']})
            if self.clock is not None:
                self.clock.advance(1.5)
            self.stdout.emit({'type': 'end', 'id': message['id'], 'stop': 'complete',
                              'text': f"text-{message['id']}-{len(request['pcm'])}"})


class FakeDeviceKind(enum.Enum):
    gpu = 'gpu'
    cpu = 'cpu'


class FakeSamples:
    """The minimum ndarray surface the resident transcribe path touches."""

    def __init__(self, pcm):
        self.pcm = pcm

    def astype(self, dtype):
        if dtype != 'float32':
            raise ValueError(f'fake astype got dtype {dtype!r}')
        return self

    def __truediv__(self, divisor):
        return [int.from_bytes(self.pcm[offset:offset + 2], 'little', signed=True) / divisor
                for offset in range(0, len(self.pcm), 2)]


BF16 = 'bfloat16'
FLOAT16 = 'float16'
FAKE_TENSORS = 4
FAKE_ELEMENTS = 4609 * 1024 + 512 * 512 + 64 + 128 * 128  # 4998208


class FakeLeaf:
    """A symbolic parameter leaf: dtype and shape metadata only, never data."""

    def __init__(self, dtype, shape):
        self.dtype, self.shape = dtype, list(shape)


def FAKE_PARAMETER_TREE():
    """A small all-BF16 tree with the dot-name nesting the port uses."""
    return {
        'embed': {'weight': FakeLeaf(BF16, [4609, 1024])},
        'encoder': {'layers': {
            '0': {'proj': FakeLeaf(BF16, [512, 512])},
            '1': {'proj': FakeLeaf(BF16, [64])},
            '2': {'proj': FakeLeaf(BF16, [128, 128])}}},
    }


def fake_tree_flatten(tree, prefix=''):
    """The mlx.utils.tree_flatten surface: stable dot names, depth first."""
    flat = []
    for key, value in tree.items():
        name = f'{prefix}.{key}' if prefix else str(key)
        if isinstance(value, dict):
            flat.extend(fake_tree_flatten(value, name))
        else:
            flat.append((name, value))
    return flat


class FakeHaasModel:
    """The real load surface only: a parameter tree plus nn.Module.set_dtype."""

    def __init__(self, tree, events=None):
        self.tree, self.events = tree, events

    def parameters(self):
        return self.tree

    def set_dtype(self, dtype):
        if self.events is not None:
            self.events.append(('set_dtype', dtype))

        def cast(node):
            return {key: cast(value) if isinstance(value, dict) else FakeLeaf(dtype, value.shape)
                    for key, value in node.items()}
        self.tree = cast(self.tree)


def fake_mlx_modules(events, device_kind='gpu'):
    """The minimum real MLX import surface; mlx.metal has no get_device_name."""
    mlx = types.ModuleType('mlx')
    core = types.ModuleType('mlx.core')
    metal = types.ModuleType('mlx.metal')
    utils = types.ModuleType('mlx.utils')
    metal.is_available = lambda: True
    core.metal = metal
    core.bfloat16, core.float16 = BF16, FLOAT16
    core.eval = lambda *values: events.append(('eval',) + values)
    core.default_device = lambda: types.SimpleNamespace(type=FakeDeviceKind(device_kind))
    core.device_info = lambda: {'device_name': 'Apple M2 Ultra', 'architecture': 'arm64'}
    core.array = lambda values: ('mx.array', values)
    utils.tree_flatten = fake_tree_flatten
    mlx.core = core
    mlx.utils = utils
    numpy = types.ModuleType('numpy')
    def fake_frombuffer(data, dtype=None):
        if dtype != '<i2':
            raise ValueError(f'fake frombuffer got dtype {dtype!r}')
        return FakeSamples(bytes(data))
    numpy.frombuffer = fake_frombuffer
    numpy.float32 = 'float32'
    sentencepiece = types.ModuleType('sentencepiece')

    class SentencePieceProcessor:
        def __init__(self, model_file):
            events.append(('tokenizer', model_file))

    sentencepiece.SentencePieceProcessor = SentencePieceProcessor
    haas = types.ModuleType('haas_mlx')
    model_module = types.ModuleType('haas_mlx.model')
    longform = types.ModuleType('haas_mlx.longform')

    class HaasModel(FakeHaasModel):
        @staticmethod
        def load(model_dir):
            events.append(('load', str(model_dir)))
            return HaasModel(FAKE_PARAMETER_TREE(), events)

    def transcribe_text(model, waveform, sp):
        events.append(('transcribe_text', waveform))
        return 'upstream-text'

    model_module.HaasModel = HaasModel
    longform.transcribe_text = transcribe_text
    haas.model, haas.longform = model_module, longform
    modules = {'mlx': mlx, 'mlx.core': core, 'mlx.metal': metal, 'mlx.utils': utils,
               'numpy': numpy, 'sentencepiece': sentencepiece, 'haas_mlx': haas,
               'haas_mlx.model': model_module, 'haas_mlx.longform': longform}
    return modules, metal


class ResidentProtocolTest(unittest.TestCase):
    """The pure protocol boundary: fake callable, StringIO in/out, no MLX."""

    def drive(self, messages):
        calls, output = [], io.StringIO()
        source = io.StringIO(''.join(json.dumps(m, ensure_ascii=False) + '\n' for m in messages))

        def transcribe(pcm):
            calls.append(pcm)
            return f'text-{len(calls)}-{len(pcm)}'

        code = resident.ResidentSession(transcribe, MLX_READY, source=source, sink=output).run()
        return code, [json.loads(line) for line in output.getvalue().splitlines()], calls

    def test_ready_identity_and_successive_requests(self):
        first, second, third = b'\x01\x00' * 50, b'\x02\x00' * 25, b'\xff\xff' * 7
        code, lines, calls = self.drive([chunk('a', 0, first), chunk('a', 1, second), terminal('a'),
                                         chunk('b', 0, third), terminal('b')])
        self.assertEqual(code, resident.EXIT_OK)
        self.assertEqual(calls, [first + second, third])  # exact reassembly, one call per request
        self.assertEqual(lines[0]['type'], 'ready')
        self.assertEqual(lines[0]['runtime'], 'haas-mlx')
        self.assertNotEqual(lines[0]['protocol'], 4)          # never the speech.cpp protocol
        self.assertNotIn('threads', lines[0].get('model', {}))  # never a thread resolution claim
        ends = [line for line in lines if line['type'] == 'end']
        self.assertEqual([end['stop'] for end in ends], ['complete', 'complete'])
        self.assertEqual([end['id'] for end in ends], ['a', 'b'])
        self.assertEqual([end['text'] for end in ends],
                         [f'text-1-{len(first + second)}', f'text-2-{len(third)}'])

    def test_rejects_bad_sequences(self):
        for messages in [[chunk('a', 1, b'\x00\x00')],
                         [chunk('a', 0, b'\x00\x00'), chunk('a', 2, b'\x00\x00')]]:
            code, lines, calls = self.drive(messages)
            self.assertEqual(code, resident.EXIT_PROTOCOL)
            self.assertEqual(calls, [])
            self.assertEqual(lines[-1]['type'], 'error')

    def test_rejects_mismatched_request_id(self):
        code, lines, calls = self.drive([chunk('a', 0, b'\x00\x00'), chunk('b', 1, b'\x00\x00')])
        self.assertEqual(code, resident.EXIT_PROTOCOL)
        self.assertEqual(calls, [])
        self.assertEqual(lines[-1]['type'], 'error')
        code, _, _ = self.drive([chunk('a', 0, b'\x00\x00'), terminal('b')])
        self.assertEqual(code, resident.EXIT_PROTOCOL)

    def test_rejects_bad_pcm(self):
        for payload in [base64.b64encode(b'\x01\x02\x03').decode(),  # odd byte count
                        '',                                          # empty
                        '###not-base64###']:                         # not strict base64
            code, lines, calls = self.drive([{'type': 'chunk', 'id': 'a', 'seq': 0, 'pcm': payload}])
            self.assertEqual(code, resident.EXIT_PROTOCOL, payload)
            self.assertEqual(calls, [])
            self.assertEqual(lines[-1]['type'], 'error')

    def test_rejects_unsupported_options(self):
        cases = [[chunk('a', 0, b'\x00\x00'), terminal('a', timestamps=True)],
                 [chunk('a', 0, b'\x00\x00'), terminal('a', sample_rate=44100)],
                 [chunk('a', 0, b'\x00\x00'), terminal('a', lang='ja')],
                 [dict(chunk('a', 0, b'\x00\x00'), extra=1), terminal('a')],
                 [{'type': 'flush', 'id': 'a'}],
                 [terminal('a')]]
        for messages in cases:
            code, lines, calls = self.drive(messages)
            self.assertEqual(code, resident.EXIT_PROTOCOL, messages)
            self.assertEqual(calls, [])
            self.assertEqual(lines[-1]['type'], 'error')

    def test_eof_while_idle_exits_cleanly_and_abandoned_request_is_rejected(self):
        code, lines, _ = self.drive([])
        self.assertEqual(code, resident.EXIT_OK)
        self.assertEqual([line['type'] for line in lines], ['ready'])
        code, lines, calls = self.drive([chunk('a', 0, b'\x00\x00' * 10)])
        self.assertEqual(code, resident.EXIT_ABANDONED)
        self.assertEqual(calls, [])
        self.assertEqual(lines[-1]['type'], 'error')

    def test_malformed_traffic_fails_explicitly(self):
        output = io.StringIO()
        calls = []
        session = resident.ResidentSession(lambda pcm: calls.append(pcm) or 'x', MLX_READY,
                                           source=io.StringIO('not json\n'), sink=output)
        code = session.run()
        self.assertEqual(code, resident.EXIT_PROTOCOL)
        self.assertEqual(calls, [])
        self.assertEqual(json.loads(output.getvalue().splitlines()[1])['type'], 'error')


class LoadEngineTest(unittest.TestCase):
    """load_engine identity and startup behavior against fake in-process modules."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.model_dir = Path(self._temporary.name) / 'model'
        self.model_dir.mkdir()
        (self.model_dir / 'tokenizer.model').write_bytes(b'tokenizer')

    def load(self, events, device_kind='gpu'):
        modules, metal = fake_mlx_modules(events, device_kind)
        versions = {'mlx': '0.32.4', 'mlx-metal': '0.32.4'}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(precision, 'distribution_version', versions.get), \
                mock.patch.multiple(precision, PINNED_PARAMETER_TENSORS=FAKE_TENSORS,
                                    PINNED_PARAMETER_ELEMENTS=FAKE_ELEMENTS):
            transcribe, identity = resident.load_engine(self.model_dir)
        return transcribe, identity, metal

    def test_identity_reports_the_actual_device_versions_and_cast(self):
        events = []
        transcribe, identity, metal = self.load(events)
        self.assertFalse(hasattr(metal, 'get_device_name'))  # a wrong API name cannot pass
        self.assertEqual(identity['runtime'], 'haas-mlx')
        self.assertEqual(identity['protocol'], resident.PROTOCOL)
        self.assertEqual(identity['mlx'], {'version': '0.32.4', 'mlx_metal_version': '0.32.4',
                                           'device': 'gpu', 'device_name': 'Apple M2 Ultra'})
        self.assertEqual(identity['python'],
                         {'executable': sys.executable, 'version': platform.python_version(),
                          'machine': platform.machine()})
        self.assertEqual(identity['model']['path'], str(self.model_dir.resolve()))
        cast = identity['model']['precision']
        self.assertEqual(cast['profile'], 'fp16-parameters')
        self.assertTrue(cast['cast'])
        self.assertEqual(cast['source_dtype'], 'bfloat16')
        self.assertEqual(cast['parameter_dtype'], 'float16')
        self.assertIsNone(cast['master_dtype'])  # honest: no BF16 master copy exists
        self.assertEqual(cast['tensor_count'], FAKE_TENSORS)
        self.assertEqual(cast['element_count'], FAKE_ELEMENTS)
        self.assertEqual(cast['shape_sample'][0], ['embed.weight', [4609, 1024]])
        kinds = [event[0] for event in events]
        self.assertLess(kinds.index('load'), kinds.index('set_dtype'))   # cast after the load
        self.assertLess(kinds.index('set_dtype'), kinds.index('eval'))   # eval after the cast
        evaluated = events[kinds.index('eval')][1]
        self.assertEqual({leaf.dtype for _, leaf in fake_tree_flatten(evaluated)}, {FLOAT16})
        self.assertIn(('tokenizer', str(self.model_dir / 'tokenizer.model')), events)
        self.assertNotIn('transcribe_text', [event[0] for event in events])  # no startup inference
        self.assertEqual(transcribe(b'\x01\x00\x00\xff'), 'upstream-text')
        waveform = [event[1] for event in events if event[0] == 'transcribe_text'][0]
        self.assertEqual(waveform, ('mx.array', [1 / 32768.0, -256 / 32768.0]))

    def test_a_cpu_default_device_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, 'GPU'):
            self.load([], device_kind='cpu')

    def test_a_tree_that_is_not_the_pinned_model_is_refused(self):
        modules, _ = fake_mlx_modules([])
        versions = {'mlx': '0.32.4', 'mlx-metal': '0.32.4'}
        with mock.patch.dict(sys.modules, modules), \
                mock.patch.object(precision, 'distribution_version', versions.get):
            # the real pinned totals apply, so the small fake tree is refused
            # and there is no alternate code path when validation fails
            with self.assertRaisesRegex(precision.PrecisionError, 'pinned'):
                resident.load_engine(self.model_dir)


class PrecisionCastTest(unittest.TestCase):
    """The shared strict cast helper on symbolic trees; no MLX, no numpy."""

    def cast(self, model, pinned=True):
        fake_mx = types.SimpleNamespace(bfloat16=BF16, float16=FLOAT16)
        expected = (FAKE_TENSORS, FAKE_ELEMENTS) if pinned else \
            (precision.PINNED_PARAMETER_TENSORS, precision.PINNED_PARAMETER_ELEMENTS)
        return precision.cast_parameters_to_float16(
            model, fake_mx, fake_tree_flatten,
            expected_tensors=expected[0], expected_elements=expected[1])

    def test_casts_every_leaf_and_reports_honest_counts(self):
        model = FakeHaasModel(FAKE_PARAMETER_TREE())
        profile = self.cast(model)
        self.assertEqual(profile, {
            'profile': 'fp16-parameters', 'cast': True, 'source_dtype': 'bfloat16',
            'parameter_dtype': 'float16', 'master_dtype': None,
            'arithmetic': precision.ARITHMETIC_NOTE,
            'tensor_count': FAKE_TENSORS, 'element_count': FAKE_ELEMENTS,
            'shape_sample': [['embed.weight', [4609, 1024]],
                             ['encoder.layers.0.proj', [512, 512]],
                             ['encoder.layers.1.proj', [64]]]})
        leaves = fake_tree_flatten(model.tree)
        self.assertEqual([leaf.dtype for _, leaf in leaves], [FLOAT16] * FAKE_TENSORS)
        self.assertEqual([name for name, _ in leaves],
                         [name for name, _ in fake_tree_flatten(FAKE_PARAMETER_TREE())])

    def test_refuses_a_leaf_that_is_not_bf16(self):
        tree = FAKE_PARAMETER_TREE()
        tree['encoder']['layers']['1']['proj'] = FakeLeaf('float32', [64])
        with self.assertRaisesRegex(precision.PrecisionError, 'bfloat16'):
            self.cast(FakeHaasModel(tree))

    def test_refuses_a_tree_that_is_not_the_pinned_model(self):
        with self.assertRaisesRegex(precision.PrecisionError, 'pinned'):
            self.cast(FakeHaasModel(FAKE_PARAMETER_TREE()), pinned=False)
        self.assertEqual((precision.PINNED_PARAMETER_TENSORS,
                          precision.PINNED_PARAMETER_ELEMENTS), (731, 615871520))

    def test_refuses_a_partial_reshaping_or_renaming_cast(self):
        class Partial(FakeHaasModel):
            def set_dtype(self, dtype):
                super().set_dtype(dtype)
                # one leaf keeps the source dtype, as if the predicate skipped it
                self.tree['encoder']['layers']['2']['proj'] = FakeLeaf(BF16, [128, 128])

        class Reshaping(FakeHaasModel):
            def set_dtype(self, dtype):
                super().set_dtype(dtype)
                self.tree['embed']['weight'] = FakeLeaf(dtype, [4609, 1023])

        class Renaming(FakeHaasModel):
            def set_dtype(self, dtype):
                super().set_dtype(dtype)
                proj = self.tree['encoder']['layers']['2'].pop('proj')
                self.tree['encoder']['layers']['2']['projx'] = proj

        for model, pattern in [(Partial(FAKE_PARAMETER_TREE()), 'float16'),
                               (Reshaping(FAKE_PARAMETER_TREE()), 'shape'),
                               (Renaming(FAKE_PARAMETER_TREE()), 'names')]:
            with self.assertRaisesRegex(precision.PrecisionError, pattern):
                self.cast(model)

    def test_refuses_an_empty_tree(self):
        with self.assertRaisesRegex(precision.PrecisionError, 'empty'):
            self.cast(FakeHaasModel({}))


class DriverTest(unittest.TestCase):
    """The driver against fake children and a fake clock; no subprocesses."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary.cleanup)
        self.root = Path(self._temporary.name)
        self.events = []
        self.clock = FakeClock(self.events)
        self.pcm_by_file = {}

    def make_records(self, count, sizes):
        records = []
        for index in range(count):
            data = bytes([index % 256]) * sizes[index]
            name = f'clip-{index}.wav'
            self.pcm_by_file[name] = data
            records.append({'file': name, 'stratum': '6-10s', 'samples': len(data) // 2,
                            'seconds': 1.0 + index, 'pcm_path': f'unused-{name}',
                            'pcm_sha256': hashlib.sha256(data).hexdigest()})
        return records

    def read_pcm(self, record):
        self.events.append(('pcm', record['file']))
        return self.pcm_by_file[record['file']]

    def run_round(self, child, records, round_index=0, side='native', receive_timeout=5.0):
        directory = self.root / f'{side}-{round_index}'
        result = pilot.run_side_round(
            side, round_index, directory, ['fake-worker'], {'FAKE': '1'},
            pilot.verify_native_ready if side == 'native' else pilot.verify_mlx_ready,
            {'identity': 'test'}, records, records, conditions={'rounds': 3},
            environment_note={'set': {}}, spawn=child.spawn, clock=self.clock,
            shutdown=recording_shutdown, read_pcm=self.read_pcm,
            receive_timeout=receive_timeout)
        self.assertTrue((directory / 'result.json').exists())
        return result, directory

    def assert_pipes_closed(self, process):
        """Cleanup closed both fake pipe ends and the reader drained to EOF."""
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)
        self.assertTrue(process.stdout.read_to_eof)

    def assert_request_boundaries(self, request_ids):
        clocks = [i for i, event in enumerate(self.events) if event[0] == 'clock']
        for request_id in request_ids:
            sends = [i for i, event in enumerate(self.events)
                     if event[0] == 'send' and event[2] == request_id]
            first_send = sends[0]
            pcm_read = max(i for i, event in enumerate(self.events)
                           if event[0] == 'pcm' and i < first_send)
            begin = min(i for i in clocks if i > pcm_read)
            self.assertLess(pcm_read, begin)   # PCM read and verified before the clock starts
            self.assertLess(begin, first_send)
            self.assertEqual(self.events[begin + 1], ('send', 'chunk', request_id, 0))
            transcribe = [i for i, event in enumerate(self.events)
                          if event[0] == 'send' and event[1] == 'transcribe' and event[2] == request_id][0]
            self.assertEqual(min(i for i in clocks if i > transcribe), transcribe + 1)

    def test_native_round_boundaries_chunking_and_order(self):
        records = self.make_records(3, [40, 32768 * 2 + 100, 120])
        child = FakeChild(self.events, NATIVE_READY, progress=True, clock=self.clock)
        result, _ = self.run_round(child, records, round_index=2)
        self.assertEqual([row['file'] for row in result['warmups']],
                         ['clip-2.wav', 'clip-1.wav', 'clip-0.wav'])  # longest first
        self.assertEqual(result['first_request']['file'], 'clip-2.wav')
        self.assertEqual([row['file'] for row in result['measured']],
                         ['clip-2.wav', 'clip-0.wav', 'clip-1.wav'])  # shift 2 * 7 % 3 == 2
        self.assertTrue(all(abs(row['seconds'] - 1.5) < 1e-9
                            for row in result['warmups'] + result['measured']))
        self.assertEqual(result['ready'], NATIVE_READY)
        self.assertEqual(result['memory']['peak_memory_footprint'], 4242)
        self.assertEqual(result['command'], ['fake-worker'])
        self.assertEqual(result['spawn_prefix'][:3], ['/usr/bin/time', '-l', '-o'])
        big = next(request for request in child.requests if len(request['sizes']) == 3)
        self.assertEqual(big['sizes'], [32768, 32768, 100])  # chunks of up to exactly 32768 bytes
        self.assertEqual(big['seqs'], [0, 1, 2])
        self.assertEqual(big['pcm'], self.pcm_by_file['clip-1.wav'])
        for message in child.terminals:
            self.assertEqual(message['sample_rate'], 16000)
            self.assertIs(message['timestamps'], False)
        self.assert_request_boundaries(['warm-0', 'warm-1', 'warm-2', '0', '1', '2'])
        self.assert_pipes_closed(child.process)
        self.assertEqual(self.events.count(('stdin_closed',)), 1)  # one EOF, not two closes

    def test_mlx_round_uses_the_same_outer_request_function(self):
        records = self.make_records(2, [60, 80])
        child = FakeChild(self.events, MLX_READY, clock=self.clock)  # no progress, ever
        result, _ = self.run_round(child, records, round_index=0, side='mlx')
        self.assertEqual(len(child.requests), 4)
        self.assertEqual(result['side'], 'mlx')
        self.assertTrue(all(abs(row['seconds'] - 1.5) < 1e-9 for row in result['measured']))
        self.assert_request_boundaries(['warm-0', 'warm-1', '0', '1'])

    def test_ready_validators_pin_placement_and_versions(self):
        pilot.verify_native_ready(NATIVE_READY)
        pilot.verify_mlx_ready(MLX_READY)
        for bad in [{'type': 'ready', 'protocol': 4},
                    {'type': 'ready', 'protocol': 4,
                     'model': {'device': 'MTL0', 'threads': 5}, 'defaults': {'decode_on_cpu': True}},
                    {'type': 'ready', 'protocol': 3, 'model': {'device': 'MTL0', 'threads': 4},
                     'defaults': {'decode_on_cpu': True}},
                    {'type': 'ready', 'protocol': 4, 'model': {'device': 'cpu', 'threads': 4},
                     'defaults': {'decode_on_cpu': True}},
                    {'type': 'ready', 'protocol': 4, 'model': {'device': 'MTL0', 'threads': 4},
                     'defaults': {'decode_on_cpu': False}},
                    {'type': 'ready', 'protocol': 4, 'model': {'device': 'MTL0', 'threads': 4}}]:
            with self.assertRaises(RuntimeError):
                pilot.verify_native_ready(bad)
        for bad in [dict(MLX_READY, protocol=4),
                    dict(MLX_READY, model={'threads': 4}),
                    dict(MLX_READY, runtime='speech'),
                    {key: value for key, value in MLX_READY.items() if key != 'mlx'},
                    dict(MLX_READY, mlx={**MLX_READY['mlx'], 'version': '0.32.5'}),
                    dict(MLX_READY, mlx={**MLX_READY['mlx'], 'mlx_metal_version': '0.32.3'}),
                    dict(MLX_READY, mlx={**MLX_READY['mlx'], 'device': 'cpu'}),
                    dict(MLX_READY, mlx={**MLX_READY['mlx'], 'device_name': 'Apple M3 Pro'}),
                    dict(MLX_READY, python={**MLX_READY['python'], 'version': '3.12.10'}),
                    dict(MLX_READY, python={**MLX_READY['python'], 'machine': 'x86_64'}),
                    dict(MLX_READY, model={}),  # an old BF16 resident reports no profile
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'parameter_dtype': 'bfloat16'}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'source_dtype': 'float32'}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'tensor_count': 730}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'element_count': 615871521}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'cast': False}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'master_dtype': 'bfloat16'}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'profile': 'bf16-parameters'}}),
                    dict(MLX_READY, model={'precision': {
                        key: value for key, value in FP16_PROFILE.items() if key != 'shape_sample'}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE, 'shape_sample': []}}),
                    dict(MLX_READY, model={'precision': {**FP16_PROFILE,
                                                         'shape_sample': [['w', [0]]]}})]:
            with self.assertRaises(RuntimeError):
                pilot.verify_mlx_ready(bad)

    def test_error_message_cleans_up_without_result(self):
        records = self.make_records(1, [40])
        child = FakeChild(self.events, NATIVE_READY, error=True, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'Recognition failed'):
            self.run_round(child, records)
        self.assertEqual(child.process.shutdowns, 1)  # the child was cleaned up
        self.assert_pipes_closed(child.process)
        self.assertFalse((self.root / 'native-0' / 'result.json').exists())
        partial = json.loads((self.root / 'native-0' / 'partial.json').read_text())
        self.assertEqual(partial['active_request_id'], 'warm-0')
        self.assertEqual(partial['warmups_completed'], [])
        self.assertFalse(partial['timed_out'])

    def test_failure_retains_completed_rows_in_partial(self):
        records = self.make_records(2, [40, 60])
        child = FakeChild(self.events, NATIVE_READY, fail_after=1, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'Recognition failed'):
            self.run_round(child, records)
        partial = json.loads((self.root / 'native-0' / 'partial.json').read_text())
        self.assertEqual(partial['status'], 'failed')
        self.assertEqual(partial['ready'], NATIVE_READY)
        self.assertEqual(partial['active_request_id'], 'warm-1')
        self.assertEqual([row['file'] for row in partial['warmups_completed']], ['clip-1.wav'])
        self.assertEqual(partial['measured_completed'], [])
        self.assertIn('Recognition failed', partial['error'])
        self.assertFalse(partial['timed_out'])
        self.assertFalse((self.root / 'native-0' / 'result.json').exists())

    def test_watchdog_releases_a_blocked_stdin_write(self):
        records = self.make_records(1, [40])
        long_pcm = b'\x00\x01' * 131072  # beyond pipe capacity; the child never reads
        blocked_child = ('import json, sys, time\n'
                         'sys.stdout.write(json.dumps(%s) + "\\n")\n'
                         'sys.stdout.flush()\n'
                         'time.sleep(600)\n' % repr(NATIVE_READY))
        spawned = []

        def spawn(command, env, memory_log, stderr_log):
            process = pilot.default_spawn(command, env, memory_log, stderr_log)
            self.assertEqual(os.getpgid(process.pid), process.pid)      # its own group
            self.assertNotEqual(os.getpgid(process.pid), os.getpgrp())  # never the pilot's
            spawned.append(process)
            return process

        with self.assertRaisesRegex(RuntimeError, 'watchdog'):
            pilot.run_side_round(
                'native', 0, self.root / 'native-blocked',
                [sys.executable, '-c', blocked_child], {'PATH': '/usr/bin:/bin'},
                pilot.verify_native_ready, {'identity': 'test'}, records, records,
                conditions={'rounds': 3}, environment_note={'set': {}}, spawn=spawn,
                clock=self.clock, shutdown=pilot.shutdown_process,
                read_pcm=lambda record: long_pcm, receive_timeout=5.0, watchdog_seconds=1.0)
        process, = spawned
        self.assertIsNotNone(process.poll())  # killed, not a leftover
        self.assertTrue(process.stdin.closed)    # both real pipe handles were closed
        self.assertTrue(process.stdout.closed)
        partial = json.loads((self.root / 'native-blocked' / 'partial.json').read_text())
        self.assertTrue(partial['timed_out'])
        self.assertEqual(partial['ready'], NATIVE_READY)
        self.assertEqual(partial['active_request_id'], 'warm-0')
        self.assertIn('watchdog', partial['error'])

    def test_timeout_and_stdout_eof_clean_up(self):
        records = self.make_records(1, [40])
        silent = FakeChild(self.events, NATIVE_READY, silent=True)
        with self.assertRaisesRegex(RuntimeError, 'Timed out'):
            self.run_round(silent, records, receive_timeout=0.05)
        self.assertEqual(silent.process.shutdowns, 1)
        self.assert_pipes_closed(silent.process)
        self.assertFalse((self.root / 'native-0' / 'result.json').exists())
        closing = FakeChild(self.events, NATIVE_READY, close_after_ready=True)
        with self.assertRaisesRegex(RuntimeError, 'stdout closed'):
            self.run_round(closing, records, round_index=1, receive_timeout=5.0)  # fresh directory
        self.assertEqual(closing.process.shutdowns, 1)
        self.assert_pipes_closed(closing.process)
        self.assertFalse((self.root / 'native-1' / 'result.json').exists())

    def test_nonzero_child_exit_fails_after_measurements(self):
        records = self.make_records(1, [40])
        child = FakeChild(self.events, NATIVE_READY, exit_code=7, clock=self.clock)
        with self.assertRaisesRegex(RuntimeError, 'exited with failure'):
            self.run_round(child, records)
        self.assertEqual(len(child.requests), 2)  # warmup and measurement both completed
        self.assertFalse((self.root / 'native-0' / 'result.json').exists())
        partial = json.loads((self.root / 'native-0' / 'partial.json').read_text())
        self.assertEqual(len(partial['warmups_completed']), 1)
        self.assertEqual(len(partial['measured_completed']), 1)
        self.assertIsNone(partial['active_request_id'])
        self.assertFalse(partial['timed_out'])

    def fake_run_result(self, side, round_index):
        measured = []
        for index in range(9):
            native_text = 'same' if (round_index, index) not in {(0, 2), (2, 5)} else 'different'
            measured.append({'file': f'clip-{index}', 'stratum': '6-10s', 'audio_seconds': 10.0,
                             'seconds': 2.0 if side == 'mlx' else 4.0,
                             'text': 'same' if side == 'mlx' else native_text,
                             'request_id': str(index)})
        return {'side': side, 'round': round_index, 'ready': {}, 'identities': {},
                'load_seconds': 5.0, 'first_request': {'file': 'w0', 'seconds': 4.0},
                'warmups': [{'file': f'w{i}', 'seconds': 1.0} for i in range(9)],
                'warmup_count': 9, 'measured': measured,
                'memory': {'peak_memory_footprint': 4096}}

    def test_round_helpers_shift_and_side_order(self):
        self.assertEqual(pilot.round_order(0), ('mlx', 'native'))
        self.assertEqual(pilot.round_order(1), ('native', 'mlx'))
        self.assertEqual(pilot.round_order(2), ('mlx', 'native'))
        clips = list(range(9))
        self.assertEqual(pilot.shifted(clips, 0), clips)
        self.assertEqual(pilot.shifted(clips, 1), [7, 8, 0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(pilot.shifted(clips, 2), [5, 6, 7, 8, 0, 1, 2, 3, 4])

    def test_execute_rounds_ab_ba_ab_and_paired_counts(self):
        seen = []
        runs = pilot.execute_rounds(lambda side, round_index:
                                    seen.append((side, round_index))
                                    or self.fake_run_result(side, round_index))
        self.assertEqual(seen, [('mlx', 0), ('native', 0), ('native', 1), ('mlx', 1),
                                ('mlx', 2), ('native', 2)])  # AB/BA/AB
        self.assertEqual([len(run['measured']) for run in runs['mlx']], [9, 9, 9])
        with tempfile.TemporaryDirectory() as out:
            directory = Path(out)
            manifest = {'conditions': {'rounds': 3}, 'pins': {}}
            summary = pilot.summarize(directory, manifest, runs)
            self.assertEqual(summary['sides']['mlx']['measured_count'], 27)
            self.assertEqual(summary['sides']['native']['measured_count'], 27)
            self.assertEqual(summary['sides']['mlx']['latency']['count'], 9)  # nine file medians
            self.assertEqual(summary['sides']['mlx']['pooled_measured_latency']['count'], 27)
            self.assertEqual(summary['paired']['requests'], 27)
            self.assertEqual(summary['paired']['text_changed'], 2)
            self.assertEqual(summary['paired']['text_equal'], 25)
            self.assertAlmostEqual(summary['paired']['median_latency_ratio'], 0.5)
            changes = json.loads((directory / 'paired-text-changes.json').read_text())
            self.assertEqual({entry['round'] for entry in changes}, {0, 2})
            self.assertTrue((directory / 'summary.json').exists())

    def test_side_summary_primary_latency_is_nine_per_file_medians(self):
        def repeat_warm(side, round_index):
            result = self.fake_run_result(side, round_index)
            for row in result['measured']:
                row['seconds'] = (2.0 if side == 'mlx' else 4.0) + 0.5 * round_index
            for row in result['warmups']:
                row['seconds'] = 1.0 + 0.25 * round_index
            if round_index == 1:
                slow = next(row for row in result['measured'] if row['file'] == 'clip-3')
                slow['seconds'] += 57.0  # one isolated slow repetition among clip-3's three
            return result

        runs = pilot.execute_rounds(repeat_warm)
        summary = pilot.side_summary(runs['mlx'])
        medians = {row['file']: row['median_seconds'] for row in summary['per_input_medians']}
        self.assertEqual(len(medians), 9)
        self.assertEqual(medians['clip-3'], 3.0)  # the isolated 59.5s repetition drops out
        self.assertEqual(sorted(set(medians.values())), [2.5, 3.0])
        expected_primary = pilot.latency([{'audio_seconds': row['audio_seconds'],
                                           'seconds': row['median_seconds']}
                                          for row in summary['per_input_medians']])
        raw_rows = [row for run in runs['mlx'] for row in run['measured']]
        self.assertEqual(summary['latency'], expected_primary)  # p50/p90/p99/RTFx over nine medians
        self.assertEqual(summary['pooled_measured_latency'], pilot.latency(raw_rows))
        # a raw-27 primary would inherit the 59.5s outlier and differ from the median-based one
        self.assertNotEqual(summary['latency'], summary['pooled_measured_latency'])
        self.assertEqual(summary['latency']['count'], 9)
        self.assertEqual(summary['pooled_measured_latency']['count'], 27)
        self.assertEqual(summary['measured_count'], 27)
        uneven = [self.fake_run_result('mlx', r) for r in range(3)]
        uneven[1]['measured'] = [dict(row, file='clip-0') if row['file'] == 'clip-8' else row
                                 for row in uneven[1]['measured']]  # 27 rows, a 4/2 split
        with self.assertRaises(ValueError):
            pilot.side_summary(uneven)

    def write_subset(self, rows):
        subset = []
        for index, row in enumerate(rows):
            data = row.get('pcm', b'\x00\x01' * 4)
            (self.root / f'pcm-{index}.bin').write_bytes(data)
            samples = row.get('samples', len(data) // 2)
            record = {'file': row.get('file', f'clip-{index}.wav'),
                      'stratum': row.get('stratum', ['6-10s', '10-20s', '20-30s'][index // 10]),
                      'samples': samples,
                      'seconds': row.get('seconds', samples / 16000.0),
                      'pcm_path': str(self.root / f'pcm-{index}.bin')}
            if row.get('tamper'):
                record['pcm_sha256'] = hashlib.sha256(b'\x01' * 8).hexdigest()
            else:
                record['pcm_sha256'] = hashlib.sha256(data).hexdigest()
            if 'drop' in row:
                record.pop(row['drop'])
            subset.append(record)
        return subset

    def test_select_inputs_accepts_the_fixed_nine(self):
        rows = self.write_subset([{} for _ in range(30)])
        chosen = pilot.select_inputs({'subset': rows})
        self.assertEqual([row['file'] for row in chosen],
                         [rows[i]['file'] for i in pilot.SUBSET_INDEXES])
        strata = {}
        for row in chosen:
            strata.setdefault(row['stratum'], []).append(row['file'])
        self.assertEqual(sorted(len(names) for names in strata.values()), [3, 3, 3])

    def test_select_inputs_rejects_bad_corpora(self):
        good = self.write_subset([{} for _ in range(30)])
        cases = [{'subset': good[:29]},
                 {'subset': self.write_subset([{'file': 'clip-0.wav'} if i == 4 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'stratum': '10-20s'} if i == 9 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'tamper': True} if i == 10 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'drop': 'samples'} if i == 14 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'samples': 0} if i == 0 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'samples': 2.0} if i == 0 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'seconds': float('nan')} if i == 4 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'seconds': 9.99} if i == 4 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'seconds': 4 / 16000.0 + 0.01} if i == 4 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'pcm': b'\x00\x01' * 4 + b'\x00'} if i == 9 else {}
                                               for i in range(30)])},
                 {'subset': self.write_subset([{'samples': 5} if i == 10 else {}
                                               for i in range(30)])}]
        for assets in cases:
            with self.assertRaises(ValueError):
                pilot.select_inputs(assets)

    def test_verify_native_refuses_unpinned_builds(self):
        assets = {'models': [{'type': 'f16', 'path': 'unused', 'sha256': 'x' * 64,
                              'verified_sha256': 'x' * 64}]}
        good = {'performance_evaluation_eligible': True, 'source_verified': True,
                'source_commit': pilot.NATIVE_SOURCE_COMMIT,
                'ggml_commit': pilot.NATIVE_GGML_COMMIT}
        with self.assertRaises(ValueError):  # PGO training-only build
            pilot.verify_native({**good, 'performance_evaluation_eligible': False}, assets)
        with self.assertRaises(ValueError):  # wrong source pin
            pilot.verify_native({**good, 'source_commit': 'f' * 40}, assets)
        with self.assertRaises(ValueError):  # no single f16 row
            pilot.verify_native(good, {'models': []})
        with self.assertRaises(ValueError):  # f16 row SHA is not the accepted pin
            pilot.verify_native(good, assets)

    def test_verify_mlx_model_rejects_incomplete_manifests(self):
        entry = {'size': 10, 'sha256': 'a' * 64, 'verified': True}
        good = {'revision': pilot.MLX_MODEL_REVISION,
                'files': {name: dict(entry) for name in pilot.MLX_MODEL_FILES}}
        model_dir = self.root / 'model'
        model_dir.mkdir()
        for name in pilot.MLX_MODEL_FILES:
            (model_dir / name).write_bytes(b'0123456789')
        with tempfile.TemporaryDirectory() as folder:
            def check(manifest):
                path = Path(folder) / 'm.json'
                path.write_text(json.dumps(manifest))
                pilot.verify_mlx_model(path, model_dir)
            with self.assertRaises(ValueError):
                check({**good, 'revision': 'wrong'})
            with self.assertRaises(ValueError):
                check({**good, 'files': {k: v for k, v in good['files'].items()
                                         if k != 'config.json'}})
            incomplete = {**good, 'files': {**good['files'], 'config.json': {'size': 10}}}
            with self.assertRaises(ValueError):
                check(incomplete)
            unverified = {**good, 'files': {**good['files'],
                                            'config.json': {**entry, 'verified': False}}}
            with self.assertRaises(ValueError):
                check(unverified)

    def test_verify_mlx_model_rejects_a_substituted_config(self):
        model_dir = self.root / 'model'
        model_dir.mkdir()
        original = b'{"model_type": "haas"}\n'
        (model_dir / 'config.json').write_bytes(original)
        for name, data in [('model.safetensors', b'weights'), ('tokenizer.model', b'tokenizer')]:
            (model_dir / name).write_bytes(data)

        def manifest_for(config):
            return {'revision': pilot.MLX_MODEL_REVISION, 'files': {
                name: {'size': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'verified': True}
                for name, data in [('model.safetensors', b'weights'),
                                   ('tokenizer.model', b'tokenizer'), ('config.json', config)]}}

        pinned_blob = hashlib.sha1(b'blob %d\x00' % len(original) + original).hexdigest()
        pins = {'MLX_CONFIG_GIT_BLOB': pinned_blob,
                'MLX_WEIGHTS_SHA256': hashlib.sha256(b'weights').hexdigest(),
                'MLX_TOKENIZER_SHA256': hashlib.sha256(b'tokenizer').hexdigest()}
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest_for(original)))
        with mock.patch.multiple(pilot, **pins):
            report = pilot.verify_mlx_model(path, model_dir)
        self.assertEqual(report['files']['config.json']['git_blob_sha1'], pinned_blob)
        substituted = b'{"model_type": "haas", "tie_words": true}\n'
        (model_dir / 'config.json').write_bytes(substituted)
        path.write_text(json.dumps(manifest_for(substituted)))  # the manifest now agrees
        with mock.patch.multiple(pilot, **pins):
            with self.assertRaisesRegex(ValueError, 'Git blob'):
                pilot.verify_mlx_model(path, model_dir)

    def test_pure_helpers(self):
        hello = self.root / 'hello.txt'
        hello.write_bytes(b'hello\n')
        self.assertEqual(pilot.git_blob_sha1(hello),
                         hashlib.sha1(b'blob 6\x00hello\n').hexdigest())
        memory = pilot.parse_memory_log('  100  maximum resident set size\n'
                                        '  4242  peak memory footprint\n')
        self.assertEqual(memory, {'maximum_resident_set_size': 100,
                                  'peak_memory_footprint': 4242})
        with mock.patch.dict(os.environ, {}, clear=True):
            pilot.require_no_mlx_tweaks()
            env = pilot.mlx_environment('/port')
            self.assertEqual(env['PYTHONPATH'], '/port/src')
            self.assertEqual(env['HF_HUB_OFFLINE'], '1')
            self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')
        with mock.patch.dict(os.environ, {'MLX_EXPERIMENT': '1'}, clear=True):
            with self.assertRaises(ValueError):
                pilot.require_no_mlx_tweaks()
            self.assertNotIn('MLX_EXPERIMENT', pilot.mlx_environment('/port'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
