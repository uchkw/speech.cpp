#!/usr/bin/env python3
"""Experiment-only persistent Haas MLX resident worker for the timing pilot.

Protocol: one request at a time over JSON Lines (stdin -> stdout), each
message flushed, with failure details on stderr:

    worker -> {"type": "ready", ...}                      actual MLX and Python identity
    driver -> {"type": "chunk", "id": "0", "seq": 0, "pcm": "<base64 S16LE>"}
    driver -> {"type": "chunk", "id": "0", "seq": 1, "pcm": "<base64 S16LE>"}
    driver -> {"type": "transcribe", "id": "0", "sample_rate": 16000, "timestamps": false}
    worker -> {"type": "end", "id": "0", "stop": "complete", "text": "..."}

This is deliberately not a speech.cpp worker: the ready message never claims
protocol 4 and never resolves CPU threads. Startup loads the model and
tokenizer exactly once, evaluates the parameters, and does no warmup, no
download, no compilation pass, and no weight or precision changes. Clean EOF
while idle exits 0; EOF with an open request is rejected as abandoned.

ResidentSession is a pure standard-library protocol boundary around a
supplied callable, so synthetic checks need no MLX, numpy or audio;
load_engine provides the real transcription callable.
"""

from __future__ import annotations

import argparse
import base64
import json
import platform
import sys
from importlib.metadata import version as distribution_version
from pathlib import Path

RUNTIME = 'haas-mlx'
PROTOCOL = 'haas-mlx-pilot-1'
EXIT_OK = 0
EXIT_PROTOCOL = 2   # invalid driver traffic
EXIT_ENGINE = 3     # model load or transcription failure
EXIT_ABANDONED = 4  # EOF with an open request
ALLOWED_CHUNK_KEYS = frozenset({'type', 'id', 'seq', 'pcm'})
ALLOWED_TRANSCRIBE_KEYS = frozenset({'type', 'id', 'sample_rate', 'timestamps'})


class ProtocolError(Exception):
    """Invalid driver traffic; reported to the driver and stderr, then fatal."""


class RequestState:
    """One open request: its id, the next expected chunk seq and the PCM buffers."""

    def __init__(self, request_id):
        self.request_id = request_id
        self.next_seq = 0
        self.chunks = []


def require_request_id(message):
    request_id = message.get('id')
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError('id must be a nonempty string')
    return request_id


def decode_chunk(message, state):
    """Validate one chunk message against the open request and return its PCM bytes."""
    extra = sorted(set(message) - ALLOWED_CHUNK_KEYS)
    if extra:
        raise ProtocolError(f'unsupported chunk fields: {extra}')
    request_id = require_request_id(message)
    if request_id != state.request_id:
        raise ProtocolError(
            f'chunk id {request_id!r} does not match the open request {state.request_id!r}')
    seq = message.get('seq')
    if type(seq) is not int or seq != state.next_seq:
        raise ProtocolError(f'chunk seq must be {state.next_seq}, got {seq!r}')
    payload = message.get('pcm')
    if not isinstance(payload, str) or not payload:
        raise ProtocolError('chunk pcm must be a nonempty base64 string')
    try:
        data = base64.b64decode(payload, validate=True)
    except (ValueError, TypeError) as error:
        raise ProtocolError(f'chunk pcm is not strict base64: {error}')
    if not data:
        raise ProtocolError('chunk pcm decoded to empty PCM')
    if len(data) % 2:
        raise ProtocolError('chunk pcm has an odd byte count; S16LE needs whole samples')
    return data


def decode_terminal(message, state):
    """Validate the terminal request and return the complete request PCM."""
    extra = sorted(set(message) - ALLOWED_TRANSCRIBE_KEYS)
    if extra:
        raise ProtocolError(f'unsupported transcribe fields: {extra}')
    request_id = require_request_id(message)
    if request_id != state.request_id:
        raise ProtocolError(
            f'transcribe id {request_id!r} does not match the open request {state.request_id!r}')
    sample_rate = message.get('sample_rate')
    if type(sample_rate) is not int or sample_rate != 16000:
        raise ProtocolError('transcribe sample_rate must be 16000')
    if message.get('timestamps') is not False:
        raise ProtocolError('timestamps are not supported by this resident; request timestamps=false')
    if not state.chunks:
        raise ProtocolError('transcribe without any PCM chunks')
    return b''.join(state.chunks)


class ResidentSession:
    """The pure protocol boundary: ready, then one request at a time until EOF."""

    def __init__(self, transcribe, identity, source=None, sink=None):
        self.transcribe = transcribe  # callable(complete S16LE PCM bytes) -> text
        self.identity = identity      # honest runtime identity merged into ready
        self.source = sys.stdin if source is None else source
        self.sink = sys.stdout if sink is None else sink

    def emit(self, message):
        self.sink.write(json.dumps(message, ensure_ascii=False) + '\n')
        self.sink.flush()

    def fail(self, message, request_id, code):
        print(f'resident: {message}', file=sys.stderr, flush=True)
        self.emit({'type': 'error', 'id': request_id, 'stop': 'error', 'message': message})
        return code

    def run(self):
        self.emit({'type': 'ready', **self.identity})
        state = None
        for line in self.source:
            try:
                message = json.loads(line)
            except ValueError as error:
                return self.fail(f'malformed JSON message: {error}',
                                 state.request_id if state else None, EXIT_PROTOCOL)
            if not isinstance(message, dict):
                return self.fail(f'message must be a JSON object, got {type(message).__name__}',
                                 state.request_id if state else None, EXIT_PROTOCOL)
            try:
                kind = message.get('type')
                if kind == 'chunk':
                    if state is None:
                        state = RequestState(require_request_id(message))
                    state.chunks.append(decode_chunk(message, state))
                    state.next_seq += 1
                elif kind == 'transcribe':
                    if state is None:
                        raise ProtocolError('transcribe for a request that was never started')
                    request_id = state.request_id
                    pcm = decode_terminal(message, state)
                    state = None  # release the request buffers around the terminal result
                    try:
                        text = self.transcribe(pcm)
                    except Exception as error:
                        return self.fail(f'transcription failed: {error}', request_id, EXIT_ENGINE)
                    self.emit({'type': 'end', 'id': request_id, 'stop': 'complete', 'text': text})
                else:
                    raise ProtocolError(f'unsupported message type: {kind!r}')
            except ProtocolError as error:
                return self.fail(str(error), state.request_id if state else None, EXIT_PROTOCOL)
        if state is not None:
            return self.fail(f'request {state.request_id!r} abandoned at EOF '
                             f'after {state.next_seq} chunk(s)', state.request_id, EXIT_ABANDONED)
        return EXIT_OK


def load_engine(model_dir):
    """Load model and tokenizer exactly once; return (transcribe, identity).

    All MLX-side imports stay here so the protocol boundary above remains
    importable without MLX, numpy or sentencepiece.
    """
    import mlx.core as mx
    import sentencepiece as spm

    from haas_mlx.model import HaasModel

    if not mx.metal.is_available():
        raise RuntimeError('MLX reports no Metal device; this resident requires a real GPU.')
    model = HaasModel.load(model_dir)
    sp = spm.SentencePieceProcessor(model_file=str(Path(model_dir) / 'tokenizer.model'))
    mx.eval(model.parameters())

    device = mx.default_device()
    kind = getattr(device, 'type', device)
    kind = getattr(kind, 'name', None) or str(kind)
    if 'gpu' not in str(kind).lower():
        raise RuntimeError(f'this resident requires a real MLX GPU device; the default is {kind}')
    device_name = mx.device_info()['device_name']

    import numpy as np
    from haas_mlx.longform import transcribe_text

    def transcribe(pcm):
        """One request: complete S16LE PCM -> text, via the upstream longform
        path (its native quiet-point chunking, per-chunk cache clearing and
        final detokenization are preserved untouched)."""
        waveform = mx.array(np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0)
        return transcribe_text(model, waveform, sp)

    identity = {
        'runtime': RUNTIME,
        'protocol': PROTOCOL,
        'mlx': {'version': distribution_version('mlx'),
                'mlx_metal_version': distribution_version('mlx-metal'),
                'device': str(kind), 'device_name': device_name},
        'python': {'executable': sys.executable, 'version': platform.python_version(),
                   'machine': platform.machine()},
        'model': {'path': str(Path(model_dir).resolve())},
    }
    return transcribe, identity


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Experiment-only persistent Haas MLX resident worker '
                    '(JSON Lines protocol; see the module docstring).')
    parser.add_argument('--model', required=True, help='converted Haas MLX model directory')
    args = parser.parse_args(argv)
    try:
        transcribe, identity = load_engine(args.model)
    except Exception as error:
        print(f'resident: cannot start: {error}', file=sys.stderr, flush=True)
        return EXIT_ENGINE
    return ResidentSession(transcribe, identity).run()


if __name__ == '__main__':
    sys.exit(main())
