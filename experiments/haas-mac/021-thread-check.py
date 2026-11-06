"""Check the experimental thread selection through the C API without running inference.

Usage: python3 021-thread-check.py LIBSPEECH HAAS_MODEL [--control-model MODEL]
This loads weights and initializes devices; run only in the coordinator's compute slot.
"""
import argparse
import ctypes as c
import json
import os
from pathlib import Path


FLAG = 'SPEECH_HAAS_AUTO_THREADS'


class Api:
    def __init__(self, path):
        self.lib = c.CDLL(str(Path(path).resolve()))
        pointer = c.POINTER(c.c_void_p)
        signatures = {
            'speech_load_params_new': (c.c_int, [pointer]),
            'speech_load_params_free': (None, [c.c_void_p]),
            'speech_load_params_set_device': (c.c_int, [c.c_void_p, c.c_char_p]),
            'speech_load_params_set_threads': (c.c_int, [c.c_void_p, c.c_int]),
            'speech_model_load': (c.c_int, [c.c_char_p, c.c_void_p, pointer]),
            'speech_model_free': (None, [c.c_void_p]),
            'speech_model_get_info': (c.c_int, [c.c_void_p, pointer]),
            'speech_model_info_open': (c.c_int, [c.c_char_p, pointer]),
            'speech_model_info_free': (None, [c.c_void_p]),
            'speech_model_info_threads': (c.c_int, [c.c_void_p]),
            'speech_model_info_json': (c.c_char_p, [c.c_void_p]),
            'speech_last_error': (c.c_char_p, []),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, arguments

    def ok(self, status):
        if status != 0:
            raise RuntimeError(self.lib.speech_last_error().decode('utf-8'))

    def metadata(self, path):
        info = c.c_void_p()
        self.ok(self.lib.speech_model_info_open(os.fsencode(path), c.byref(info)))
        try:
            data = json.loads(self.lib.speech_model_info_json(info))
            if self.lib.speech_model_info_threads(info) != 0 or 'threads' in data:
                raise AssertionError('Metadata-only information reports loaded threads.')
            return data
        finally:
            self.lib.speech_model_info_free(info)

    def load(self, path, device='gpu', threads=0, null_params=False, refused=False):
        params, model, info = c.c_void_p(), c.c_void_p(), c.c_void_p()
        try:
            if not null_params:
                self.ok(self.lib.speech_load_params_new(c.byref(params)))
                self.ok(self.lib.speech_load_params_set_device(params, device.encode('utf-8')))
                if threads:
                    self.ok(self.lib.speech_load_params_set_threads(params, threads))
            status = self.lib.speech_model_load(os.fsencode(path), params, c.byref(model))
            if refused:
                message = (self.lib.speech_last_error() or b'').decode('utf-8')
                if status != -1 or model.value or FLAG not in message:
                    raise AssertionError(f'The invalid selector was not refused: {status}, {message}')
                return {'status': status, 'error': message}
            self.ok(status)
            self.ok(self.lib.speech_model_get_info(model, c.byref(info)))
            data = json.loads(self.lib.speech_model_info_json(info))
            count = self.lib.speech_model_info_threads(info)
            if data['threads'] != count:
                raise AssertionError('The C getter and JSON disagree on the effective thread count.')
            return {'device': data['device'], 'threads': count, 'name': data['name']}
        finally:
            self.lib.speech_model_info_free(info)
            self.lib.speech_model_free(model)
            self.lib.speech_load_params_free(params)


def check(args):
    api, results = Api(args.library), []
    os.environ[FLAG] = 'invalid'
    metadata = api.metadata(args.model)
    if metadata['name'] != 'Haas-ja-v1.0':
        raise ValueError('Give the Haas-ja-v1.0 model for the positive cases.')
    params = c.c_void_p()
    api.ok(api.lib.speech_load_params_new(c.byref(params)))
    try:
        if api.lib.speech_load_params_set_threads(params, 0) != -1:
            raise AssertionError('Public threads=0 must remain invalid; auto means omitting the setter.')
    finally:
        api.lib.speech_load_params_free(params)

    def case(label, setting, expected=None, **options):
        if setting is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = setting
        result = api.load(args.model, **options)
        if expected is not None and result['threads'] != expected:
            raise AssertionError(f'{label}: expected {expected} threads, got {result}')
        results.append({'case': label, 'selector': setting, 'requested_threads': options.get('threads', 0), **result})
        return result

    ordinary = case('unset', None)
    if not ordinary['device'].startswith('MTL'):
        raise ValueError('The positive check requires an available Metal device; other platforms are not a pass.')
    machine_default = ordinary['threads']
    if machine_default < 1:
        raise AssertionError('Haas must report its positive CPU decoding thread preference.')
    case('off', '0', machine_default)
    case('on', '1', min(4, machine_default))
    case('null parameters', '1', min(4, machine_default), null_params=True)
    case('invalid auto', 'invalid', refused=True)
    for count in [1, 2, 4, 8, 16]:
        case(f'explicit {count}', '1', count, threads=count)
    case('explicit ignores selector', 'invalid', 16, threads=16)
    case('CPU ignores selector', 'invalid', machine_default, device='cpu')
    if args.control_model:
        if api.metadata(args.control_model)['name'] == 'Haas-ja-v1.0':
            raise ValueError('The control model must be a different model.')
        control = []
        for setting in ['0', '1', 'invalid']:
            os.environ[FLAG] = setting
            control.append(api.load(args.control_model))
        if any(item != control[0] for item in control[1:]):
            raise AssertionError('The selector changed an unrelated model.')
        results.append({'case': 'unrelated model', 'results': control})
    print(json.dumps({'cases': results, 'control_model_checked': bool(args.control_model),
                      'scope': 'C API load selection and information only; inference, lazy CPU pool execution, '
                               'official dumps and non-Apple behavior are not checked'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library')
    parser.add_argument('model')
    parser.add_argument('--control-model')
    args = parser.parse_args()
    saved = os.environ.get(FLAG)
    try:
        check(args)
    finally:
        if saved is None:
            os.environ.pop(FLAG, None)
        else:
            os.environ[FLAG] = saved
