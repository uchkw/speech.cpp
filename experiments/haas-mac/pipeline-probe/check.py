#!/usr/bin/env python3
"""Checks diagnostic log integrity with synthetic records, without a model or Metal.

Usage: /usr/bin/python3 -B check.py [--out NEW_DIRECTORY]
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
from pipeline_trace import COUNTERS, PHASE_PREFIX, TRACE_PREFIX, analyze, compare, parse


def fixture(trace, warmup):
    inputs = [{'file': str(i), 'seconds': i + 1, 'pcm_sha256': str(i) * 64} for i in range(3)]
    manifest = {'trace': str(int(trace)), 'warmup': warmup, 'round': 1, 'threads': 4,
                'model': {'sha256': 'model'}, 'build': {'speech': {'sha256': 'binary'}},
                'records': inputs, 'source_sha256': {'run.py': 'source'}}
    entries, rows = [], []
    values = dict.fromkeys(COUNTERS, 0)
    graph_number, names_emitted = 0, 0

    def marker(event, **fields):
        entries.append((PHASE_PREFIX, {'event': event, **fields}))

    def snapshot(phase):
        nonlocal names_emitted
        count = values['compile_misses']
        entries.append((TRACE_PREFIX, {'event': 'pipeline_snapshot', 'phase': phase, 'library': 1,
                                       'context': 'ctx' if phase != 'library_free' else '0x0',
                                       'graph': graph_number if phase != 'library_free' else 0,
                                       'nodes': 12 if phase != 'library_free' else 0,
                                       'n_cb': 1 if phase != 'library_free' else 0, **values,
                                       'miss_names': ['kernel_example'] if count > names_emitted else [],
                                       'miss_names_recorded': count, 'miss_names_omitted': 0, 'miss_names_limit': 128}))
        names_emitted = count

    def graph():
        nonlocal graph_number
        graph_number += 1
        snapshot('encode_begin')
        values['lookups'] += 2
        if values['cache_inserts'] == 0:
            values['hits'] += 1
            for key in ('misses', 'compile_entries', 'compile_misses', 'new_function_calls',
                        'new_compute_pipeline_state_calls', 'new_compute_pipeline_state_successes', 'cache_inserts'):
                values[key] += 1
        else:
            values['hits'] += 2
        snapshot('encode_end')

    marker('launch_begin')
    if trace:
        entries.append((TRACE_PREFIX, {'event': 'source_libraries', 'library': 1,
                                       'new_library_with_source_calls': 2, 'successes': 2,
                                       'parallel_wall_us': 10, 'sum_library_api_us': 16,
                                       'kinds': [{'name': name, 'called': True, 'success': True, 'api_us': 8}
                                                 for name in ('a', 'b')]}))
        if warmup == 'standard':
            graph()
    marker('ready', warmup=warmup)
    for phase, order in (('first', inputs[::-1]), ('repeat', inputs[1:] + inputs[:1]), ('quality', inputs)):
        for i, record in enumerate(order):
            request_id = f'{phase}-{i}'
            marker('request_begin', phase=phase, id=request_id, file=record['file'], timestamps=phase == 'quality')
            if trace:
                graph()
            marker('request_end', phase=phase, id=request_id, file=record['file'])
            terminal = {'text': record['file'], 'stop': 'complete'}
            if phase == 'quality':
                terminal.update(tokens=[{'text': record['file'], 'start': 0}], segments=[])
            rows.append({'phase': phase, 'id': request_id, 'file': record['file'], 'seconds': .01,
                         'pcm_sha256': record['pcm_sha256'], 'terminal': terminal})
    if trace:
        snapshot('library_free')
    marker('process_end')
    return entries, manifest, {'ready_seconds': .1, 'records': rows}


def lines(entries):
    return [prefix + json.dumps(row) for prefix, row in entries]


def write_fixture(path, data):
    entries, manifest, result = copy.deepcopy(data)
    path.mkdir()
    encoded = json.dumps(manifest).encode()
    (path / 'run.json').write_bytes(encoded)
    result['manifest_sha256'] = hashlib.sha256(encoded).hexdigest()
    (path / 'result.json').write_text(json.dumps(result), encoding='utf-8')
    (path / 'stderr.log').write_text('\n'.join(lines(entries)) + '\n', encoding='utf-8')


def rejected(label, operation):
    try:
        operation()
    except ValueError:
        return label
    raise AssertionError('Accepted invalid evidence: ' + label)


def check(destination):
    checks = []
    for warmup in ('none', 'standard'):
        for trace in (False, True):
            path = destination / f'{warmup}-{int(trace)}'
            write_fixture(path, fixture(trace, warmup))
            summary = analyze(path)
            assert summary['integrity'] == 'passed'
            if trace:
                assert summary['phases']['repeat']['new_compute_pipeline_state_calls'] == 0
                assert summary['phases']['repeat']['lookups'] == 6
                assert len(summary['miss_names']) == 1
            checks.append(f'valid {warmup} trace={int(trace)}')
        paired = compare(destination / f'{warmup}-0', destination / f'{warmup}-1')
        assert paired['comparisons'] == 9 and paired['median_repeat_on_over_off'] == 1
        checks.append(f'exact output pair {warmup}')

    base = fixture(True, 'standard')[0]
    first_end = next(i for i, (_, row) in enumerate(base) if row.get('phase') == 'encode_end')
    saturated = copy.deepcopy(base[first_end][1])
    saturated.update(dict.fromkeys(COUNTERS, 0))
    for key in ('lookups', 'misses', 'compile_entries', 'compile_misses', 'new_function_calls',
                'new_compute_pipeline_state_calls', 'new_compute_pipeline_state_successes', 'cache_inserts'):
        saturated[key] = 130
    saturated.update(miss_names=[f'kernel_{i}' for i in range(128)], miss_names_recorded=128, miss_names_omitted=2)
    final = {**saturated, 'phase': 'library_free', 'graph': 0, 'nodes': 0, 'n_cb': 0, 'miss_names': []}
    bounded = base[:first_end] + [(TRACE_PREFIX, saturated), base[first_end + 1], (TRACE_PREFIX, final), base[-1]]
    parsed = parse(lines(bounded))
    assert len(parsed['miss_names']) == 128 and parsed['library_final'][0]['miss_names_omitted'] == 2
    checks.append('bounded miss names with complete aggregate count')
    second_begin = next(i for i, (_, row) in enumerate(base) if i > first_end and row.get('phase') == 'encode_begin')
    mutations = [
        ('missing graph end', lambda rows: rows.pop(first_end)),
        ('changed command buffer policy', lambda rows: rows[first_end][1].update(n_cb=0)),
        ('counter accounting', lambda rows: rows[first_end][1].update(lookups=99)),
        ('missing miss name', lambda rows: rows[first_end][1].update(miss_names=[])),
        ('wrong omitted count', lambda rows: rows[first_end][1].update(miss_names_omitted=1)),
        ('truncated source API sum', lambda rows: rows[1][1].update(sum_library_api_us=8)),
        ('invalid parallel wall', lambda rows: rows[1][1].update(parallel_wall_us=4)),
        ('missing final snapshot', lambda rows: rows.pop(-2)),
        ('duplicate marker', lambda rows: rows.insert(1, copy.deepcopy(rows[0]))),
        ('decreasing counters', lambda rows: rows[second_begin][1].update(dict.fromkeys(COUNTERS, 0))),
    ]
    for label, mutate in mutations:
        changed = copy.deepcopy(base)
        mutate(changed)
        checks.append(rejected(label, lambda: parse(lines(changed))))
    checks.append(rejected('interleaved line', lambda: parse(['other log ' + lines(base)[0]])))
    wrong_order = fixture(True, 'none')
    wrong_order[2]['records'][0], wrong_order[2]['records'][1] = wrong_order[2]['records'][1], wrong_order[2]['records'][0]
    write_fixture(destination / 'wrong-order', wrong_order)
    checks.append(rejected('wrong request order', lambda: analyze(destination / 'wrong-order')))
    changed_output = fixture(True, 'none')
    changed_output[2]['records'][-1]['terminal']['tokens'][0]['start'] = .1
    write_fixture(destination / 'changed-time', changed_output)
    checks.append(rejected('changed public timestamp', lambda: compare(destination / 'none-0', destination / 'changed-time')))
    trace_off = fixture(True, 'none')
    trace_off[1]['trace'] = '0'
    write_fixture(destination / 'false-off', trace_off)
    checks.append(rejected('trace output while OFF', lambda: analyze(destination / 'false-off')))
    return {'passed': len(checks), 'checks': checks, 'scope': 'Synthetic parser/output fixtures only; no worker, build, model or GPU.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out')
    args = parser.parse_args()
    root = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix='speech-haas-pipeline-check-'))
    if args.out:
        root.mkdir(parents=True, exist_ok=False)
    result = check(root)
    (root / 'check.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'out': str(root), **result}, indent=2))
