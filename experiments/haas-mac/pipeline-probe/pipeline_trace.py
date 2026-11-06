#!/usr/bin/env python3
"""Checks pipeline counter integrity and pairs diagnostic output without rerunning inference.

Usage: pipeline_trace.py DIRECTORY
       pipeline_trace.py --compare TRACE_OFF_DIRECTORY TRACE_ON_DIRECTORY
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

COUNTERS = ('lookups', 'hits', 'misses', 'compile_entries', 'recheck_hits', 'compile_misses',
            'missing_kernels', 'new_function_calls', 'new_function_failures',
            'new_compute_pipeline_state_calls', 'new_compute_pipeline_state_failures',
            'new_compute_pipeline_state_successes', 'incompatible_pipeline_states', 'cache_inserts')
TRACE_PREFIX = 'GGML_METAL_PIPELINE_TRACE '
PHASE_PREFIX = 'HAAS_PIPELINE_PHASE '


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def counters(record):
    result = {key: record[key] for key in COUNTERS}
    require(all(type(value) is int and value >= 0 for value in result.values()), 'Invalid counter value.')
    require(record['lookups'] == record['hits'] + record['misses'], 'Lookup accounting differs.')
    require(record['compile_entries'] == record['recheck_hits'] + record['compile_misses'], 'Compile entry accounting differs.')
    require(record['compile_misses'] == record['missing_kernels'] + record['new_function_calls'], 'Function-call accounting differs.')
    require(record['new_function_calls'] == record['new_function_failures'] + record['new_compute_pipeline_state_calls'],
            'Pipeline-state call accounting differs.')
    require(record['new_compute_pipeline_state_calls'] ==
            record['new_compute_pipeline_state_failures'] + record['new_compute_pipeline_state_successes'],
            'Pipeline-state result accounting differs.')
    require(record['new_compute_pipeline_state_successes'] == record['incompatible_pipeline_states'] + record['cache_inserts'],
            'Cache insertion accounting differs.')
    return result


def parse(lines):
    phase = None
    request = None
    launched = ready = ended = False
    completed = []
    active = {}
    latest = {}
    freed = set()
    graphs = []
    libraries = []
    misses = []
    events = 0
    for line in lines:
        if line.startswith(PHASE_PREFIX):
            row = json.loads(line[len(PHASE_PREFIX):])
            event = row['event']
            require(not active, 'A Python marker intersects a Metal graph encoding interval.')
            if event == 'launch_begin':
                require(not launched, 'Duplicate launch marker.')
                launched, phase = True, 'load'
            elif event == 'ready':
                require(launched and not ready and not ended, 'Invalid ready marker.')
                ready, phase = True, None
            elif event == 'request_begin':
                require(ready and not ended and request is None, 'Invalid request begin marker.')
                require(row['phase'] in ('first', 'repeat', 'quality'), 'Unknown request phase.')
                require(row['timestamps'] is (row['phase'] == 'quality'), 'Request timestamp option differs from its phase.')
                phase, request = row['phase'], row
            elif event == 'request_end':
                require(request is not None and all(row[key] == request[key] for key in ('id', 'file', 'phase')),
                        'Request marker identities differ.')
                completed.append(request)
                phase, request = None, None
            elif event == 'process_end':
                require(ready and not ended and request is None, 'Invalid process end marker.')
                ended = True
            else:
                raise ValueError('Unknown Python marker event: ' + event)
            continue
        if not line.startswith(TRACE_PREFIX):
            require(TRACE_PREFIX.strip() not in line and PHASE_PREFIX.strip() not in line, 'Interleaved diagnostic record.')
            continue
        require(launched and not ended, 'Metal record outside the worker process markers.')
        row = json.loads(line[len(TRACE_PREFIX):])
        events += 1
        if row['event'] in ('source_libraries', 'source_library_from_source'):
            require(phase == 'load', 'Unexpected source-library creation after ready.')
            require(row['new_library_with_source_calls'] == row['successes'] > 0, 'Source-library creation failed.')
            if row['event'] == 'source_libraries':
                kinds = row['kinds']
                require(row['new_library_with_source_calls'] == sum(item['called'] for item in kinds), 'Source calls differ.')
                require(row['successes'] == sum(item['success'] for item in kinds), 'Source successes differ.')
                require(row['sum_library_api_us'] == sum(item['api_us'] for item in kinds), 'Source API time sum differs.')
                require(row['parallel_wall_us'] >= max(item['api_us'] for item in kinds), 'Invalid parallel source wall time.')
            libraries.append(row)
            continue
        require(row['event'] == 'pipeline_snapshot', 'Unknown Metal diagnostic event.')
        lib = row['library']
        require(type(lib) is int and lib > 0 and lib not in freed, 'Invalid or freed library identity.')
        values = counters(row)
        previous = latest.get(lib)
        if previous:
            require(all(row[key] >= previous[key] for key in COUNTERS), 'Cumulative counters went backwards.')
        previous_names = previous['miss_names_recorded'] if previous else 0
        require(row['miss_names_limit'] == 128 and 0 <= row['miss_names_recorded'] <= 128, 'Invalid name bound.')
        require(row['miss_names_recorded'] - previous_names == len(row['miss_names']), 'Missing or repeated name records.')
        require(row['miss_names_omitted'] == row['compile_misses'] - row['miss_names_recorded'], 'Omitted-name accounting differs.')
        require(all(isinstance(name, str) and len(name.encode('utf-8')) < 256 for name in row['miss_names']), 'Invalid pipeline name.')
        misses += [{'library': lib, 'phase': phase, 'id': request['id'] if request else None, 'name': name}
                   for name in row['miss_names']]
        latest[lib] = row
        key = (lib, row['context'], row['graph'])
        if row['phase'] == 'encode_begin':
            require(not active and row['nodes'] > 0 and row['n_cb'] == 1, 'Overlapping graphs or changed command-buffer policy.')
            active[key] = (row, phase, request['id'] if request else None)
        elif row['phase'] == 'encode_end':
            require(key in active, 'Graph has no matching begin snapshot.')
            begin, begin_phase, request_id = active.pop(key)
            require((row['nodes'], row['n_cb']) == (begin['nodes'], begin['n_cb']), 'Graph metadata changed while encoding.')
            require((phase, request['id'] if request else None) == (begin_phase, request_id), 'Graph phase changed.')
            delta = {name: values[name] - begin[name] for name in COUNTERS}
            require(all(value >= 0 for value in delta.values()) and delta['lookups'] > 0, 'Graph counters are incomplete.')
            graphs.append({'library': lib, 'context': row['context'], 'graph': row['graph'], 'nodes': row['nodes'],
                           'n_cb': row['n_cb'], 'phase': phase, 'id': request_id, **delta})
        elif row['phase'] == 'library_free':
            require(not active and row['graph'] == 0, 'Library freed during graph encoding.')
            freed.add(lib)
        else:
            raise ValueError('Unknown snapshot phase: ' + row['phase'])
    require(ended and not active and request is None, 'Incomplete worker markers or graph snapshots.')
    require(freed == set(latest), 'A traced library lacks its final snapshot.')
    failures = ('missing_kernels', 'new_function_failures', 'new_compute_pipeline_state_failures', 'incompatible_pipeline_states')
    require(all(not row[key] for row in latest.values() for key in failures), 'Metal creation failure in the trace.')
    return {'events': events, 'graphs': graphs, 'source_libraries': libraries, 'miss_names': misses,
            'library_final': list(latest.values()), 'requests': completed}


def analyze(directory):
    directory = Path(directory)
    manifest, result = read(directory / 'run.json'), read(directory / 'result.json')
    require(hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest() == result['manifest_sha256'],
            'The manifest differs from the measured record.')
    parsed = parse((directory / 'stderr.log').read_text(encoding='utf-8').splitlines())
    expected = [(row['phase'], row['id'], row['file']) for row in result['records']]
    observed = [(row['phase'], row['id'], row['file']) for row in parsed['requests']]
    require(len(set(expected)) == len(expected) and expected == observed, 'Marker/record keys or order differ.')
    records = manifest['records']
    shift = (manifest['round'] * 7) % len(records)
    orders = [('first', sorted(records, key=lambda row: (-row['seconds'], row['file']))),
              ('repeat', records[shift:] + records[:shift]), ('quality', records)]
    require(expected == [(phase, f'{phase}-{i}', row['file']) for phase, rows in orders for i, row in enumerate(rows)],
            'The phase or input order differs from the plan.')
    inputs = {row['file']: row for row in records}
    for row in result['records']:
        require(row['pcm_sha256'] == inputs[row['file']]['pcm_sha256'] and row['seconds'] > 0, 'Invalid measured input or duration.')
        require(row['terminal']['stop'] == 'complete', 'A request did not complete.')
        if row['phase'] == 'quality':
            require(all(isinstance(row['terminal'].get(key), list) for key in ('tokens', 'segments')), 'Missing public timing output.')
    for file in {row['file'] for row in result['records']}:
        require(len({row['terminal']['text'] for row in result['records'] if row['file'] == file}) == 1,
                'A file changed text between first, repeat and quality requests.')
    traced = manifest['trace'] == '1'
    require(bool(parsed['events']) == traced and bool(parsed['graphs']) == traced, 'Trace switch and observed records differ.')
    if traced:
        require(parsed['source_libraries'], 'The source-embedded diagnostic build did not report source-library calls.')
        require(any(row['cache_inserts'] > 0 for row in parsed['library_final']), 'No pipeline was created in a fresh process.')
        require({row['id'] for row in parsed['graphs'] if row['phase'] in ('first', 'repeat', 'quality')} ==
                {row['id'] for row in result['records']}, 'A request has no traced graph.')
        require((manifest['warmup'] == 'standard') == any(row['phase'] == 'load' for row in parsed['graphs']),
                'Standard/no-warmup load graph evidence differs.')
    phases = {}
    for phase in ('load', 'first', 'repeat', 'quality'):
        graphs = [row for row in parsed['graphs'] if row['phase'] == phase]
        phases[phase] = {'graphs': len(graphs), **{name: sum(row[name] for row in graphs) for name in COUNTERS}}
    unscoped = {name: sum(row[name] for row in parsed['library_final']) -
                sum(row[name] for row in parsed['graphs']) for name in COUNTERS}
    require(all(value >= 0 for value in unscoped.values()), 'Graph counts exceed process totals.')
    return {'schema': 1, 'integrity': 'passed', 'trace': manifest['trace'], 'warmup': manifest['warmup'],
            'phases': phases, 'outside_graph_counters': unscoped, **parsed,
            'counter_scope': 'Host encoding only; API calls do not expose Apple internal compiler cache misses.',
            'warm_compile_avoidance': ('no remaining observed API calls' if traced and phases['repeat']['compile_misses'] == 0
                                      and phases['repeat']['new_compute_pipeline_state_calls'] == 0
                                      and unscoped['compile_misses'] == 0 else 'not established'),
            'timing_scope': 'Source parallel wall and sum of per-library API durations are separate; neither is subtracted from Python latency.'}


def compare(off_path, on_path):
    off_manifest, on_manifest = (read(Path(path) / 'run.json') for path in (off_path, on_path))
    require(off_manifest['trace'] == '0' and on_manifest['trace'] == '1', 'Expected trace OFF followed by trace ON.')
    for key in ('model', 'records', 'threads', 'round', 'warmup', 'source_sha256'):
        require(off_manifest[key] == on_manifest[key], 'The paired condition differs: ' + key)
    require(off_manifest['build']['speech']['sha256'] == on_manifest['build']['speech']['sha256'], 'The pair must use one binary.')
    analyze(off_path)
    analyze(on_path)
    off, on = (read(Path(path) / 'result.json') for path in (off_path, on_path))
    left = {(row['phase'], row['file']): row for row in off['records']}
    right = {(row['phase'], row['file']): row for row in on['records']}
    require(left.keys() == right.keys(), 'Paired file keys differ.')
    ratios = []
    for key, a in left.items():
        b = right[key]
        require(a['pcm_sha256'] == b['pcm_sha256'], 'Paired PCM identities differ.')
        require(all(a['terminal'].get(name) == b['terminal'].get(name) for name in ('text', 'tokens', 'segments', 'stop')),
                'Trace ON changes a public output: ' + str(key))
        ratios.append({'phase': key[0], 'file': key[1], 'on_over_off': b['seconds'] / a['seconds']})
    return {'exact_public_outputs': True, 'comparisons': len(left), 'warmup': off_manifest['warmup'],
            'ready_on_over_off': on['ready_seconds'] / off['ready_seconds'], 'request_ratios': ratios,
            'median_repeat_on_over_off': statistics.median(row['on_over_off'] for row in ratios if row['phase'] == 'repeat'),
            'interpretation': 'Diagnostic overhead only; this is not a speed-up candidate or an attribution of request time.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    require(len(args.directories) == (2 if args.compare else 1), 'Wrong number of directories.')
    output = compare(*args.directories) if args.compare else analyze(args.directories[0])
    print(json.dumps(output, ensure_ascii=False, indent=2))
