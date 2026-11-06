"""Prepare, run and audit the bounded 002b worker timing diagnostic.

Usage: 002b-worker-probe.py prepare --runner PATH --assets JSON --plain-build JSON
                                  --trace-build JSON --intervals JSON --out PATH
       runner-snapshot/002b-worker-probe.py run --manifest ../run.json
       runner-snapshot/002b-worker-probe.py analyze --manifest ../run.json
Only run starts model processes. Each worker holds the shared campaign lock.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
LOCK = '/private/tmp/speech-haas-mac-perf.lock'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def prepare(args):
    out = Path(args.out).resolve()
    data = read(args.assets)
    selected = {row['file']: row for row in data['subset']}
    records = [selected[row['file']] for row in data['stages']]
    if len(records) != 3 or len({r['stratum'] for r in records}) != 3:
        raise ValueError('The diagnostic requires the three original stage inputs, one per stratum.')
    builds = {'plain': read(args.plain_build), 'trace': read(args.trace_build)}
    for build in builds.values():
        if not build['source_verified'] or sha(build['speech']['path']) != build['speech']['sha256']:
            raise ValueError('Every diagnostic binary must match a clean verified build manifest.')
    inputs = []
    for record in records:
        pcm = Path(record['pcm_path']).read_bytes()
        if hashlib.sha256(pcm).hexdigest() != record['pcm_sha256'] or len(pcm) != round(record['seconds'] * 32000):
            raise ValueError('The diagnostic PCM differs from its recorded identity or 16 kHz PCM16 length.')
        inputs.append({key: record[key] for key in ['file', 'seconds', 'stratum', 'pcm_sha256']})
    intervals = read(args.intervals)
    if set(intervals) != {'intervals', 'python_contained_end', 'notes'}:
        raise ValueError('Name the trace intervals, last Python-contained point and timing notes explicitly.')
    out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'runner-snapshot'; snapshot.mkdir()
    for name in ['worker.py']:
        shutil.copyfile(Path(args.runner) / name, snapshot / name)
    shutil.copyfile(Path(__file__), snapshot / Path(__file__).name)
    models = {m['type']: m for m in data['models']}
    variants = {
        'effect': {
            'baseline': {'build': 'trace', 'environment': {'SPEECH_HAAS_FFN_F16': '0', 'SPEECH_HAAS_WORKER_TRACE': '1'}},
            'candidate': {'build': 'trace', 'environment': {'SPEECH_HAAS_FFN_F16': '1', 'SPEECH_HAAS_WORKER_TRACE': '1'}},
        },
        'overhead': {
            'plain': {'build': 'plain', 'environment': {'SPEECH_HAAS_FFN_F16': '0', 'SPEECH_HAAS_WORKER_TRACE': '0'}},
            'trace_disabled': {'build': 'trace', 'environment': {'SPEECH_HAAS_FFN_F16': '0', 'SPEECH_HAAS_WORKER_TRACE': '0'}},
            'trace_enabled': {'build': 'trace', 'environment': {'SPEECH_HAAS_FFN_F16': '0', 'SPEECH_HAAS_WORKER_TRACE': '1'}},
        },
    }
    jobs = []
    for phase, kinds in [('effect', ['q4_k', 'f16']), ('overhead', ['q4_k'])]:
        for kind in kinds:
            for rnd in range(3):
                order = list(variants[phase])
                if phase == 'effect' and rnd % 2:
                    order.reverse()
                elif phase == 'overhead':
                    order = order[rnd:] + order[:rnd]
                for side in order:
                    variant = variants[phase][side]
                    spec = {'directory': str(out / phase / kind / side / f'round-{rnd}'), 'phase': 'latency',
                            'round': rnd, 'side': side, 'build': builds[variant['build']], 'model': models[kind],
                            'environment': variant['environment'], 'threads': 4, 'warmup': records, 'records': records}
                    name = f'{phase}-{kind}-{side}-{rnd}.json'
                    write(out / name, spec)
                    jobs.append({'phase': phase, 'type': kind, 'side': side, 'round': rnd, 'spec': name,
                                 'spec_sha256': sha(out / name)})
    manifest = {'scope': 'worker_diagnostic', 'measure_id': '002b', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'inputs': inputs, 'assets_sha256': sha(args.assets), 'builds': builds, 'variants': variants,
                'models': {k: models[k] for k in ['q4_k', 'f16']}, 'jobs': jobs, **intervals,
                'runner_sha256': {p.name: sha(p) for p in snapshot.iterdir()},
                'conditions': {'threads': 4, 'device': 'gpu', 'decode_on_cpu': True, 'timestamps': False, 'python': sys.version,
                               'rounds': 3, 'warmup': 'All three inputs once per fresh process, longest first.',
                               'order': 'Effect AB/BA/AB. Overhead rotates all three variants. Input order shifts by round * 7.',
                               'latency_definition': 'Unmodified campaign worker.py timer: after PCM read/hash, before first base64 encoding, through parsed and validated terminal result.',
                               'trace_definition': 'C++ monotonic elapsed milliseconds relative to the entry of the same worker request; no cross-process clock epochs are subtracted.',
                               'limitations': ['Three inputs are a diagnostic sample, not the primary 30-input latency evaluation or 650-input quality evaluation.',
                                               'Nested intervals overlap and must not be summed. Median subintervals need not sum to a median parent.',
                                               'The outside-worker residual includes upload, parsing, queuing and terminal delivery; it does not isolate any one transport operation.',
                                               'Points after terminal output may lie outside the Python timer and are not subtracted from it.',
                                               'The trace emits one stderr record after cleanup; instrumentation and logging can affect following requests. The overhead phase bounds observed impact.']}}
    write(out / 'run.json', manifest)
    print(out / 'run.json')


def verify(root, manifest):
    for name, digest in manifest['runner_sha256'].items():
        if sha(root / 'runner-snapshot' / name) != digest:
            raise ValueError(f'The frozen diagnostic runner changed: {name}')
    for job in manifest['jobs']:
        if sha(root / job['spec']) != job['spec_sha256']:
            raise ValueError(f'The diagnostic job changed: {job["spec"]}')


def run(args):
    path = Path(args.manifest).resolve(); root = path.parent; manifest = read(path)
    verify(root, manifest)
    if any(key.startswith(('SPEECH_HAAS_', 'GGML_METAL_MM_TILE_N')) for key in os.environ):
        raise ValueError('Unset inherited campaign experiment variables; every diagnostic variant records its values.')
    for job in manifest['jobs']:
        subprocess.run(['/usr/bin/lockf', '-k', LOCK, sys.executable, '-B',
                        str(root / 'runner-snapshot' / Path(__file__).name), '_worker', '--spec', str(root / job['spec'])], check=True)


def trace_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        if '"haas_worker_trace"' not in line:
            continue
        row = json.loads(line)
        if row['haas_worker_trace'] != 1 or row['id'] in rows:
            raise ValueError('Trace version or request identity is invalid.')
        points = row['points_ms']
        if not points or any(not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in points.values()):
            raise ValueError('Trace points must be finite nonnegative milliseconds.')
        rows[row['id']] = points
    return rows


def analyze(args):
    path = Path(args.manifest).resolve(); root = path.parent; manifest = read(path)
    verify(root, manifest)
    rows = []
    expected_files = {r['file'] for r in manifest['inputs']}
    for job in manifest['jobs']:
        spec = read(root / job['spec']); directory = Path(spec['directory']); result = read(directory / 'result.json')
        if result['environment'] != spec['environment'] or result['build'] != spec['build'] or result['worker_source_sha256'] != manifest['runner_sha256']['worker.py']:
            raise ValueError('Worker identity differs from its diagnostic job.')
        if any(result[key] != spec[key] for key in ['phase', 'round', 'side']) or result['type'] != spec['model']['type']:
            raise ValueError('Worker phase, round, side or model type differs from its diagnostic job.')
        if result['warmup_count'] != 3 or len(result['records']) != 3 or {r['file'] for r in result['records']} != expected_files:
            raise ValueError('The diagnostic must warm and measure every fixed input once.')
        shift = spec['round'] * 7 % len(spec['records'])
        order = spec['records'][shift:] + spec['records'][:shift]
        if [r['file'] for r in result['records']] != [r['file'] for r in order]:
            raise ValueError('Measured input order differs from the frozen job; trace IDs cannot be paired.')
        traces = trace_rows(directory / 'stderr.log')
        enabled = spec['environment']['SPEECH_HAAS_WORKER_TRACE'] == '1'
        if enabled and set(traces) != {'warm-0', 'warm-1', 'warm-2', '0', '1', '2'}:
            raise ValueError('Each warmed and measured request must have exactly one trace.')
        if not enabled and traces:
            raise ValueError('Disabled tracing unexpectedly emitted trace data.')
        for index, record in enumerate(result['records']):
            row = {key: job[key] for key in ['phase', 'type', 'side', 'round']}
            row.update({'file': record['file'], 'stratum': record['stratum'], 'request_ms': record['seconds'] * 1000,
                        'text': record['text'], 'intervals_ms': {}})
            if enabled:
                points = traces[str(index)]
                row['points_ms'] = points
                for name, (begin, end) in manifest['intervals'].items():
                    duration = points[end] - points[begin]
                    if duration < -1e-6:
                        raise ValueError(f'Trace interval runs backwards: {name}')
                    row['intervals_ms'][name] = duration
                contained = points[manifest['python_contained_end']]
                residual = row['request_ms'] - contained
                if residual < -0.001:
                    raise ValueError('The contained worker interval exceeds its Python request time.')
                row['intervals_ms']['worker_entry_to_contained_end_ms'] = contained
                row['intervals_ms']['outside_to_contained_end_ms'] = residual
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        groups[(row['phase'], row['type'], row['side'])].append(row)
    profiles = {}
    for (phase, kind, side), group in groups.items():
        by_file = defaultdict(list)
        for row in group:
            by_file[row['file']].append(row)
        if any({r['round'] for r in items} != {0, 1, 2} for items in by_file.values()):
            raise ValueError('Every diagnostic input requires the same three distinct rounds.')
        medians = {file: {'request_ms': statistics.median(r['request_ms'] for r in items),
                          'intervals_ms': {key: statistics.median(r['intervals_ms'][key] for r in items) for key in items[0]['intervals_ms']},
                          'request_rounds_ms': [r['request_ms'] for r in sorted(items, key=lambda r: r['round'])],
                          'within_variant_text_changes': len({r['text'] for r in items}) != 1}
                   for file, items in by_file.items()}
        profiles.setdefault(phase, {}).setdefault(kind, {})[side] = medians
    comparisons = {}
    for phase, kinds in profiles.items():
        comparisons[phase] = {}
        base = 'baseline' if phase == 'effect' else 'plain'
        for kind, sides in kinds.items():
            comparisons[phase][kind] = {}
            for side in sides:
                if side == base:
                    continue
                a = {(r['round'], r['file']): r for r in groups[(phase, kind, base)]}
                b = {(r['round'], r['file']): r for r in groups[(phase, kind, side)]}
                if a.keys() != b.keys():
                    raise ValueError('Diagnostic sides have different paired requests.')
                metrics = {'request_ms': statistics.median(sides[side][f]['request_ms'] / sides[base][f]['request_ms'] for f in expected_files)}
                for key in next(iter(sides[base].values()))['intervals_ms']:
                    if all(sides[base][f]['intervals_ms'][key] > 0 for f in expected_files):
                        metrics[key] = statistics.median(sides[side][f]['intervals_ms'][key] / sides[base][f]['intervals_ms'][key] for f in expected_files)
                comparisons[phase][kind][side] = {'reference': base, 'paired_count': len(a),
                    'text_changed': sum(a[k]['text'] != b[k]['text'] for k in a), 'median_ratios': metrics,
                    'round_request_ratios': [statistics.median(b[k]['request_ms'] / a[k]['request_ms'] for k in a if k[0] == rnd) for rnd in range(3)]}
    summary = {'scope': manifest['scope'], 'measure_id': manifest['measure_id'], 'run_manifest_sha256': sha(path),
               'conditions': manifest['conditions'], 'interval_definitions': manifest['intervals'], 'notes': manifest['notes'],
               'inputs': manifest['inputs'], 'comparisons': comparisons, 'per_input_medians': profiles,
               'checks': {'same_three_inputs_every_round': True, 'all_trace_records_complete': True,
                          'nested_interval_directions': True, 'worker_binary_and_environment_identity': True,
                          'all_paired_texts_exact': all(c['text_changed'] == 0 for kinds in comparisons.values() for sides in kinds.values() for c in sides.values()),
                          'all_within_variant_texts_stable': all(not values['within_variant_text_changes'] for kinds in profiles.values() for sides in kinds.values() for files in sides.values() for values in files.values())},
               'rows': rows}
    destination = root / 'summary.json'
    if destination.exists():
        raise ValueError('Do not overwrite a diagnostic summary; retain the original run.')
    write(destination, summary)
    print(json.dumps(comparisons, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for name in ['runner', 'assets', 'plain-build', 'trace-build', 'intervals', 'out']:
        p.add_argument('--' + name, required=True)
    for name in ['run', 'analyze']:
        p = commands.add_parser(name); p.add_argument('--manifest', required=True)
    p = commands.add_parser('_worker'); p.add_argument('--spec', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    elif args.command == 'run': run(args)
    elif args.command == 'analyze': analyze(args)
    else:
        from worker import run as worker_run
        worker_run(read(args.spec))


if __name__ == '__main__':
    main()
