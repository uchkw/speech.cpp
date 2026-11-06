"""Repeat a completed primary run's 30-input latency block without pooling results or rescoring CER.

Usage: latency-repeat.py run --primary-run DIRECTORY --runner FROZEN_RUNNER --assets ASSETS_JSON --out NEW_DIRECTORY
The recorded primary worker, Python, binaries, models, PCM, options and timer remain unchanged.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import compare

HERE = Path(__file__).resolve().parent
SCOPE = 'independent_30_input_latency_repeat'
SIDES = ['baseline', 'candidate']


def primary(args):
    directory, runner = Path(args.primary_run).resolve(), Path(args.runner).resolve()
    manifest, summary, data = [compare.read(p) for p in [directory / 'run.json', directory / 'summary.json', args.assets]]
    if manifest['scope'] != 'full' or summary['scope'] != 'full' or summary['measure_id'] != manifest['measure_id']:
        raise ValueError('Repeat a completed primary full comparison.')
    if summary['evidence']['run_manifest_sha256'] != compare.sha(directory / 'run.json'):
        raise ValueError('The primary summary does not match its run manifest.')
    policy = manifest['conditions']
    if policy['rounds'] != 3 or policy['latency_count'] != 30 or policy['quality_count'] != 650 or not policy['decode_on_cpu']:
        raise ValueError('Retain the primary three-round 30/650 CPU-decoder conditions.')
    if policy['python'] != platform.python_version() or not policy.get('python_executable') or Path(policy['python_executable']).resolve() != Path(sys.executable).resolve():
        raise ValueError('Use the primary recorded Python version and executable.')
    for name in ['compare.py', 'worker.py']:
        if compare.sha(runner / name) != policy['runner_sha256'][name]:
            raise ValueError(f'Use the identical primary frozen runner source: {name}')
    if compare.sha(HERE / 'compare.py') != policy['runner_sha256']['compare.py']:
        raise ValueError('Use the identical primary comparison arithmetic source.')
    if compare.sha(args.assets) != manifest['assets_sha256'] or data['input_hash'] != manifest['input_hash']:
        raise ValueError('Use the primary source assets without changing their manifest.')
    rows = data['subset']
    if len(rows) != 30 or len({r['file'] for r in rows}) != 30 or {s: sum(r['stratum'] == s for r in rows) for s in ['6-10s', '10-20s', '20-30s']} != {'6-10s': 10, '10-20s': 10, '20-30s': 10}:
        raise ValueError('Repeat the original 30 unique recordings, ten per duration stratum, not the six prefixes.')
    for side in SIDES:
        build = manifest['builds'][side]
        if not build['source_verified'] or compare.sha(build['speech']['path']) != build['speech']['sha256']:
            raise ValueError('Every repeat binary must match its verified primary build.')
        compare.execution_environment(manifest['execution_environments'][side])
    models = {m['type']: m for m in data['models']}
    for kind in manifest['types']:
        if models[kind]['sha256'] != manifest['models'][kind]['sha256'] or compare.sha(models[kind]['path']) != models[kind]['sha256']:
            raise ValueError('Every repeat model must match the primary SHA-256.')
    for row in rows:
        if compare.sha(row['pcm_path']) != row['pcm_sha256']:
            raise ValueError('A primary PCM input changed.')
    order = []
    for kind in manifest['types']:
        for side in manifest['builds']:
            for rnd in range(3):
                result = compare.read(directory / kind / side / f'latency-{rnd}/result.json')
                order.append({k: result[k] for k in ['started_utc', 'type', 'round', 'side']})
    return manifest, data, sorted(order, key=lambda row: row['started_utc'])


def source_identity():
    repository = HERE.parents[1]
    commit = compare.git(repository, 'rev-parse', 'HEAD')
    path = str(Path(__file__).resolve().relative_to(repository))
    blob = subprocess.check_output(['git', '-C', str(repository), 'show', f'{commit}:{path}'])
    digest = compare.sha(__file__)
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError('Commit this driver before starting an evaluated repeat.')
    return {'commit': commit, 'path': path, 'sha256': digest}


def run_worker(directory, spec):
    subprocess.run([sys.executable, '-B', str(directory / 'runner-snapshot/latency-repeat.py'), '_worker', '--spec', str(spec)], check=True)


def run(args):
    identity = source_identity()
    source, data, original_order = primary(args)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'runner-snapshot'; snapshot.mkdir()
    shutil.copyfile(__file__, snapshot / 'latency-repeat.py')
    for name in ['compare.py', 'worker.py']:
        shutil.copyfile(Path(args.runner) / name, snapshot / name)
        if compare.sha(snapshot / name) != source['conditions']['runner_sha256'][name]:
            raise ValueError('A runner source changed while the repeat snapshot was copied.')
    policy = source['conditions']
    manifest = {'scope': SCOPE, 'measure_id': source['measure_id'], 'types': source['types'],
                'primary_run_sha256': compare.sha(Path(args.primary_run) / 'run.json'),
                'primary_summary_sha256': compare.sha(Path(args.primary_run) / 'summary.json'),
                'assets_sha256': source['assets_sha256'], 'input_hash': source['input_hash'],
                'inputs': [{k: v for k, v in r.items() if k != 'pcm_path'} for r in data['subset']],
                'builds': {k: source['builds'][k] for k in SIDES}, 'models': source['models'],
                'execution_environments': {k: source['execution_environments'][k] for k in SIDES},
                'driver_source': identity, 'runner_sha256': {p.name: compare.sha(p) for p in snapshot.iterdir()},
                'primary_latency_block_order': original_order,
                'conditions': {**policy, 'quality_count': 0, 'latency_timestamps': False, 'quality_timestamps': None,
                    'created_utc': datetime.now(timezone.utc).isoformat(),
                    'order': 'Per type, AB/BA/AB with no anchor, quality or stage workers between blocks; input order shifts by round * 7.',
                    'warmup': 'All original 30 inputs once per fresh worker, longest first.',
                    'input_scope': 'The original thirty 6-30-second recordings, not the separate six 2/4-second artificial prefixes.',
                    'accuracy_scope': 'Compare every timestamp-disabled latency text; no new CER, public timestamp arrays or 650-input quality evaluation.',
                    'pooling': 'This independent block is never pooled with the primary or prefix supplement.',
                    'fixed_anchor_measured': False},
                'created_utc': datetime.now(timezone.utc).isoformat()}
    compare.write(out / 'run.json', manifest)
    models = {m['type']: m for m in data['models']}
    for kind in manifest['types']:
        for rnd in range(3):
            for side in SIDES if rnd % 2 == 0 else SIDES[::-1]:
                spec = {'directory': str(out / kind / side / f'latency-{rnd}'), 'build': manifest['builds'][side],
                        'model': models[kind], 'threads': policy['threads'], 'phase': 'latency', 'round': rnd, 'side': side,
                        'environment': manifest['execution_environments'][side], 'warmup': data['subset'], 'records': data['subset']}
                path = out / f'{kind}-{side}-{rnd}.json'; compare.write(path, spec); run_worker(out, path)
    summarize(out)


def comparison(a, b):
    ratios = [b[f]['seconds'] / a[f]['seconds'] for f in a]
    ratio = statistics.median(ratios)
    return {'count': len(ratios), 'median_latency_ratio': ratio, 'latency_reduction_percent': (1 - ratio) * 100,
            'improved': sum(r < 1 for r in ratios), 'equal': sum(r == 1 for r in ratios), 'slower': sum(r > 1 for r in ratios)}


def summarize(directory):
    if (directory / 'summary.json').exists():
        raise ValueError('Do not overwrite an evaluated latency repeat.')
    manifest = compare.read(directory / 'run.json'); profiles = {}; exact = True
    expected = {r['file']: r for r in manifest['inputs']}
    for kind in manifest['types']:
        sides, by_side = {}, {}
        for side in SIDES:
            runs = [compare.read(directory / kind / side / f'latency-{rnd}/result.json') for rnd in range(3)]
            groups = defaultdict(list)
            for rnd, run in enumerate(runs):
                if (run['round'], run['side'], run['type'], run['phase'], run['warmup_count']) != (rnd, side, kind, 'latency', 30):
                    raise ValueError('Each repeat worker must warm all thirty inputs and record its phase, side and round.')
                if run['build'] != manifest['builds'][side] or run['environment'] != manifest['execution_environments'][side] or run['worker_source_sha256'] != manifest['runner_sha256']['worker.py']:
                    raise ValueError('Repeat worker identity differs from its frozen manifest.')
                command = run['command']
                if command[command.index('--threads') + 1] != str(manifest['conditions']['threads']) or '--decode-on-cpu' not in command:
                    raise ValueError('Repeat workers must use the recorded CPU decoder threads.')
                if len(run['records']) != 30 or {r['file'] for r in run['records']} != expected.keys():
                    raise ValueError('Every repeat round must contain the same thirty unique recordings.')
                for row in run['records']:
                    source = expected[row['file']]
                    if row['reference'] != source['reference'] or row['audio_seconds'] != source['seconds'] or row['stratum'] != source['stratum'] or row['tokens'] is not None or row['segments'] is not None:
                        raise ValueError('Repeat records must retain the original input metadata with timestamps disabled.')
                    groups[row['file']].append(row)
            medians = {f: {**rows[0], 'seconds': statistics.median(r['seconds'] for r in rows)} for f, rows in groups.items()}
            repeated = sum(len({r['text'] for r in rows}) > 1 for rows in groups.values()); exact = exact and repeated == 0
            values = list(medians.values())
            sides[side] = {'latency': compare.latency(values),
                          'strata': {s: compare.latency([r for r in values if r['stratum'] == s]) for s in sorted({r['stratum'] for r in values})},
                          'rounds': [compare.latency(r['records']) for r in runs], 'repeated_text_changes': repeated,
                          'literal_cer': None, 'repository_cer': None,
                          'peak_memory_bytes': max(r['memory']['peak_memory_footprint'] for r in runs),
                          'load_ms': statistics.median(r['load_seconds'] for r in runs) * 1000,
                          'first_request_ms': statistics.median(r['first_request']['seconds'] for r in runs) * 1000}
            by_side[side] = {'runs': runs, 'medians': medians}
        rounds = []
        for rnd in range(3):
            a, b = [{r['file']: r for r in by_side[side]['runs'][rnd]['records']} for side in SIDES]
            rounds.append({'round': rnd, **comparison(a, b), 'text_changed': sum(a[f]['text'] != b[f]['text'] for f in a)})
        a, b = [by_side[side]['medians'] for side in SIDES]
        changed = sum(r['text_changed'] for r in rounds); exact = exact and changed == 0
        paired = {**comparison(a, b), 'latency_text_comparisons': 90, 'text_changed': changed, 'rounds': rounds,
                  'strata': {s: comparison({f: r for f, r in a.items() if r['stratum'] == s}, b) for s in sorted({r['stratum'] for r in a.values()})},
                  'inputs': [{'file': f, 'stratum': a[f]['stratum'], 'baseline_ms': a[f]['seconds'] * 1000,
                              'candidate_ms': b[f]['seconds'] * 1000, 'ratio': b[f]['seconds'] / a[f]['seconds']} for f in sorted(a)]}
        profiles[kind] = {'sides': sides, 'comparison': paired}
    compare.write(directory / 'summary.json', {**manifest, 'run_manifest_sha256': compare.sha(directory / 'run.json'),
                  'types': profiles, 'status': 'exact_text_matched' if exact else 'text_changed', 'pooled_with_primary': False,
                  'adoption_evidence_alone': False, 'quality_evaluation': 'not repeated'})
    print(directory / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    for name in ['run', '_run']:
        p = commands.add_parser(name)
        for field in ['primary-run', 'runner', 'assets', 'out']: p.add_argument('--' + field, required=True)
    p = commands.add_parser('_worker'); p.add_argument('--spec', required=True)
    args = parser.parse_args()
    if args.command == 'run':
        subprocess.run(['/usr/bin/lockf', '-k', compare.LOCK, sys.executable, '-B', str(Path(__file__).resolve()), '_run', *sys.argv[2:]], check=True)
    elif args.command == '_run': run(args)
    else:
        from worker import run as worker_run
        worker_run(compare.read(args.spec))


if __name__ == '__main__':
    main()
