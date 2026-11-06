"""Measure six unlabeled 2/4-second prefixes as a supplement to a completed primary Haas run.

Usage: short.py prepare --assets ASSETS_JSON --out INPUT_DIRECTORY
       short.py run --primary-run RUN_DIRECTORY --inputs INPUTS_JSON --out RUN_DIRECTORY
The existing worker timer is unchanged. This supplement never computes CER or changes a primary result.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
import compare

HERE = Path(__file__).resolve().parent
SCOPE = 'short_prefix_latency_exact_supplement'
LIMITATIONS = [
    'These are truncated prefixes of three longer FLEURS recordings, not naturally completed short utterances.',
    'The full-recording reference text does not label a prefix. Literal CER and repository CER are null and are never scored.',
    'Exact text agreement compares two implementations; it does not establish recognition accuracy on short utterances.',
    'Six prefixes from three recordings are a small related sample. Their p99 is not a production tail estimate.',
    'This supplement is separate from the primary 30-input latency and 650-input quality evaluation and cannot support adoption by itself.',
]


def input_hash(records):
    identity = [{k: r[k] for k in ['file', 'pcm_sha256', 'source_file', 'source_pcm_sha256', 'start_sample', 'sample_count']} for r in records]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def prepare(args):
    data = compare.read(args.assets)
    subset = {r['file']: r for r in data['subset']}
    sources = [subset[r['file']] for r in data['stages']]
    if len(sources) != 3 or len({r['file'] for r in sources}) != 3 or len({r['stratum'] for r in sources}) != 3:
        raise ValueError('Choose the three existing stage inputs, with distinct files and duration strata.')
    pcm = {}
    for source in sources:
        value = Path(source['pcm_path']).read_bytes()
        if hashlib.sha256(value).hexdigest() != source['pcm_sha256'] or len(value) < 4 * 32000 or len(value) % 2:
            raise ValueError('A source PCM changed or cannot provide a four-second PCM16 prefix.')
        pcm[source['file']] = value
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    records = []
    for source in sources:
        for seconds in [2, 4]:
            name = f"{Path(source['file']).stem}.prefix-{seconds}s.pcm"
            path = out / name; path.write_bytes(pcm[source['file']][:seconds * 32000])
            records.append({'file': name, 'pcm_path': str(path), 'pcm_sha256': compare.sha(path),
                            'seconds': seconds, 'stratum': f'{seconds}s', 'reference': None,
                            'source_file': source['file'], 'source_pcm_sha256': source['pcm_sha256'],
                            'source_samples': len(pcm[source['file']]) // 2, 'start_sample': 0, 'sample_count': seconds * 16000})
    manifest = {'scope': 'short_prefix_inputs', 'sample_rate': 16000, 'format': 'mono PCM16 little endian',
                'source_assets_sha256': compare.sha(args.assets), 'source_input_hash': data['input_hash'],
                'input_hash': input_hash(records),
                'records': records, 'models': data['models'], 'literal_cer': None, 'repository_cer': None,
                'accuracy_reference': 'unavailable', 'limitations': LIMITATIONS}
    compare.write(out / 'inputs.json', manifest)
    print(out / 'inputs.json')


def inputs(path):
    data = compare.read(path)
    if data.get('scope') != 'short_prefix_inputs' or data.get('sample_rate') != 16000:
        raise ValueError('Use the independently prepared 16 kHz short-prefix manifest.')
    rows = data['records']
    if len(rows) != 6 or len({r['file'] for r in rows}) != 6 or len({r['source_file'] for r in rows}) != 3:
        raise ValueError('The supplement requires six unique prefixes from three source files.')
    for source in {r['source_file'] for r in rows}:
        if sorted(r['seconds'] for r in rows if r['source_file'] == source) != [2, 4]:
            raise ValueError('Each source must provide exactly one two-second and one four-second prefix.')
    for row in rows:
        if row['reference'] is not None or row['start_sample'] != 0 or row['sample_count'] != row['seconds'] * 16000 or row['stratum'] != f"{row['seconds']}s":
            raise ValueError('A prefix must remain unlabeled and start at sample zero with the recorded duration.')
        if Path(row['pcm_path']).stat().st_size != row['sample_count'] * 2 or compare.sha(row['pcm_path']) != row['pcm_sha256']:
            raise ValueError('A short-prefix PCM differs from its recorded hash or sample count.')
    if input_hash(rows) != data['input_hash']:
        raise ValueError('The short-prefix input identity changed after preparation.')
    return data


def primary(path, data):
    directory = Path(path).resolve()
    manifest = compare.read(directory / 'run.json')
    summary = compare.read(directory / 'summary.json')
    if manifest['scope'] != 'full' or summary['scope'] != 'full' or summary['measure_id'] != manifest['measure_id']:
        raise ValueError('The supplement requires a completed primary full comparison.')
    if summary['evidence']['run_manifest_sha256'] != compare.sha(directory / 'run.json'):
        raise ValueError('The primary summary does not match its run manifest.')
    if manifest['assets_sha256'] != data['source_assets_sha256'] or manifest['input_hash'] != data['source_input_hash']:
        raise ValueError('The short inputs and primary run must derive from the same recorded source assets.')
    policy = manifest['conditions']
    if policy['python'] != platform.python_version():
        raise ValueError(f"Use the primary run's Python {policy['python']}; the current interpreter is {platform.python_version()}.")
    executable = policy.get('python_executable')
    if not executable:
        raise ValueError('Use a primary run that records its interpreter; do not backfill an old manifest.')
    if Path(executable).resolve() != Path(sys.executable).resolve():
        raise ValueError(f"Use the primary run's recorded interpreter: {executable}")
    if policy['rounds'] != 3 or policy['latency_count'] != 30 or policy['quality_count'] != 650 or not policy['decode_on_cpu']:
        raise ValueError('The primary comparison must retain the three-round 30/650 CPU-decoder campaign conditions.')
    for side in ['baseline', 'candidate']:
        build = manifest['builds'][side]
        if not build['source_verified'] or compare.sha(build['speech']['path']) != build['speech']['sha256']:
            raise ValueError('Every supplement binary must match its verified primary build.')
        environment = manifest['execution_environments'][side]
        compare.environment([f'{key}={value}' for key, value in environment.items()])
    models = {m['type']: m for m in data['models']}
    for kind in manifest['types']:
        if models[kind]['sha256'] != manifest['models'][kind]['sha256']:
            raise ValueError('A supplement model differs from the primary comparison.')
    return manifest


def run_worker(directory, spec):
    subprocess.run(['/usr/bin/lockf', '-k', compare.LOCK, sys.executable, '-B',
                    str(directory / 'runner-snapshot/compare.py'), '_worker', '--spec', str(spec)], check=True)


def run(args):
    data = inputs(args.inputs); source = primary(args.primary_run, data)
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    snapshot = out / 'runner-snapshot'; snapshot.mkdir()
    for name in ['short.py', 'compare.py', 'worker.py']:
        shutil.copyfile(HERE / name, snapshot / name)
    policy = source['conditions']
    manifest = {'scope': SCOPE, 'measure_id': source['measure_id'], 'types': source['types'],
                'primary_run_sha256': compare.sha(Path(args.primary_run) / 'run.json'),
                'primary_summary_sha256': compare.sha(Path(args.primary_run) / 'summary.json'),
                'input_manifest_sha256': compare.sha(args.inputs), 'input_hash': data['input_hash'],
                'inputs': [{k: v for k, v in r.items() if k != 'pcm_path'} for r in data['records']],
                'builds': {k: source['builds'][k] for k in ['baseline', 'candidate']}, 'models': source['models'],
                'execution_environments': {k: source['execution_environments'][k] for k in ['baseline', 'candidate']},
                'conditions': {'rounds': 3, 'latency_count': 6, 'quality_count': 0, 'threads': policy['threads'],
                    'decode_on_cpu': True, 'timestamps': False, 'python': platform.python_version(), 'python_executable': sys.executable,
                    'warmup': 'All six prefixes once per fresh worker, longest first.', 'order': 'AB/BA/AB; input order shifts by round * 7.',
                    'latency_definition': policy['latency_definition'], 'aggregation': policy['aggregation'],
                    'accuracy_reference': 'unavailable', 'literal_cer': None, 'repository_cer': None,
                    'limitations': LIMITATIONS},
                'created_utc': datetime.now(timezone.utc).isoformat(),
                'runner_sha256': {p.name: compare.sha(p) for p in snapshot.iterdir()}}
    compare.write(out / 'run.json', manifest)
    models = {m['type']: m for m in data['models']}
    for kind in manifest['types']:
        for rnd in range(3):
            order = ['baseline', 'candidate'] if rnd % 2 == 0 else ['candidate', 'baseline']
            for side in order:
                spec = {'directory': str(out / kind / side / f'latency-{rnd}'), 'build': manifest['builds'][side],
                        'model': models[kind], 'threads': policy['threads'], 'phase': 'latency', 'round': rnd, 'side': side,
                        'environment': manifest['execution_environments'][side], 'warmup': data['records'], 'records': data['records']}
                path = out / f'{kind}-{side}-{rnd}.json'; compare.write(path, spec); run_worker(out, path)
    summarize(out)


def summarize(directory):
    manifest = compare.read(directory / 'run.json'); profiles = {}; all_exact = True
    if (directory / 'summary.json').exists():
        raise ValueError('Do not overwrite a recorded short supplement.')
    expected = {r['file'] for r in manifest['inputs']}
    for kind in manifest['types']:
        sides, paired = {}, {}
        for side in ['baseline', 'candidate']:
            runs = [compare.read(directory / kind / side / f'latency-{rnd}/result.json') for rnd in range(3)]
            groups = defaultdict(list)
            for rnd, run in enumerate(runs):
                if run['round'] != rnd or run['side'] != side or run['type'] != kind or run['phase'] != 'latency' or run['warmup_count'] != 6:
                    raise ValueError('Each short-input worker must warm all six prefixes and record its exact phase, side and round.')
                if run['build'] != manifest['builds'][side] or run['environment'] != manifest['execution_environments'][side] or run['worker_source_sha256'] != manifest['runner_sha256']['worker.py']:
                    raise ValueError('Supplement worker identity differs from the frozen run manifest.')
                if len(run['records']) != 6 or {r['file'] for r in run['records']} != expected:
                    raise ValueError('Every short-input round must contain the same six unique prefixes.')
                for row in run['records']:
                    if row['reference'] is not None or row['tokens'] is not None or row['segments'] is not None:
                        raise ValueError('Short prefixes have no CER reference or timestamp-enabled quality outputs.')
                    groups[row['file']].append(row)
            medians = [{**rows[0], 'seconds': statistics.median(r['seconds'] for r in rows)} for rows in groups.values()]
            repeated = sum(len({r['text'] for r in rows}) > 1 for rows in groups.values())
            all_exact = all_exact and repeated == 0
            sides[side] = {'latency': compare.latency(medians),
                          'strata': {s: compare.latency([r for r in medians if r['stratum'] == s]) for s in ['2s', '4s']},
                          'repeated_text_changes': repeated, 'literal_cer': None, 'repository_cer': None,
                          'peak_memory_bytes': max(r['memory']['peak_memory_footprint'] for r in runs),
                          'load_ms': statistics.median(r['load_seconds'] for r in runs) * 1000,
                          'first_request_ms': statistics.median(r['first_request']['seconds'] for r in runs) * 1000}
            paired[side] = {'runs': runs, 'medians': {r['file']: r for r in medians}}
        rounds = []
        for rnd in range(3):
            a, b = [{r['file']: r for r in paired[side]['runs'][rnd]['records']} for side in ['baseline', 'candidate']]
            rounds.append({'round': rnd, 'count': 6, 'text_changed': sum(a[f]['text'] != b[f]['text'] for f in expected)})
        changed = sum(r['text_changed'] for r in rounds); all_exact = all_exact and changed == 0
        ratios = [paired['candidate']['medians'][f]['seconds'] / paired['baseline']['medians'][f]['seconds'] for f in expected]
        profiles[kind] = {'sides': sides, 'comparison': {'count': 18, 'text_changed': changed, 'rounds': rounds,
                           'median_latency_ratio': statistics.median(ratios), 'latency_reduction_percent': (1 - statistics.median(ratios)) * 100}}
    summary = {**manifest, 'run_manifest_sha256': compare.sha(directory / 'run.json'), 'types': profiles,
               'status': 'exact_text_matched' if all_exact else 'text_changed', 'adoption_evidence_alone': False}
    compare.write(directory / 'summary.json', summary)
    print(directory / 'summary.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__); commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare'); p.add_argument('--assets', required=True); p.add_argument('--out', required=True)
    p = commands.add_parser('run'); p.add_argument('--primary-run', required=True); p.add_argument('--inputs', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args)
    else: run(args)


if __name__ == '__main__':
    main()
