"""Prepare or execute the isolated CPU-thread sweep with a pinned campaign worker.

Usage: 021-thread-sweep.py prepare --baseline BUILD_JSON --candidate BUILD_JSON --assets ASSETS_JSON
           --machine-default 16 --out DIRECTORY [--types q4_k] [--quality SELECTED_CONDITION]
       021-thread-sweep.py run DIRECTORY
Preparation does not load models or run inference. Execution requires the coordinator's compute lease.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


sys.dont_write_bytecode = True
FLAG = 'SPEECH_HAAS_AUTO_THREADS'
RUNNER_COMMIT = 'a239cdb483ff90e05969bdecfb5efe39137119da'
WORKER_SHA = '46483a7bde5bf3d09f0898224b71e99288f17acec33912a4b09752b0acc09c9b'
GGML_COMMIT = 'e1a9ac12edbb57402618fd6936598603f5ee6266'
LOCK = '/private/tmp/speech-haas-mac-perf.lock'
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditions(machine_default):
    rows = [('baseline-auto', 'baseline', 0, 0, machine_default),
            ('baseline-4', 'baseline', 0, 4, 4),
            ('candidate-auto-off', 'candidate', 0, 0, machine_default),
            ('candidate-auto-on', 'candidate', 1, 0, min(4, machine_default))]
    rows.extend((f'candidate-{count}', 'candidate', 1, count, count) for count in [1, 2, 4, 8, 16])
    return {name: {'build': build, 'environment': {FLAG: str(flag)}, 'threads': count, 'expected_threads': expected}
            for name, build, flag, count, expected in rows}


def prepare(args):
    if args.machine_default < 1:
        raise ValueError('machine-default must be positive and verified by the ordinary C API load check.')
    builds = {'baseline': read(args.baseline), 'candidate': read(args.candidate)}
    for build in builds.values():
        if not build.get('source_verified') or build['ggml_commit'] != GGML_COMMIT:
            raise ValueError('Both builds must have verified clean source and the unchanged experiment ggml.')
    if builds['baseline']['build_flags'] != builds['candidate']['build_flags']:
        raise ValueError('The two builds must use the same configuration and compiler build flags.')
    data = read(args.assets)
    kinds = args.types.split(',')
    models = {item['type']: item for item in data['models']}
    if not kinds or len(set(kinds)) != len(kinds) or any(kind not in models for kind in kinds):
        raise ValueError('Choose distinct model types present in the assets manifest.')
    variants = conditions(args.machine_default)
    if args.quality:
        if args.quality not in variants or not args.quality.startswith('candidate-'):
            raise ValueError('The selected quality condition must be one of the candidate screen conditions.')
        variants = {name: variants[name] for name in ['baseline-auto', 'baseline-4', args.quality]}
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    runner = directory / 'runner'
    runner.mkdir()
    snapshots = {}
    for name in ['compare.py', 'worker.py', 'campaign.json']:
        content = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                           f'{RUNNER_COMMIT}:experiments/haas-mac/{name}'])
        (runner / name).write_bytes(content)
        snapshots[name] = sha(runner / name)
    if snapshots['worker.py'] != WORKER_SHA:
        raise ValueError('The frozen worker does not match the reviewed auto-thread runner.')
    sys.path.insert(0, str(runner))
    spec = importlib.util.spec_from_file_location('thread_sweep_compare', runner / 'compare.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    policy = read(runner / 'campaign.json')
    screen = compare.screen_inputs(data['subset'])
    phase, rounds = ('quality', 1) if args.quality else ('latency', policy['rounds'])
    records = data['corpus'] if args.quality else screen
    jobs = []
    for kind in kinds:
        for round_index in range(rounds):
            names = list(variants)
            names = names[round_index % len(names):] + names[:round_index % len(names)]
            for name in names:
                condition = variants[name]
                compare.execution_environment(condition['environment'])
                destination = directory / kind / name / f'{phase}-{round_index}'
                spec_path = directory / f'{kind}-{name}-{phase}-{round_index}.json'
                write(spec_path, {'directory': str(destination), 'build': builds[condition['build']],
                                  'model': models[kind], 'threads': condition['threads'],
                                  'expected_threads': condition['expected_threads'], 'phase': phase,
                                  'round': round_index, 'side': name, 'environment': condition['environment'],
                                  'warmup': screen, 'records': records})
                jobs.append({'spec': str(spec_path), 'sha256': sha(spec_path),
                             'command': ['/usr/bin/lockf', '-k', LOCK, sys.executable,
                                         str(runner / 'compare.py'), '_worker', '--spec', str(spec_path)]})
    write(directory / 'plan.json', {'measure_id': '021', 'status': 'prepared, not executed',
          'created_utc': datetime.now(timezone.utc).isoformat(), 'phase': phase, 'rounds': rounds,
          'types': kinds, 'conditions': variants, 'builds': builds,
          'source_environment': data['source_environment'], 'input_hash': data['input_hash'],
          'assets_sha256': sha(args.assets), 'baseline_manifest_sha256': sha(args.baseline),
          'candidate_manifest_sha256': sha(args.candidate), 'runner_commit': RUNNER_COMMIT,
          'runner_sha256': snapshots, 'driver_sha256': sha(__file__), 'jobs': jobs,
          'comparisons': {'default_user': 'baseline-auto', 'tuned_campaign': 'baseline-4'},
          'aggregation': policy['aggregation'],
          'scope': 'Screen timing uses nine inputs and three rounds. Full quality is a separate selected-condition run. '
                   'No API auto-selection claim comes from the stage timing tool. No cumulative code gain is reported.'})
    print(directory / 'plan.json')


def run(directory):
    directory = Path(directory).resolve()
    plan = read(directory / 'plan.json')
    if plan['driver_sha256'] != sha(__file__):
        raise ValueError('The sweep driver changed after preparation.')
    for name, expected in plan['runner_sha256'].items():
        if sha(directory / 'runner' / name) != expected:
            raise ValueError(f'The frozen runner changed: {name}')
    if plan['runner_sha256']['worker.py'] != WORKER_SHA:
        raise ValueError('The plan does not use the reviewed worker.')
    for job in plan['jobs']:
        if sha(job['spec']) != job['sha256']:
            raise ValueError('A worker specification changed after preparation.')
    try:
        for job in plan['jobs']:
            subprocess.run(job['command'], check=True)
    except Exception as error:
        write(directory / 'execution.json', {'status': 'failed', 'error': str(error)})
        raise
    write(directory / 'execution.json', {'status': 'executed, awaiting output and performance analysis',
          'plan_sha256': sha(directory / 'plan.json'), 'jobs': len(plan['jobs']),
          'results': [{'path': str(Path(read(job['spec'])['directory']) / 'result.json'),
                       'sha256': sha(Path(read(job['spec'])['directory']) / 'result.json')} for job in plan['jobs']]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for name in ['baseline', 'candidate', 'assets', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--machine-default', required=True, type=int)
    p.add_argument('--types', default='q4_k')
    p.add_argument('--quality')
    p = commands.add_parser('run')
    p.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        run(args.directory)
