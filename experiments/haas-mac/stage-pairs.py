"""Compare existing FastConformer stage binaries in three alternating rounds.

Usage: stage-pairs.py --run PRIMARY_RUN_JSON --assets ASSETS_JSON --out NEW_DIR
This supplementary stage measurement does not replace worker request latency.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys

LOCK = '/private/tmp/speech-haas-mac-perf.lock'
METRICS = ['decoder_ms', 'encoder_compute_ms', 'core_total_ms']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['run', 'assets', 'out']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    primary, assets = read(args.run), read(args.assets)
    if primary['assets_sha256'] != sha(args.assets):
        raise ValueError('The assets differ from the primary comparison.')
    if primary['conditions']['python'] != platform.python_version() or primary['conditions']['python_executable'] != sys.executable:
        raise ValueError('Use the primary comparison Python version and executable.')
    if any(k.startswith('SPEECH_HAAS_') or k == 'GGML_METAL_MM_TILE_N' for k in os.environ):
        raise ValueError('Do not inherit experiment variables; the primary manifest supplies them.')
    builds = {side: primary['builds'][side] for side in ['baseline', 'candidate']}
    for build in builds.values():
        if not build['source_verified'] or sha(build['stage_tool']['path']) != build['stage_tool']['sha256']:
            raise ValueError('A stage binary differs from its verified build.')
    for stage in assets['stages']:
        if sha(Path(stage['directory']) / 'audio.npy') != stage['npy_sha256']:
            raise ValueError('Stage input changed: ' + stage['file'])
    models = {m['type']: m for m in assets['models']}
    for kind in primary['types']:
        if sha(models[kind]['path']) != primary['models'][kind]['sha256']:
            raise ValueError('Model changed: ' + kind)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / 'stage-pairs.py')
    manifest = {'scope': 'stage_diagnostic', 'primary_run_sha256': sha(args.run), 'assets_sha256': sha(args.assets),
                'python': platform.python_version(), 'python_executable': sys.executable,
                'runner_sha256': sha(__file__), 'created_utc': datetime.now(timezone.utc).isoformat(),
                'builds': builds, 'execution_environments': primary['execution_environments'],
                'types': primary['types'], 'inputs': assets['stages'], 'rounds': 3, 'runs_per_input': 3,
                'threads': primary['conditions']['threads'],
                'warmup': 'Each input is warmed once immediately before its three measured stage runs in each fresh process.',
                'limitation': 'The worker warms all inputs before measurement. Separate stage and worker times cannot be subtracted to isolate I/O.',
                'observations': []}
    for rnd in range(3):
        stages = assets['stages'][rnd:] + assets['stages'][:rnd]
        for kind in primary['types']:
            for side in (['baseline', 'candidate'] if rnd % 2 == 0 else ['candidate', 'baseline']):
                stem = '{}-{}-{}'.format(kind, side, rnd)
                env = primary['execution_environments'][side]
                command = [builds[side]['stage_tool']['path'], models[kind]['path'], 'gpu', 'greedy', 'cpu', '3',
                           str(manifest['threads']), *[s['directory'] for s in stages]]
                with (out / (stem + '.jsonl')).open('w') as output, (out / (stem + '.stderr.log')).open('w') as log:
                    result = subprocess.run(['/usr/bin/lockf', '-k', LOCK, *command], stdout=output, stderr=log, env={**os.environ, **env})
                manifest['observations'].append({'type': kind, 'side': side, 'round': rnd, 'command': command,
                                                 'environment': env, 'returncode': result.returncode,
                                                 'model_sha256': models[kind]['sha256'], 'output': stem + '.jsonl'})
                write(out / 'run.json', manifest)
                print(stem, result.returncode, flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)
    summary = {'scope': 'stage_diagnostic', 'run_manifest_sha256': sha(out / 'run.json'),
               'method': 'Per-input medians of three warm runs in each process; paired ratios are computed per input and round. Round totals sum those three input medians.',
               'types': {}}
    expected = {Path(s['directory']).name for s in assets['stages']}
    for kind in primary['types']:
        rounds, comparisons = [], 0
        all_ratios = {metric: [] for metric in METRICS}
        for rnd in range(3):
            rows = {side: [json.loads(line) for line in (out / '{}-{}-{}.jsonl'.format(kind, side, rnd)).read_text().splitlines()]
                    for side in ['baseline', 'candidate']}
            for side, values in rows.items():
                if {(r['input'], r['run']) for r in values} != {(f, n) for f in expected for n in range(3)} or len(values) != 9:
                    raise ValueError('Incomplete or duplicate stage observations.')
                if any(r['threads'] != manifest['threads'] or r['decode_on'] != 'cpu' for r in values):
                    raise ValueError('Stage execution conditions differ from the manifest.')
            paired = {side: {(r['input'], r['run']): r for r in values} for side, values in rows.items()}
            for key, baseline in paired['baseline'].items():
                candidate = paired['candidate'][key]
                if any(baseline[field] != candidate[field] for field in ['text', 'tokens', 'graphs', 'audio_seconds']):
                    raise ValueError('Stage outputs differ at ' + str(key))
                comparisons += 1
            medians = {side: {f: {metric: statistics.median(r[metric] for r in values if r['input'] == f) for metric in METRICS}
                              for f in sorted(expected)} for side, values in rows.items()}
            ratios = {f: {metric: medians['candidate'][f][metric] / medians['baseline'][f][metric] for metric in METRICS} for f in sorted(expected)}
            for metric in METRICS:
                all_ratios[metric].extend(ratios[f][metric] for f in sorted(expected))
            totals = {side: {metric: sum(v[metric] for v in medians[side].values()) for metric in METRICS} for side in medians}
            rounds.append({'round': rnd, 'input_medians': medians, 'input_ratios': ratios, 'totals': totals,
                           'total_ratios': {metric: totals['candidate'][metric] / totals['baseline'][metric] for metric in METRICS}})
        summary['types'][kind] = {'rounds': rounds, 'median_input_round_ratios': {m: statistics.median(v) for m, v in all_ratios.items()},
                                  'output_comparisons': comparisons, 'output_changes': 0}
    write(out / 'summary.json', summary)
    print(out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
