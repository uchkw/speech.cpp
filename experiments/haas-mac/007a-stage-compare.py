"""Run the bounded 007a stage/path diagnostic against preserved product libraries.

Usage: 007a-stage-compare.py --assets ASSETS_JSON --candidate-build BUILD_JSON
       --anchor-build BUILD_JSON --out NEW_DIRECTORY
Each of three rounds rotates OFF, ON and anchor. Every invocation holds the shared lock.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

sys.dont_write_bytecode = True
LOCK = '/private/tmp/speech-haas-mac-perf.lock'
KINDS = ['q4_k', 'q8_0', 'f16']
PATH_KEYS = ['ids', 'frames', 'durations', 'tokens', 'graphs', 'text']
METRICS = ['encoder_compute_ms', 'core_total_ms', 'decoder_ms', 'frontend_ms']


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def summarize(out, records):
    types = {}
    all_exact = True
    for kind in KINDS:
        selected = [row for row in records if row['type'] == kind]
        by_side = defaultdict(lambda: defaultdict(list))
        path_reference = {}
        path_exact = True
        for record in selected:
            rows = [json.loads(line) for line in (out / record['stdout']).read_text().splitlines()]
            assert len(rows) == 9 and len({(row['input'], row['run']) for row in rows}) == 9
            for row in rows:
                assert row['threads'] == 4 and row['decode_on'] == 'cpu' and row['decoding'] == 'greedy'
                assert row['tokens'] == len(row['ids']) == len(row['frames']) == len(row['durations'])
                key = row['input']
                path = {key: row[key] for key in PATH_KEYS}
                if key not in path_reference:
                    path_reference[key] = path
                path_exact = path_exact and path == path_reference[key]
                by_side[record['side']][key].append({**row, 'round': record['round']})
        pairs = {}
        for label, left, right in [('on/off', 'off', 'on'), ('on/anchor', 'anchor', 'on'), ('off/anchor', 'anchor', 'off')]:
            assert by_side[left].keys() == by_side[right].keys() and len(by_side[left]) == 3
            results = {}
            for name in by_side[left]:
                a, b = by_side[left][name], by_side[right][name]
                assert len(a) == len(b) == 9
                results[name] = {}
                for metric in METRICS:
                    rounds = []
                    for index in range(3):
                        before = statistics.median(row[metric] for row in a if row['round'] == index)
                        after = statistics.median(row[metric] for row in b if row['round'] == index)
                        rounds.append({'round': index, 'before_ms': before, 'after_ms': after, 'ratio': after / before})
                    before = statistics.median(row[metric] for row in a)
                    after = statistics.median(row[metric] for row in b)
                    results[name][metric] = {'before_ms': before, 'after_ms': after, 'ratio': after / before, 'rounds': rounds}
            pairs[label] = results
        types[kind] = {'path_exact': path_exact, 'measured_outputs': 81, 'comparisons': pairs}
        all_exact = all_exact and path_exact
    result = {'status': 'passed' if all_exact else 'failed', 'all_paths_exact': all_exact, 'types': types,
              'scope': 'Alternating stage diagnostic, not worker request latency or official-reference accuracy.',
              'warmup': 'One untimed recognition immediately before three measured repeats of each input, per process.',
              'path_logging': 'JSON arrays are written after all timers end.', 'runs': records}
    write(out / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['assets', 'candidate-build', 'anchor-build', 'out']:
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    inherited = {key: value for key, value in os.environ.items() if key.startswith(('GGML_', 'SPEECH_'))}
    if inherited:
        raise ValueError('Unset inherited GGML_* and SPEECH_* values before this diagnostic: ' + ', '.join(inherited))
    assets = read(args.assets)
    builds = {'candidate': read(args.candidate_build), 'anchor': read(args.anchor_build)}
    for build in builds.values():
        assert sha(build['binary']['path']) == build['binary']['sha256']
    for stage in assets['stages']:
        assert sha(Path(stage['directory']) / 'audio.npy') == stage['npy_sha256']
    assert len(assets['stages']) == 3
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'assets_sha256': sha(args.assets), 'builds': builds, 'python': platform.python_version(),
                'python_executable': sys.executable, 'driver_sha256': sha(__file__), 'inherited_experiment_environment': {},
                'rounds': 3, 'repeats_per_input': 3, 'threads': 4, 'models': [], 'stages': assets['stages']}
    records = []
    for kind in KINDS:
        model = next(model for model in assets['models'] if model['type'] == kind)
        assert sha(model['path']) == model['sha256']
        manifest['models'].append(model)
        write(out / 'run.json', manifest)
        for index in range(3):
            sides = ['off', 'on', 'anchor']
            order = sides[index:] + sides[:index]
            for side in order:
                build = builds['anchor' if side == 'anchor' else 'candidate']
                env = {'GGML_METAL_ADD_SOFTMAX': '1' if side == 'on' else '0'}
                stem = f'{kind}-{side}-{index}'
                command = ['/usr/bin/lockf', '-k', LOCK, build['binary']['path'], model['path'], 'gpu', 'greedy', 'cpu',
                           '3', '4', *[stage['directory'] for stage in assets['stages']]]
                record = {'type': kind, 'side': side, 'round': index, 'order': order, 'command': command,
                          'environment': env, 'model_sha256': model['sha256'], 'binary_sha256': build['binary']['sha256'],
                          'stdout': stem + '.jsonl', 'stderr': stem + '.stderr.log',
                          'started_utc': datetime.now(timezone.utc).isoformat()}
                with (out / record['stdout']).open('w') as stdout, (out / record['stderr']).open('w') as stderr:
                    result = subprocess.run(command, env={**os.environ, **env}, stdout=stdout, stderr=stderr)
                record.update(returncode=result.returncode, completed_utc=datetime.now(timezone.utc).isoformat(),
                              stdout_sha256=sha(out / record['stdout']), stderr_sha256=sha(out / record['stderr']))
                records.append(record)
                write(out / 'records.json', records)
                print(kind, side, index, result.returncode, flush=True)
                if result.returncode:
                    raise SystemExit(result.returncode)
    result = summarize(out, records)
    print(out / 'summary.json', result['status'], flush=True)
    raise SystemExit(0 if result['all_paths_exact'] else 1)


if __name__ == '__main__':
    main()
