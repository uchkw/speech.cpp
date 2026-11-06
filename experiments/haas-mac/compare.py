"""Run the fork's bounded Haas campaign and record one measure.

Usage: compare.py assets --previous PATH --out PATH
       compare.py build --source WORKTREE --out BUILD_DIRECTORY
       compare.py run --measure 001 --baseline BUILD_JSON --candidate BUILD_JSON --anchor BUILD_JSON
                      --assets ASSETS_JSON --types q4_k,q8_0 --out RUN_DIRECTORY
       compare.py record --measure 001 --run RUN_DIRECTORY --checks CHECKS_JSON
                         --verdict conditional --reason TEXT
Large evidence stays outside reports/. No command pushes, merges or changes a model default.
"""
import argparse
import array
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import unicodedata

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REPORTS = REPO / 'reports/haas-mac'
LOCK = '/private/tmp/speech-haas-mac-perf.lock'
EXPERIMENT_ENVIRONMENT = {'GGML_METAL_MM_TILE_N'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args], text=True).strip()


def locked(arguments):
    subprocess.run(['/usr/bin/lockf', '-k', LOCK, sys.executable, str(Path(__file__).resolve()), *arguments], check=True)


def environment(values):
    result = {}
    for value in values:
        key, separator, setting = value.partition('=')
        if not separator or key not in EXPERIMENT_ENVIRONMENT:
            raise ValueError('Only explicitly allowed experiment environment variables can be recorded.')
        result[key] = setting
    for key in EXPERIMENT_ENVIRONMENT:
        if key in os.environ and key not in result:
            raise ValueError(f'{key} is inherited but not explicitly recorded; pass its value for each side.')
    return result


def build(args):
    source, out = Path(args.source).resolve(), Path(args.out).resolve()
    if git(source, 'status', '--porcelain'):
        raise ValueError('Build provenance requires a clean source worktree, including its submodule.')
    commit = git(source, 'rev-parse', 'HEAD')
    ggml = git(source / 'ggml', 'rev-parse', 'HEAD')
    if ggml != git(source, 'rev-parse', 'HEAD:ggml'):
        raise ValueError('The ggml checkout does not match the source commit.')
    out.mkdir(parents=True, exist_ok=False)
    configure = ['cmake', '-S', str(source), '-B', str(out), '-DCMAKE_BUILD_TYPE=Release',
                 '-DGGML_NATIVE=ON', '-DGGML_METAL=ON']
    compile_command = ['cmake', '--build', str(out), '--target', 'speech-cli', 'fastconformer-timing', '-j4']
    with (out / 'build.log').open('w') as log:
        for command in [configure, compile_command]:
            log.write(' '.join(command) + '\n'); log.flush()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    if git(source, 'rev-parse', 'HEAD') != commit or git(source, 'status', '--porcelain'):
        raise ValueError('The source changed during the build; no verified manifest is written.')
    manifest = {'source_commit': commit, 'source_tree': git(source, 'rev-parse', 'HEAD^{tree}'),
                'source_path': str(source), 'source_verified': True, 'ggml_commit': ggml,
                'created_utc': datetime.now(timezone.utc).isoformat(), 'commands': [configure, compile_command],
                'cmake_cache_sha256': sha(out / 'CMakeCache.txt'), 'build_log_sha256': sha(out / 'build.log'),
                'stage_threads_argument': '<runs> <threads>' in (source / 'measure/fastconformer-timing.cpp').read_text()}
    for key, name in [('speech', 'speech'), ('stage_tool', 'fastconformer-timing')]:
        manifest[key] = {'path': str(out / name), 'sha256': sha(out / name)}
    write(out / 'build.json', manifest)
    print(out / 'build.json', flush=True)


def assets(args):
    previous, out = Path(args.previous).resolve(), Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / '.gitignore').write_text('*\n')
    corpus, subset = read(previous / 'corpus.json'), read(previous / 'subset.json')
    if len(corpus) != 650 or len(subset) != 30 or len({r['file'] for r in corpus}) != 650:
        raise ValueError('The corpus must contain 650 unique files and the subset 30.')
    for r in corpus:
        if sha(r['pcm_path']) != r['pcm_sha256']:
            raise ValueError(f"PCM hash mismatch: {r['file']}")
        r['stratum'] = '6-10s' if r['seconds'] <= 10 else '10-20s' if r['seconds'] <= 20 else '20-30s'
    models = [m for m in read(previous / 'models.json') if m['type'] in ['q4_k', 'q8_0', 'f16', 'q6_k', 'q5_k']]
    for m in models:
        if sha(m['path']) != m['sha256']:
            raise ValueError(f"Model hash mismatch: {m['type']}")
    stages = []
    for target in [8, 15, 25]:
        r = min(subset, key=lambda r: abs(r['seconds'] - target))
        directory = out / 'stage-audio' / Path(r['file']).stem
        directory.mkdir(parents=True)
        samples = array.array('h', Path(r['pcm_path']).read_bytes())
        if sys.byteorder != 'little':
            samples.byteswap()
        floats = array.array('f', (value / 32768.0 for value in samples))
        if sys.byteorder != 'little':
            floats.byteswap()
        header = repr({'descr': '<f4', 'fortran_order': False, 'shape': (len(floats),)})
        header = (header + ' ' * ((64 - (10 + len(header) + 1) % 64) % 64) + '\n').encode('ascii')
        path = directory / 'audio.npy'
        path.write_bytes(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + floats.tobytes())
        stages.append({'file': r['file'], 'directory': str(directory), 'seconds': r['seconds'],
                       'pcm_sha256': r['pcm_sha256'], 'npy_sha256': sha(path), 'kind': 'Benchmark audio only, not official reference tensors.'})
    identity = [{k: r[k] for k in ['file', 'reference', 'samples', 'pcm_sha256']} for r in corpus]
    write(out / 'assets.json', {'models': models, 'corpus': corpus, 'subset': subset, 'stages': stages,
                              'input_hash': hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode()).hexdigest(),
                              'source_environment': read(previous / 'environment.json')})
    print(out / 'assets.json', flush=True)


def plain(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if unicodedata.category(c)[0] not in 'PSZ' and not c.isspace())


def distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def score(records):
    result = {'literal_edits': 0, 'literal_characters': 0, 'repository_edits': 0, 'repository_characters': 0}
    for r in records:
        for name, reference, text in [('literal', r['reference'], r['text']), ('repository', plain(r['reference']), plain(r['text']))]:
            result[name + '_edits'] += distance(reference, text)
            result[name + '_characters'] += len(reference)
    for name in ['literal', 'repository']:
        result[name + '_cer'] = result[name + '_edits'] / result[name + '_characters']
    return result


def percentile(values, p):
    values = sorted(values); at = (len(values) - 1) * p; lo, hi = math.floor(at), math.ceil(at)
    return values[lo] if lo == hi else values[lo] * (hi - at) + values[hi] * (at - lo)


def latency(records):
    values = [r['seconds'] for r in records]
    return {**{f'p{int(p * 100)}_ms': percentile(values, p) * 1000 for p in [.5, .9, .99]},
            'count': len(values), 'rtfx': sum(r['audio_seconds'] for r in records) / sum(values)}


def paired(a, b):
    left, right = {r['file']: r for r in a}, {r['file']: r for r in b}
    if len(left) != len(a) or len(right) != len(b) or set(left) != set(right):
        raise ValueError('Paired inputs contain missing, duplicate or different file names.')
    changed = []
    counts = {name: 0 for name in ['text_changed', 'public_tokens_changed', 'segments_changed',
                                  'literal_better', 'literal_worse', 'literal_equal',
                                  'repository_better', 'repository_worse', 'repository_equal']}
    for file, x in left.items():
        y = right[file]
        if x['reference'] != y['reference']:
            raise ValueError('Paired references differ.')
        differences = {'text_changed': x['text'] != y['text'], 'public_tokens_changed': x.get('tokens') != y.get('tokens'),
                       'segments_changed': x.get('segments') != y.get('segments')}
        counts.update({key: counts[key] + int(value) for key, value in differences.items()})
        for name, transform in [('literal', lambda s: s), ('repository', plain)]:
            delta = distance(transform(x['reference']), transform(y['text'])) - distance(transform(x['reference']), transform(x['text']))
            counts[name + ('_better' if delta < 0 else '_worse' if delta > 0 else '_equal')] += 1
        if any(differences.values()):
            changed.append({'file': file, 'baseline': x, 'candidate': y, **differences})
    ratios = [right[file]['seconds'] / x['seconds'] for file, x in left.items()]
    return {**counts, 'count': len(left), 'median_latency_ratio': statistics.median(ratios)}, changed


def summarize(directory):
    manifest = read(directory / 'run.json')
    profiles = {}
    for kind in manifest['types']:
        sides, quality_rows, medians = {}, {}, {}
        for side in manifest['builds']:
            runs = [read(p) for p in sorted((directory / kind / side).glob('latency-*/result.json'))]
            quality = read(directory / kind / side / 'quality-0/result.json')
            by_file = defaultdict(list)
            for run in runs:
                for record in run['records']:
                    by_file[record['file']].append(record)
            rows = [{**items[0], 'seconds': statistics.median(r['seconds'] for r in items)} for items in by_file.values()]
            medians[side], quality_rows[side] = rows, quality['records']
            stages_path = directory / kind / side / 'stages.jsonl'
            stages = [json.loads(line) for line in stages_path.read_text().splitlines()] if stages_path.exists() else []
            stage_groups = defaultdict(list)
            for stage in stages:
                stage_groups[stage['input']].append(stage)
            stage_summary = {name: {key: statistics.median(r[key] for r in records) for key in records[0] if key.endswith('_ms')}
                             for name, records in stage_groups.items()}
            sides[side] = {'latency': latency(rows), 'strata': {s: latency([r for r in rows if r['stratum'] == s]) for s in sorted({r['stratum'] for r in rows})},
                           'rounds': [latency(r['records']) for r in runs], 'quality': score(quality['records']), 'stages': stage_summary,
                           'stage_status': 'measured' if stages else 'unavailable',
                           'peak_memory_bytes': max(r['memory']['peak_memory_footprint'] for r in [*runs, quality]),
                           'load_ms': statistics.median(r['load_seconds'] for r in runs) * 1000,
                           'first_request_ms': statistics.median(r['first_request']['seconds'] for r in runs) * 1000,
                           'repeated_text_changes': sum(len({r['text'] for r in items}) > 1 for items in by_file.values())}
        comparisons = {}
        for label, side in [('relative', 'baseline'), ('cumulative', 'anchor' if 'anchor' in sides else 'baseline')]:
            counts, changes = paired(quality_rows[side], quality_rows['candidate'])
            timing, _ = paired(medians[side], medians['candidate'])
            counts['median_latency_ratio'] = timing['median_latency_ratio']
            counts['latency_reduction_percent'] = (1 - timing['median_latency_ratio']) * 100
            write(directory / kind / f'{label}-changes.json', changes)
            comparisons[label] = counts
        profiles[kind] = {'sides': sides, 'comparisons': comparisons}
    result = {'measure_id': manifest['measure_id'], 'status': 'conditional', 'reason': 'Measurements await review and required checks.',
              'scope': manifest['scope'], 'conditions': manifest['conditions'], 'types': profiles,
              'builds': manifest['builds'], 'execution_environments': manifest['execution_environments'], 'models': manifest['models'],
              'assets_sha256': manifest['assets_sha256'], 'input_hash': manifest['input_hash'],
              'evidence': {'directory': str(directory), 'run_manifest_sha256': sha(directory / 'run.json')}, 'checks': []}
    write(directory / 'summary.json', result)


def run_comparison(args):
    directory = Path(args.out).resolve()
    data, policy = read(args.assets), read(HERE / 'campaign.json')
    builds = {'baseline': read(args.baseline), 'candidate': read(args.candidate)}
    environments = {'baseline': environment(args.baseline_env), 'candidate': environment(args.candidate_env)}
    if args.anchor:
        anchor = read(args.anchor)
        anchor_environment = environment(args.anchor_env)
        if anchor['speech']['sha256'] != builds['baseline']['speech']['sha256'] or anchor_environment != environments['baseline']:
            builds['anchor'] = anchor
            environments['anchor'] = anchor_environment
    for b in builds.values():
        if not b.get('source_verified') or any(sha(b[k]['path']) != b[k]['sha256'] for k in ['speech', 'stage_tool']):
            raise ValueError('Every binary must match a verified build manifest.')
        if not args.smoke and not b['stage_threads_argument']:
            raise ValueError('Full comparisons require the stage tool with explicit threads.')
    for stage in data['stages']:
        if sha(Path(stage['directory']) / 'audio.npy') != stage['npy_sha256']:
            raise ValueError(f"Stage audio changed: {stage['file']}")
    kinds = args.types.split(',')
    if len(set(kinds)) != len(kinds) or any(kind not in {m['type'] for m in data['models']} for kind in kinds):
        raise ValueError('Choose distinct model types present in the assets manifest.')
    directory.mkdir(parents=True, exist_ok=False)
    subset = data['subset'] if not args.smoke else [data['subset'][i] for i in [0, 14, 29]]
    corpus = data['corpus'] if not args.smoke else subset
    manifest = {'measure_id': args.measure, 'scope': 'smoke' if args.smoke else 'full', 'types': kinds,
                'builds': builds, 'execution_environments': environments,
                'models': {m['type']: {k: m[k] for k in ['type', 'sha256', 'size', 'revision'] if k in m} for m in data['models'] if m['type'] in args.types.split(',')},
                'assets_sha256': sha(args.assets), 'input_hash': data['input_hash'],
                'conditions': {**policy, 'quality_policy': args.quality_policy, 'latency_count': len(subset), 'quality_count': len(corpus),
                               'hardware': data['source_environment']['hardware'],
                               'current_machine': {key: subprocess.check_output(command, text=True).strip() for key, command in {
                                   'chip': ['sysctl', '-n', 'machdep.cpu.brand_string'], 'cpu_cores': ['sysctl', '-n', 'hw.ncpu'],
                                   'memory_bytes': ['sysctl', '-n', 'hw.memsize'], 'macos': ['sw_vers', '-productVersion']}.items()},
                               'python': platform.python_version(),
                               'runner_sha256': {name: sha(HERE / name) for name in ['compare.py', 'worker.py', 'campaign.json']},
                               'created_utc': datetime.now(timezone.utc).isoformat()}}
    write(directory / 'run.json', manifest)
    try:
        for kind in manifest['types']:
            model = next(m for m in data['models'] if m['type'] == kind)
            if sha(model['path']) != model['sha256']:
                raise ValueError(f'Model changed: {kind}')
            sides = list(builds)
            for phase, rounds, records in [('latency', policy['rounds'], subset), ('quality', 1, corpus)]:
                for round_index in range(rounds):
                    order = sides[round_index % len(sides):] + sides[:round_index % len(sides)]
                    for side in order:
                        destination = directory / kind / side / f'{phase}-{round_index}'
                        spec_path = directory / f'{kind}-{side}-{phase}-{round_index}.json'
                        write(spec_path, {'directory': str(destination), 'build': builds[side], 'model': model, 'threads': policy['threads'],
                                          'phase': phase, 'round': round_index, 'side': side, 'environment': environments[side], 'warmup': subset, 'records': records})
                        locked(['_worker', '--spec', str(spec_path)])
            for side, b in builds.items():
                if not b['stage_threads_argument']:
                    continue
                path = directory / kind / side / 'stages.jsonl'
                command = ['/usr/bin/lockf', '-k', LOCK, b['stage_tool']['path'], model['path'], 'gpu', 'greedy', 'cpu',
                           str(policy['rounds']), str(policy['threads']), *[r['directory'] for r in data['stages']]]
                with path.open('w') as output, path.with_suffix('.stderr.log').open('w') as log:
                    subprocess.run(command, stdout=output, stderr=log, env={**os.environ, **environments[side]}, check=True)
        summarize(directory)
    except Exception as error:
        write(directory / 'failure.json', {'status': 'blocked', 'error': str(error), 'run_manifest': str(directory / 'run.json')})
        raise
    print(directory / 'summary.json', flush=True)


def record(args):
    paths = list(REPORTS.glob(args.measure + '-*.json'))
    if not paths and re.fullmatch(r'\d{3}[a-z]', args.measure) and args.title:
        parent = list(REPORTS.glob(args.measure[:3] + '-*.json'))
        if len(parent) != 1:
            raise ValueError('A trial requires an existing parent measure.')
        slug = re.sub('[^a-z0-9]+', '-', args.title.lower()).strip('-')
        paths = [REPORTS / f'{args.measure}-{slug}.json']
        entry = {'schema_version': 1, 'measure_id': args.measure, 'parent_measure_id': args.measure[:3],
                 'title': args.title, 'status': 'pending', 'reason': 'Not evaluated.'}
    elif len(paths) == 1:
        entry = read(paths[0])
    else:
        raise ValueError('Choose an existing measure, or a new trial such as 001a with --title.')
    result = read(Path(args.run) / 'summary.json') if args.run else {}
    if entry.get('evidence') and result.get('evidence') and entry['evidence']['run_manifest_sha256'] != result['evidence']['run_manifest_sha256']:
        raise ValueError('Do not overwrite a measured trial; give the next run a new trial id.')
    checks = read(args.checks)['checks'] if args.checks else []
    allowed = {'passed', 'failed', 'refused', 'skipped', 'unavailable'}
    if any(c['status'] not in allowed for c in checks):
        raise ValueError('Checks must explicitly distinguish passed, failed, refused, skipped and unavailable.')
    if args.verdict in ['accepted', 'validated_mac']:
        required = [c for c in checks if c.get('required', True) and (args.verdict == 'accepted' or c.get('scope', 'mac') == 'mac')]
        if result.get('scope') != 'full' or not required or any(c['status'] != 'passed' for c in required):
            raise ValueError('Acceptance requires a full run and every required check passed; refusals and skips are not passes.')
        for profile in result['types'].values():
            count = profile['comparisons']['relative']
            if result['conditions']['quality_policy'] == 'exact' and any(count[k] for k in ['text_changed', 'public_tokens_changed', 'segments_changed']):
                raise ValueError('Exact-output policy does not allow changed text, public tokens or times.')
            old, new = profile['sides']['baseline']['quality'], profile['sides']['candidate']['quality']
            if any(new[k] > old[k] for k in ['literal_cer', 'repository_cer']):
                raise ValueError('Acceptance does not allow a CER regression; use conditional and state the tradeoff.')
    if result:
        if not args.artifact_root:
            raise ValueError('Recording a run requires --artifact-root for portable local evidence paths.')
        artifact_root = Path(args.artifact_root).resolve()
        for b in result['builds'].values():
            b.pop('source_path', None)
            b.pop('commands', None)
            for key in ['speech', 'stage_tool']:
                b[key]['artifact_path'] = os.path.relpath(b[key].pop('path'), artifact_root)
        result['evidence']['directory'] = os.path.relpath(result['evidence']['directory'], artifact_root)
        result['evidence']['paths_relative_to'] = 'External campaign artifact directory supplied to compare.py record.'
    entry.update(result); entry.update({'status': args.verdict, 'reason': args.reason, 'checks': checks,
                                       'updated_utc': datetime.now(timezone.utc).isoformat()})
    if '/Users/' in json.dumps(entry):
        raise ValueError('Versioned reports must not contain absolute user-specific host paths.')
    write(paths[0], entry)
    subprocess.run([sys.executable, str(HERE / 'render.py')], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('assets'); p.add_argument('--previous', required=True); p.add_argument('--out', required=True)
    for name in ['build', '_build']:
        p = commands.add_parser(name); p.add_argument('--source', required=True); p.add_argument('--out', required=True)
    p = commands.add_parser('_worker'); p.add_argument('--spec', required=True)
    p = commands.add_parser('run')
    for name in ['measure', 'baseline', 'candidate', 'assets', 'types', 'out']:
        p.add_argument('--' + name, required=True)
    p.add_argument('--anchor'); p.add_argument('--smoke', action='store_true')
    for side in ['baseline', 'candidate', 'anchor']:
        p.add_argument('--' + side + '-env', action='append', default=[])
    p.add_argument('--quality-policy', choices=['exact', 'cer'], default='exact')
    p = commands.add_parser('record'); p.add_argument('--measure', required=True); p.add_argument('--run'); p.add_argument('--checks')
    p.add_argument('--title')
    p.add_argument('--artifact-root')
    p.add_argument('--verdict', choices=['pending', 'in_progress', 'accepted', 'rejected', 'conditional', 'blocked', 'not_applicable', 'validated_mac', 'pending_official_reference'], required=True)
    p.add_argument('--reason', required=True)
    args = parser.parse_args()
    if args.command == 'assets': assets(args)
    elif args.command == 'build': locked(['_build', '--source', args.source, '--out', args.out])
    elif args.command == '_build': build(args)
    elif args.command == '_worker':
        from worker import run
        run(read(args.spec))
    elif args.command == 'run': run_comparison(args)
    elif args.command == 'record': record(args)


if __name__ == '__main__':
    main()
