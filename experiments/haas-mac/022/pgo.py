"""Train and merge 022c profiles on audio disjoint from every campaign evaluation input.

Usage: pgo.py audit --training training.json --assets assets.json --short inputs.json
       pgo.py train --training training.json --assets assets.json --short inputs.json
                    --build instrumented-build.json --toolchain toolchain.json
                    --worker FROZEN_WORKER_PY --type q4_k|q8_0|f16 --out NEW_DIRECTORY
       pgo.py merge --run Q4/train.json --run Q8/train.json --run F16/train.json
                    --toolchain toolchain.json --out NEW_DIRECTORY
Each train command owns one serialized worker run. Training timings are never performance evidence.
"""
import argparse
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
import build

REVISION = '70bb2e84b976b7e960aa89f1c648e09c59f894dd'


def digest_string(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_split(training, assets, short):
    dataset = training['dataset']
    if dataset['repository'] != 'google/fleurs' or dataset['revision'] != REVISION or dataset['config'] != 'ja_jp':
        raise ValueError('Training requires the pinned official Japanese FLEURS dataset.')
    if dataset['split'] not in ('train', 'validation'):
        raise ValueError('Evaluation/test recordings cannot train PGO.')
    if not all(digest_string(dataset[key]) for key in ('archive_sha256', 'metadata_sha256')):
        raise ValueError('Pin the source audio archive and metadata by SHA-256.')
    if not all(isinstance(dataset[key], str) and dataset[key].startswith('https://') for key in ('archive_url', 'metadata_url')):
        raise ValueError('Record the HTTPS URLs of the pinned training archive and metadata.')
    if training['sample_rate'] != 16000 or training['format'] != 'mono PCM16 little endian':
        raise ValueError('Training inputs must match the campaign worker PCM format.')
    evaluated = assets['corpus'] + assets['subset'] + assets['stages'] + short['records']
    excluded_recordings, excluded_pcm, excluded_ids = set(), set(), set()
    for record in evaluated:
        excluded_recordings.update(str(record[key]) for key in ('file', 'source_file', 'source_recording_id') if key in record)
        excluded_pcm.update(record[key] for key in ('pcm_sha256', 'source_pcm_sha256') if key in record)
        if 'id' in record:
            excluded_ids.add(str(record['id']))
    records = training['records']
    if not 9 <= len(records) <= 30:
        raise ValueError('The bounded training subset must contain 9 to 30 recordings.')
    names, sources, pcm = set(), set(), set()
    strata = {'3-10s': 0, '10-20s': 0, '20-30s': 0}
    for record in records:
        source = str(record['source_recording_id'])
        digest = record['pcm_sha256']
        if not source or source in excluded_recordings or str(record['file']) in excluded_recordings:
            raise ValueError('A training recording or its source belongs to the evaluation set.')
        if str(record['id']) in excluded_ids or digest in excluded_pcm:
            raise ValueError('A training utterance ID or PCM belongs to the evaluation set.')
        if record['file'] in names or source in sources or digest in pcm or not digest_string(digest):
            raise ValueError('Training filenames, source recordings and PCM hashes must be distinct.')
        if type(record['samples']) is not int or not 48000 <= record['samples'] <= 480000:
            raise ValueError('Use complete training recordings of 3 to 30 seconds.')
        if record['seconds'] != record['samples'] / 16000 or record['start_sample'] != 0:
            raise ValueError('Training duration or recording origin is inconsistent.')
        if record['source_samples'] != record['samples']:
            raise ValueError('The training subset must not crop evaluation or other source recordings.')
        stratum = '3-10s' if record['seconds'] <= 10 else '10-20s' if record['seconds'] <= 20 else '20-30s'
        if record['stratum'] != stratum:
            raise ValueError('A training duration stratum is incorrect.')
        strata[stratum] += 1
        names.add(record['file']); sources.add(source); pcm.add(digest)
    if min(strata.values()) < 3:
        raise ValueError('Training requires at least three recordings in each duration stratum.')
    return {'recordings': len(records), 'strata': strata, 'excluded_recordings': len(excluded_recordings),
            'excluded_pcm': len(excluded_pcm), 'excluded_utterance_ids': len(excluded_ids),
            'recording_id_disjoint': True, 'pcm_disjoint': True, 'utterance_id_disjoint': True}


def train(args):
    build.environment()
    build.verify_toolchain(args.toolchain)
    manifest = build.read(args.build)
    if not manifest['instrumented'] or manifest['performance_evaluation_eligible'] or manifest['variant'] != '022c-train':
        raise ValueError('Training requires the separately marked instrumented build.')
    if manifest['toolchain_sha256'] != build.sha(args.toolchain):
        raise ValueError('Training must use the instrumented build toolchain.')
    identity = build.source_identity(Path(manifest['source_path']))
    if identity['source_commit'] != manifest['source_commit']:
        raise ValueError('The instrumented source worktree changed before training.')
    training, assets, short = build.read(args.training), build.read(args.assets), build.read(args.short)
    audit = validate_split(training, assets, short)
    for record in training['records']:
        path = Path(record['pcm_path'])
        if path.stat().st_size != record['samples'] * 2 or build.sha(path) != record['pcm_sha256']:
            raise ValueError('Training PCM bytes differ from the frozen subset manifest.')
    models = [m for m in assets['models'] if m['type'] == args.type]
    if len(models) != 1:
        raise ValueError('The selected model type must occur once in the evaluation assets.')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    profiles = out / 'profiles'; profiles.mkdir()
    worker_path = Path(args.worker).resolve()
    worker_artifact = build.artifact(worker_path)
    module_spec = importlib.util.spec_from_file_location('worker022', worker_path)
    worker = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(worker)
    worker.execution_environment({})
    pattern = str(profiles / '%m-%p.profraw')
    spec = {'directory': str(out / 'worker'), 'build': manifest, 'model': models[0], 'threads': 4,
            'phase': 'pgo_train', 'round': 0, 'side': 'training_only', 'environment': {},
            'warmup': training['records'], 'records': training['records']}
    build.write(out / 'spec.json', spec)
    build.write(out / 'inputs.json', {'training': training, 'assets_sha256': build.sha(args.assets),
                                    'short_sha256': build.sha(args.short), 'audit': audit})
    os.environ['LLVM_PROFILE_FILE'] = pattern
    try:
        worker.run(spec)
    finally:
        del os.environ['LLVM_PROFILE_FILE']
    raw = [build.artifact(p) for p in sorted(profiles.glob('*.profraw'))]
    if not raw or any(Path(p['path']).stat().st_size == 0 for p in raw):
        raise ValueError('The instrumented worker did not write nonempty profiles on normal exit.')
    if build.artifact(worker_path) != worker_artifact or build.source_identity(Path(manifest['source_path'])) != identity:
        raise ValueError('The worker or source changed during training.')
    build.write(out / 'train.json', {'purpose': 'PGO training only; timings are not performance evidence',
                                   'performance_evaluation_eligible': False, 'instrumented_build': manifest,
                                   'instrumented_build_manifest': build.artifact(args.build), 'type': args.type,
                                   'model': models[0], 'training_manifest': build.artifact(args.training),
                                   'evaluation_assets': build.artifact(args.assets), 'short_inputs': build.artifact(args.short),
                                   'toolchain_sha256': build.sha(args.toolchain), 'lto': manifest['lto'],
                                   'worker': worker_artifact, 'python_executable': sys.executable,
                                   'python_version': sys.version, 'driver_sha256': build.sha(__file__),
                                   'profile_environment': {'LLVM_PROFILE_FILE': pattern}, 'raw_profiles': raw, 'audit': audit,
                                   'worker_result': build.artifact(out / 'worker' / 'result.json')})
    print(out / 'train.json')


def merge(args):
    env = build.environment()
    pinned = build.verify_toolchain(args.toolchain)
    runs = [build.read(path) for path in args.run]
    if len(runs) != 3 or {run['type'] for run in runs} != {'q4_k', 'q8_0', 'f16'}:
        raise ValueError('Merge one training run for each of Q4_K, Q8_0 and F16.')
    first = runs[0]
    keys = ('instrumented_build_manifest', 'training_manifest', 'evaluation_assets', 'short_inputs', 'toolchain_sha256',
            'lto', 'worker', 'python_executable', 'python_version', 'driver_sha256', 'audit')
    if any(any(run[key] != first[key] for key in keys) for run in runs):
        raise ValueError('The training runs differ in source, subset, tools, runtime or conditions.')
    if first['toolchain_sha256'] != build.sha(args.toolchain):
        raise ValueError('Merge requires the training toolchain.')
    raw = [item for run in runs for item in run['raw_profiles']]
    if len({item['path'] for item in raw}) != len(raw) or any(build.sha(item['path']) != item['sha256'] for item in raw):
        raise ValueError('Raw training profiles changed or were included more than once.')
    out = Path(args.out).resolve(); out.mkdir(parents=True, exist_ok=False)
    profile = out / 'code.profdata'
    cmd = [pinned['tools']['profdata']['path'], 'merge', '-o', str(profile), *[item['path'] for item in raw]]
    with (out / 'merge.log').open('x') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
    build.write(out / 'profile.json', {'instrumented_build': first['instrumented_build'], 'lto': first['lto'],
                                     'toolchain_sha256': first['toolchain_sha256'], 'profile': build.artifact(profile),
                                     'training_runs': [build.artifact(path) for path in args.run],
                                     'training_manifest': first['training_manifest'], 'evaluation_assets': first['evaluation_assets'],
                                     'short_inputs': first['short_inputs'], 'audit': first['audit'], 'command': cmd,
                                     'merge_log_sha256': build.sha(out / 'merge.log')})
    print(out / 'profile.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--locked', action='store_true', help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('audit', 'train'):
        entry = sub.add_parser(name)
        for option in ('training', 'assets', 'short'):
            entry.add_argument('--' + option, required=True)
        if name == 'train':
            for option in ('build', 'toolchain', 'worker', 'out'):
                entry.add_argument('--' + option, required=True)
            entry.add_argument('--type', required=True, choices=('q4_k', 'q8_0', 'f16'))
    entry = sub.add_parser('merge')
    entry.add_argument('--run', action='append', required=True)
    entry.add_argument('--toolchain', required=True); entry.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command != 'audit' and not args.locked:
        subprocess.run(['/usr/bin/lockf', '-k', build.LOCK, sys.executable, '-B', __file__, '--locked', *sys.argv[1:]], check=True)
    elif args.command == 'audit':
        print(validate_split(build.read(args.training), build.read(args.assets), build.read(args.short)))
    elif args.command == 'train':
        train(args)
    else:
        merge(args)


if __name__ == '__main__':
    main()
