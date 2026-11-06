"""Check repeat identity, separate aggregation and changed latency text without model or GPU execution."""
import importlib.util
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

import compare

spec = importlib.util.spec_from_file_location('latency_repeat', Path(__file__).with_name('latency-repeat.py'))
repeat = importlib.util.module_from_spec(spec); spec.loader.exec_module(repeat)


def check(root):
    root.mkdir()
    binary = root / 'fixture-binary'; binary.write_bytes(b'not executable')
    build = {'source_verified': True, 'speech': {'path': str(binary), 'sha256': compare.sha(binary)}}
    model = {'type': 'q4_k', 'path': str(binary), 'sha256': compare.sha(binary)}
    rows = [{'file': f'{s}-{i}.wav', 'pcm_path': str(binary), 'pcm_sha256': compare.sha(binary),
             'seconds': seconds, 'stratum': s, 'reference': 'reference'}
            for s, seconds in [('6-10s', 6), ('10-20s', 12), ('20-30s', 24)] for i in range(10)]
    assets = root / 'assets.json'; compare.write(assets, {'subset': rows, 'models': [model], 'input_hash': 'fixture'})
    primary = root / 'primary'; primary.mkdir()
    manifest = {'scope': 'full', 'measure_id': 'fixture', 'types': ['q4_k'], 'assets_sha256': compare.sha(assets),
                'input_hash': 'fixture', 'builds': {side: build for side in ['baseline', 'candidate', 'anchor']},
                'execution_environments': {side: {} for side in ['baseline', 'candidate', 'anchor']},
                'models': {'q4_k': {'sha256': model['sha256']}},
                'conditions': {'python': platform.python_version(), 'python_executable': sys.executable, 'rounds': 3,
                    'runner_sha256': {name: compare.sha(repeat.HERE / name) for name in ['compare.py', 'worker.py']},
                    'latency_count': 30, 'quality_count': 650, 'decode_on_cpu': True, 'threads': 4,
                    'latency_definition': 'Fixture timer.', 'aggregation': 'Median per input, then percentiles.'}}

    def save_primary():
        compare.write(primary / 'run.json', manifest)
        compare.write(primary / 'summary.json', {'scope': 'full', 'measure_id': 'fixture',
                      'evidence': {'run_manifest_sha256': compare.sha(primary / 'run.json')}})

    save_primary()
    for rnd, order in enumerate([['baseline', 'candidate', 'anchor'], ['candidate', 'anchor', 'baseline'], ['anchor', 'baseline', 'candidate']]):
        for index, side in enumerate(order):
            directory = primary / 'q4_k' / side / f'latency-{rnd}'; directory.mkdir(parents=True)
            compare.write(directory / 'result.json', {'type': 'q4_k', 'round': rnd, 'side': side, 'started_utc': f'2026-01-01T00:00:{rnd * 3 + index:02}Z'})
    args = SimpleNamespace(primary_run=primary, assets=assets, runner=repeat.HERE, out=root / 'exact')

    def refused(call, reason):
        try: call()
        except ValueError as error: assert reason in str(error), (reason, str(error))
        else: raise AssertionError('Invalid repeat evidence was accepted.')

    original = manifest['conditions']['runner_sha256']['worker.py']
    manifest['conditions']['runner_sha256']['worker.py'] = 'changed'; save_primary()
    refused(lambda: repeat.primary(args), 'identical primary frozen runner source: worker.py')
    manifest['conditions']['runner_sha256']['worker.py'] = original; save_primary()
    with patch.object(repeat, 'source_identity', return_value={'commit': 'fixture'}), patch.object(repeat, 'run_worker') as launch:
        for side in ['baseline', 'candidate']:
            manifest['builds'][side] = {**build, 'performance_evaluation_eligible': False}; save_primary()
            refused(lambda: repeat.run(args), 'performance_evaluation_eligible=false')
            launch.assert_not_called()
            assert not Path(args.out).exists()
            manifest['builds'][side] = build; save_primary()
    with patch.object(repeat.platform, 'python_version', return_value='different'):
        refused(lambda: repeat.primary(args), 'recorded Python')
    with patch.dict(compare.os.environ, {'GGML_METAL_FUSION_DISABLE': '0'}):
        refused(lambda: repeat.primary(args), 'inherited')
    binary.write_bytes(b'changed')
    refused(lambda: repeat.primary(args), 'verified primary build')
    binary.write_bytes(b'not executable')

    launches, mode = [], 'exact'

    def worker(directory, spec_path):
        spec = compare.read(spec_path); launches.append((spec['round'], spec['side']))
        assert spec['phase'] == 'latency' and spec['threads'] == 4 and len(spec['records']) == len(spec['warmup']) == 30
        factor = .8 if spec['side'] == 'candidate' else 1
        records = [{'file': r['file'], 'stratum': r['stratum'], 'audio_seconds': r['seconds'], 'reference': r['reference'],
                    'seconds': factor * (r['seconds'] * .01 + spec['round'] * .001), 'text': 'same', 'tokens': None, 'segments': None} for r in spec['records']]
        if mode == 'changed' and spec['side'] == 'candidate' and spec['round'] == 1: records[0]['text'] = 'changed'
        if mode == 'duplicate': records[-1] = records[0]
        target = Path(spec['directory']); target.mkdir(parents=True)
        compare.write(target / 'result.json', {'type': 'q4_k', 'side': spec['side'], 'round': spec['round'], 'phase': 'latency',
                      'warmup_count': 30, 'build': spec['build'], 'environment': spec['environment'],
                      'command': ['fixture', '--threads', '4', '--decode-on-cpu'],
                      'worker_source_sha256': compare.sha(directory / 'runner-snapshot/worker.py'), 'records': records,
                      'memory': {'peak_memory_footprint': 100}, 'load_seconds': .1, 'first_request': {'seconds': .2}})

    primary_hashes = {p: compare.sha(primary / p) for p in ['run.json', 'summary.json']}
    with patch.object(repeat, 'source_identity', return_value={'commit': 'fixture'}), patch.object(repeat, 'run_worker', worker), patch.object(compare, 'score', side_effect=AssertionError('A repeat must not score CER.')):
        repeat.run(args)
        assert launches == [(0, 'baseline'), (0, 'candidate'), (1, 'candidate'), (1, 'baseline'), (2, 'baseline'), (2, 'candidate')]
        summary = compare.read(args.out / 'summary.json'); result = summary['types']['q4_k']
        assert summary['status'] == 'exact_text_matched' and summary['pooled_with_primary'] is False
        assert summary['conditions']['quality_count'] == 0 and not summary['conditions']['fixed_anchor_measured']
        assert len(summary['primary_latency_block_order']) == 9 and result['comparison']['latency_text_comparisons'] == 90
        assert abs(result['comparison']['latency_reduction_percent'] - 20) < 1e-12
        assert all(s['literal_cer'] is None and s['repository_cer'] is None for s in result['sides'].values())
        assert all(v['count'] == 10 for v in result['comparison']['strata'].values())
        refused(lambda: repeat.summarize(args.out), 'Do not overwrite')
        mode = 'changed'; args.out = root / mode; repeat.run(args)
        summary = compare.read(args.out / 'summary.json')
        assert summary['status'] == 'text_changed' and summary['types']['q4_k']['comparison']['text_changed'] == 1
        assert summary['types']['q4_k']['sides']['candidate']['repeated_text_changes'] == 1
        mode = 'duplicate'; args.out = root / mode
        refused(lambda: repeat.run(args), 'same thirty unique recordings')
    assert {p: compare.sha(primary / p) for p in primary_hashes} == primary_hashes
