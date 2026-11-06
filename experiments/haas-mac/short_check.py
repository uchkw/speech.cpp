"""Check short-prefix preparation and paired evidence with synthetic workers, without a model or GPU."""
from pathlib import Path
import platform
import sys
from types import SimpleNamespace
from unittest.mock import patch

import compare
import short


def check(root):
    root.mkdir()
    binary = root / 'synthetic-binary'; binary.write_bytes(b'fixture, not executable')
    build = {'source_verified': True, 'speech': {'path': str(binary), 'sha256': compare.sha(binary)}}
    model = {'type': 'q4_k', 'path': str(binary), 'sha256': compare.sha(binary)}
    sources = []
    for index, seconds in enumerate([6, 12, 24]):
        path = root / f'{index}.pcm'; path.write_bytes(bytes(range(256)) * (seconds * 125))
        sources.append({'file': f'{index}.wav', 'pcm_path': str(path), 'pcm_sha256': compare.sha(path),
                        'seconds': seconds, 'stratum': str(index), 'reference': 'A full-recording label must not become a prefix label.'})
    asset_path = root / 'assets.json'
    compare.write(asset_path, {'subset': sources, 'stages': sources, 'models': [model], 'input_hash': 'source-input-fixture'})
    input_dir = root / 'prefixes'; short.prepare(SimpleNamespace(assets=asset_path, out=input_dir))
    data = short.inputs(input_dir / 'inputs.json')
    assert len(data['records']) == 6 and data['literal_cer'] is None and data['repository_cer'] is None
    source_by_file = {r['file']: r for r in sources}
    for row in data['records']:
        original = source_by_file[row['source_file']]
        assert Path(row['pcm_path']).read_bytes() == Path(original['pcm_path']).read_bytes()[:row['sample_count'] * 2]
        assert row['reference'] is None and row['source_pcm_sha256'] == original['pcm_sha256']
    primary = root / 'primary'; primary.mkdir()
    manifest = {'scope': 'full', 'measure_id': '003', 'types': ['q4_k'], 'assets_sha256': compare.sha(asset_path),
                'input_hash': 'source-input-fixture', 'builds': {'baseline': build, 'candidate': build},
                'execution_environments': {'baseline': {'SPEECH_HAAS_TDT_RESIDENT_STATE': '0'}, 'candidate': {'SPEECH_HAAS_TDT_RESIDENT_STATE': '1'}},
                'models': {'q4_k': {'sha256': model['sha256']}},
                'conditions': {'python': platform.python_version(), 'python_executable': sys.executable, 'rounds': 3,
                               'runner_sha256': {'worker.py': compare.sha(short.HERE / 'worker.py')},
                               'latency_count': 30, 'quality_count': 650, 'decode_on_cpu': True, 'threads': 4,
                               'latency_definition': 'Fixture timer.', 'aggregation': 'Median per input, then percentiles.'}}

    def save_primary():
        compare.write(primary / 'run.json', manifest)
        compare.write(primary / 'summary.json', {'scope': 'full', 'measure_id': '003',
                                               'evidence': {'run_manifest_sha256': compare.sha(primary / 'run.json')}})

    save_primary()

    def refused(call, message):
        try:
            call()
        except ValueError as error:
            assert message in str(error), (message, str(error))
        else:
            raise AssertionError('Invalid short-prefix evidence was accepted.')

    with patch.object(short.platform, 'python_version', return_value='different-version'):
        refused(lambda: short.primary(primary, data), 'primary run\'s Python')
    manifest['conditions']['python_executable'] = '/different/interpreter'; save_primary()
    refused(lambda: short.primary(primary, data), 'recorded interpreter')
    del manifest['conditions']['python_executable']; save_primary()
    refused(lambda: short.primary(primary, data), 'do not backfill')
    manifest['conditions']['python_executable'] = sys.executable; save_primary()
    timer = manifest['conditions']['runner_sha256']['worker.py']
    manifest['conditions']['runner_sha256']['worker.py'] = 'different-timer'; save_primary()
    refused(lambda: short.primary(primary, data), 'same worker timer source')
    manifest['conditions']['runner_sha256']['worker.py'] = timer; save_primary()
    changed = compare.read(input_dir / 'inputs.json'); changed['records'][0]['reference'] = 'full recording'
    compare.write(root / 'labeled.json', changed)
    refused(lambda: short.inputs(root / 'labeled.json'), 'must remain unlabeled')
    altered = compare.read(input_dir / 'inputs.json'); altered['records'][0]['source_pcm_sha256'] = 'wrong'
    compare.write(root / 'changed-identity.json', altered)
    refused(lambda: short.inputs(root / 'changed-identity.json'), 'input identity changed')

    mode = 'exact'

    def synthetic_worker(directory, spec_path):
        spec = compare.read(spec_path); target = Path(spec['directory']); target.mkdir(parents=True)
        assert spec['threads'] == 4 and len(spec['warmup']) == 6
        factor = .8 if spec['side'] == 'candidate' else 1.
        rows = [{'file': r['file'], 'stratum': r['stratum'], 'audio_seconds': r['seconds'],
                 'seconds': factor * (.01 * r['seconds'] + .001 * spec['round']),
                 'reference': None, 'text': 'same', 'tokens': None, 'segments': None} for r in spec['records']]
        if mode == 'changed' and spec['side'] == 'candidate' and spec['round'] == 1:
            rows[0]['text'] = 'different'
        if mode == 'duplicate':
            rows[-1] = rows[0]
        compare.write(target / 'result.json', {'round': spec['round'], 'side': spec['side'], 'type': 'q4_k', 'phase': 'latency',
            'warmup_count': 6, 'build': spec['build'], 'environment': spec['environment'],
            'worker_source_sha256': compare.sha(directory / 'runner-snapshot/worker.py'), 'records': rows,
            'memory': {'peak_memory_footprint': 100}, 'load_seconds': .1, 'first_request': {'seconds': .2}})

    def run(name):
        destination = root / name
        short.run(SimpleNamespace(primary_run=primary, inputs=input_dir / 'inputs.json', out=destination))
        return destination

    with patch.object(short, 'run_worker', synthetic_worker), patch.object(compare, 'score', side_effect=AssertionError('CER must not run for prefixes.')):
        destination = run('exact')
        result = compare.read(destination / 'summary.json'); profile = result['types']['q4_k']
        assert result['status'] == 'exact_text_matched' and not result['adoption_evidence_alone']
        assert profile['comparison']['count'] == 18 and profile['comparison']['text_changed'] == 0
        assert abs(profile['comparison']['median_latency_ratio'] - .8) < 1e-12
        assert all(side['literal_cer'] is None and side['repository_cer'] is None for side in profile['sides'].values())
        assert profile['sides']['baseline']['strata']['2s']['count'] == 3
        assert profile['sides']['baseline']['strata']['4s']['count'] == 3
        refused(lambda: short.summarize(destination), 'Do not overwrite')
        mode = 'changed'; destination = run('changed')
        result = compare.read(destination / 'summary.json')
        assert result['status'] == 'text_changed' and result['types']['q4_k']['comparison']['text_changed'] == 1
        assert result['types']['q4_k']['sides']['candidate']['repeated_text_changes'] == 1
        mode = 'duplicate'
        refused(lambda: run('duplicate'), 'same six unique prefixes')

