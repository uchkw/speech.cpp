"""Check campaign arithmetic, pairing, acceptance gates and standalone reports without a GPU.

Usage: python3 experiments/haas-mac/check.py [--previous PREVIOUS_EXPERIMENT_DIRECTORY]
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
import compare
import render
import worker


def refused(operation, message, expected=None):
    try:
        operation()
    except ValueError as error:
        if expected is not None:
            assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(message)


def environment_launch_checks():
    values = {'GGML_METAL_FUSION_DEBUG': '2', 'GGML_METAL_GRAPH_DEBUG': '1',
              'GGML_METAL_FUSION_DISABLE': '0', 'GGML_METAL_N_CB': '0',
              'GGML_METAL_GRAPH_OPTIMIZE_DISABLE': '0', 'GGML_NATIVE': 'OFF',
              'GGML_CPU_NORM_MUL_ADD_TRACE': '0',
              'SPEECH_UNRECORDED_EXPERIMENT': '0'}
    for key, value in values.items():
        refused(lambda: compare.environment([f'{key}={value}']),
                'An unsupported recorded switch was accepted.', expected='Only explicitly allowed')
        with patch.dict(compare.os.environ, {key: value}):
            refused(lambda: compare.environment([]), 'An unrecorded backend switch was accepted.', expected='inherited')
            with patch.object(worker.subprocess, 'Popen') as start:
                refused(lambda: worker.run({'environment': {}}), 'Worker launch inherited an unrecorded switch.', expected='inherited')
                start.assert_not_called()
            with patch.object(compare, 'git') as git:
                refused(lambda: compare.build(SimpleNamespace()), 'Build setup inherited an unrecorded switch.', expected='inherited')
                git.assert_not_called()


def diagnostic_environment_checks():
    for key in ['GGML_CPU_NORM_MUL_ADD_FUSION_TRACE', 'SPEECH_HAAS_WORKER_TRANSPORT_TRACE']:
        assert compare.environment([f'{key}=0'])[key] == '0'
        refused(lambda: compare.environment([f'{key}=1']), 'A trace entered primary timing.', expected='diagnostic phase')
        refused(lambda: worker.execution_environment({key: '2'}, diagnostic=True),
                'An invalid trace value was accepted.', expected='0 or 1')
        for value in ['0', '1']:
            assert worker.execution_environment({key: value}, diagnostic=True)[key] == value
            with patch.dict(worker.os.environ, {key: value}):
                refused(lambda: worker.execution_environment({}, diagnostic=True),
                        'A diagnostic inherited an unrecorded trace.', expected='inherited')
                assert worker.execution_environment({key: '0'})[key] == '0'
                assert worker.execution_environment({key: '1'}, diagnostic=True)[key] == '1'
        for phase in ['latency', 'quality', 'pgo_train']:
            with patch.object(worker.subprocess, 'Popen') as launch:
                refused(lambda: worker.run({'phase': phase, 'environment': {key: '1'}}),
                        'A trace reached a nondiagnostic worker.', expected='diagnostic phase')
                launch.assert_not_called()


def performance_build_checks(root):
    root.mkdir()
    binary = root / 'binary'; binary.write_bytes(b'fixture')
    identity = {'path': str(binary), 'sha256': compare.sha(binary)}
    build = {'source_verified': True, 'source_commit': 'fixture', 'speech': identity,
             'stage_tool': identity, 'stage_threads_argument': True}
    rows = [{'file': f'{group}-{i}', 'seconds': seconds, 'stratum': group}
            for group, seconds in [('6-10s', 6), ('10-20s', 12), ('20-30s', 24)] for i in range(10)]
    assets = root / 'assets.json'
    compare.write(assets, {'models': [{**identity, 'type': 'q4_k'}], 'stages': [], 'subset': rows,
                          'input_hash': 'fixture', 'source_environment': {'hardware': {}}})
    paths = {side: root / f'{side}.json' for side in ['baseline', 'candidate', 'anchor']}
    for side in paths:
        for name, path in paths.items():
            compare.write(path, {**build, **({'performance_evaluation_eligible': False} if name == side else {})})
        args = SimpleNamespace(out=root / f'refused-{side}', assets=assets, baseline=paths['baseline'],
                               candidate=paths['candidate'], anchor=paths['anchor'], baseline_env=[],
                               candidate_env=[], anchor_env=[], smoke=False, screen=True, types='q4_k',
                               quality_policy='exact', measure='fixture')
        with patch.object(compare, 'locked', side_effect=AssertionError('An ineligible build reached a worker launch.')) as launch, \
                patch.object(compare.subprocess, 'check_output', return_value='fixture'):
            refused(lambda: compare.run_comparison(args), 'An instrumented build was evaluated.', expected='performance_evaluation_eligible=false')
            launch.assert_not_called()
            assert not Path(args.out).exists()


def decoder_device_run_checks(root):
    root.mkdir()
    binary = root / 'binary'; binary.write_bytes(b'fixture')
    anchor_binary = root / 'anchor-binary'; anchor_binary.write_bytes(b'anchor-fixture')

    def build_manifest(path):
        return {'source_verified': True, 'source_commit': 'fixture', 'stage_threads_argument': True,
                'speech': {'path': str(path), 'sha256': compare.sha(path)},
                'stage_tool': {'path': str(path), 'sha256': compare.sha(path)}}

    paths = {side: root / f'{side}.json' for side in ['baseline', 'candidate']}
    compare.write(paths['baseline'], build_manifest(binary)); compare.write(paths['candidate'], build_manifest(binary))
    anchor = root / 'anchor.json'; compare.write(anchor, build_manifest(anchor_binary))
    rows = [{'file': f'{group}-{i}', 'seconds': seconds + i / 10, 'stratum': group}
            for group, seconds in [('6-10s', 6), ('10-20s', 12), ('20-30s', 24)] for i in range(10)]
    stage_directory = root / 'stage-audio' / 'fixture-input'; stage_directory.mkdir(parents=True)
    (stage_directory / 'audio.npy').write_bytes(b'fixture-npy')
    assets = root / 'assets.json'
    compare.write(assets, {'models': [{'type': 'q4_k', 'path': str(binary), 'sha256': compare.sha(binary)}],
                           'stages': [{'file': 'fixture-input.wav', 'directory': str(stage_directory), 'seconds': 6.,
                                       'npy_sha256': compare.sha(stage_directory / 'audio.npy')}],
                           'subset': rows, 'input_hash': 'fixture', 'source_environment': {'hardware': {}}})
    specs, stage_commands = [], []

    def locked_worker(arguments):
        spec = compare.read(arguments[2])
        specs.append(spec)
        target = Path(spec['directory']); target.mkdir(parents=True)
        records = [{'file': r['file'], 'stratum': r['stratum'], 'audio_seconds': r['seconds'], 'seconds': .1,
                    'reference': 'a', 'text': 'a', 'tokens': [], 'segments': []} for r in spec['records']]
        ready = {'model': {'threads': spec['threads']}}
        if spec['decoder_device'] == 'metal':
            ready = {'model': {'threads': spec['threads'], 'device': 'MTL0'}, 'defaults': {'decode_on_cpu': False}}
        compare.write(target / 'result.json', {'round': spec['round'], 'side': spec['side'], 'records': records,
                                               'decoder_device': spec['decoder_device'], 'decode_on_cpu': spec['decode_on_cpu'],
                                               'ready': ready, 'memory': {'peak_memory_footprint': 100}, 'load_seconds': .1,
                                               'first_request': {'seconds': .2}})

    def stage_tool(command, **options):
        stage_commands.append(list(command))
        for directory in command[10:]:
            options['stdout'].write(json.dumps({'input': Path(directory).name, 'device': command[5], 'decoding': command[6],
                                                'decode_on': command[7], 'threads': int(command[9]), 'run': 0,
                                                'audio_seconds': 6., 'core_total_ms': 50., 'decoder_ms': 5.}) + '\n')

    def comparison(name, decoder=None, anchor_path=None):
        arguments = SimpleNamespace(out=root / name, assets=assets, baseline=paths['baseline'], candidate=paths['candidate'],
                                    anchor=anchor_path, baseline_env=[], candidate_env=[], anchor_env=[], smoke=False,
                                    screen=True, types='q4_k', quality_policy='exact', measure='fixture')
        if decoder is not None:
            arguments.decoder_device = decoder
        specs.clear(); stage_commands.clear()
        with patch.object(compare, 'locked', locked_worker), \
                patch.object(compare.subprocess, 'check_output', return_value='fixture'), \
                patch.object(compare.subprocess, 'run', stage_tool):
            compare.run_comparison(arguments)
        return root / name

    cpu_expectations = {'decoder_device': 'cpu', 'decode_on_cpu': True, 'decoding': 'greedy', 'decode_on': 'cpu', 'threads': 4}
    default_run = comparison('cpu-default')
    cpu_commands = [list(command) for command in stage_commands]
    cpu_specs = [dict(spec) for spec in specs]
    explicit_run = comparison('cpu-explicit', 'cpu')
    assert [list(command) for command in stage_commands] == cpu_commands
    assert [{k: v for k, v in spec.items() if k != 'directory'} for spec in specs] == \
           [{k: v for k, v in spec.items() if k != 'directory'} for spec in cpu_specs]
    for run_directory in [default_run, explicit_run]:
        manifest = compare.read(run_directory / 'run.json')
        assert manifest['conditions']['decoder_device'] == 'cpu' and manifest['conditions']['decode_on_cpu'] is True
        assert manifest['conditions']['stage_expectations'] == cpu_expectations
        assert compare.read(run_directory / 'summary.json')['conditions'] == manifest['conditions']
    for spec in cpu_specs:
        assert spec['decoder_device'] == 'cpu' and spec['decode_on_cpu'] is True
    for command in cpu_commands:
        assert command[5:10] == ['gpu', 'greedy', 'cpu', '3', '4'] and command[10:] == [str(stage_directory)]
    assert compare.read(compare.HERE / 'campaign.json')['decode_on_cpu'] is True

    metal_run = comparison('metal', 'metal', anchor)
    manifest = compare.read(metal_run / 'run.json'); conditions = manifest['conditions']
    assert conditions['decoder_device'] == 'metal' and conditions['decode_on_cpu'] is False
    assert conditions['stage_expectations'] == {'decoder_device': 'metal', 'decode_on_cpu': False, 'decoding': 'greedy',
                                                'decode_on': 'device', 'threads': 4, 'device': 'MTL0'}
    assert set(manifest['builds']) == {'baseline', 'candidate', 'anchor'}
    assert manifest['fixed_anchor']['measured_side'] == 'anchor'
    assert {spec['side'] for spec in specs} == {'baseline', 'candidate', 'anchor'}
    for spec in specs:
        assert spec['decoder_device'] == 'metal' and spec['decode_on_cpu'] is False
    for command in stage_commands:
        assert command[5:10] == ['MTL0', 'greedy', 'device', '3', '4']
    summary = compare.read(metal_run / 'summary.json')
    assert summary['conditions'] == conditions
    assert all(side['stage_status'] == 'measured' for side in summary['types']['q4_k']['sides'].values())

    deduplicated = comparison('metal-dedup', 'metal', paths['baseline'])
    manifest = compare.read(deduplicated / 'run.json')
    assert set(manifest['builds']) == {'baseline', 'candidate'} and manifest['fixed_anchor']['measured_side'] == 'baseline'
    assert all(spec['decoder_device'] == 'metal' for spec in specs)

    invalid = SimpleNamespace(out=root / 'refused-selector', assets=assets, baseline=paths['baseline'],
                              candidate=paths['candidate'], anchor=None, baseline_env=[], candidate_env=[],
                              anchor_env=[], smoke=False, screen=True, types='q4_k', quality_policy='exact',
                              measure='fixture', decoder_device='cuda')
    with patch.object(compare, 'locked') as launch, patch.object(compare.subprocess, 'check_output', return_value='fixture'):
        refused(lambda: compare.run_comparison(invalid), 'An invalid decoder selector reached a launch.',
                expected='decoder_device must be')
        launch.assert_not_called()
        assert not (root / 'refused-selector').exists()
    with patch.object(sys, 'argv', ['compare.py', 'run', '--measure', 'fixture', '--baseline', str(paths['baseline']),
                                    '--candidate', str(paths['candidate']), '--assets', str(assets), '--types', 'q4_k',
                                    '--out', str(root / 'refused-cli'), '--decoder-device', 'cuda']):
        try:
            compare.main()
        except SystemExit as failure:
            assert failure.code == 2
        else:
            raise AssertionError('The command line accepted an invalid decoder selector.')
    assert not (root / 'refused-cli').exists()

    valid_metal_stage = {'input': 'fixture-input', 'device': 'MTL0', 'decoding': 'greedy', 'decode_on': 'device',
                         'threads': 4, 'run': 0, 'audio_seconds': 6., 'core_total_ms': 50., 'decoder_ms': 5.}
    for side in ['baseline', 'candidate', 'anchor']:
        path = metal_run / 'q4_k' / side / 'stages.jsonl'
        for key, value in [('decode_on', 'cpu'), ('device', 'Metal'), ('decoding', 'beam'), ('threads', 8)]:
            path.write_text(json.dumps({**valid_metal_stage, key: value}) + '\n')
            refused(lambda: compare.summarize(metal_run), f'A mislabeled {side} metal stage row was summarized.',
                    expected=f'{key}={value!r}')
        path.write_text(json.dumps(valid_metal_stage) + '\n')
    for side, phase, label in [('anchor', 'latency-0', 'latency round'), ('candidate', 'quality-0', 'quality')]:
        path = metal_run / 'q4_k' / side / phase / 'result.json'
        original = compare.read(path)
        compare.write(path, {**original, 'decoder_device': 'cpu', 'decode_on_cpu': True, 'ready': {'model': {'threads': 4}}})
        refused(lambda: compare.summarize(metal_run), f'A CPU {label} result was summarized under a metal manifest.',
                expected="records decoder_device='cpu'")
        compare.write(path, original)
    valid_cpu_stage = {**valid_metal_stage, 'device': 'gpu', 'decode_on': 'cpu'}
    cpu_stage = default_run / 'q4_k' / 'baseline' / 'stages.jsonl'
    for key, value in [('decode_on', 'device'), ('decoding', 'beam'), ('threads', 8)]:
        cpu_stage.write_text(json.dumps({**valid_cpu_stage, key: value}) + '\n')
        refused(lambda: compare.summarize(default_run), 'A mislabeled CPU stage row was summarized.',
                expected=f'{key}={value!r}')
    cpu_stage.write_text(json.dumps(valid_cpu_stage) + '\n')
    compare.summarize(default_run)
    manifest = compare.read(metal_run / 'run.json')
    manifest['conditions']['stage_expectations']['decode_on'] = 'cpu'
    compare.write(metal_run / 'run.json', manifest)
    refused(lambda: compare.summarize(metal_run), 'Stage expectations contradicting the decoder were summarized.',
            expected='disagree')
    manifest['conditions']['stage_expectations']['decode_on'] = 'device'
    compare.write(metal_run / 'run.json', manifest)
    compare.summarize(metal_run)

    campaign = compare.read(compare.HERE / 'campaign.json')
    page = render.report_html({**compare.read(metal_run / 'summary.json'), 'title': 'Fixture', 'reason': 'Fixture',
                               'status': 'conditional'}, 'fixture.json', campaign)
    assert 'Metal decoding on MTL0 with 4 CPU load threads.' in page
    assert 'CPU decoding with' not in page and campaign['thread_setting_note'] not in page
    cpu_summary = compare.read(default_run / 'summary.json')
    page = render.report_html({**cpu_summary, 'title': 'Fixture', 'reason': 'Fixture', 'status': 'conditional'},
                              'fixture.json', campaign)
    assert 'CPU decoding with 4 threads.' in page and campaign['thread_setting_note'] in page
    legacy = {**cpu_summary, 'conditions': {key: value for key, value in cpu_summary['conditions'].items()
                                            if key != 'decoder_device'}}
    page = render.report_html({**legacy, 'title': 'Fixture', 'reason': 'Fixture', 'status': 'conditional'},
                              'fixture.json', campaign)
    assert 'CPU decoding with 4 threads.' in page and 'Metal decoding' not in page


def cpu_decoder_supplement_gates(root):
    root.mkdir()
    import importlib.util
    import short
    module = importlib.util.spec_from_file_location('latency_repeat_gate', compare.HERE / 'latency-repeat.py')
    repeat = importlib.util.module_from_spec(module); module.loader.exec_module(repeat)
    binary = root / 'binary'; binary.write_bytes(b'fixture')
    identity = {'path': str(binary), 'sha256': compare.sha(binary)}
    build = {'source_verified': True, 'speech': identity}
    model = {'type': 'q4_k', **identity}
    assets = root / 'assets.json'
    compare.write(assets, {'models': [model], 'input_hash': 'fixture'})
    short_manifest = {'scope': 'full', 'measure_id': '003', 'types': ['q4_k'], 'assets_sha256': compare.sha(assets),
                      'input_hash': 'fixture', 'builds': {'baseline': build, 'candidate': build},
                      'execution_environments': {'baseline': {}, 'candidate': {}},
                      'models': {'q4_k': {'sha256': identity['sha256']}},
                      'conditions': {'python': platform.python_version(), 'python_executable': sys.executable, 'rounds': 3,
                                     'latency_count': 30, 'quality_count': 650, 'decode_on_cpu': True, 'threads': 4,
                                     'runner_sha256': {'worker.py': compare.sha(compare.HERE / 'worker.py')},
                                     'latency_definition': 'Fixture timer.', 'aggregation': 'Median per input, then percentiles.'}}
    short_primary = root / 'short-primary'; short_primary.mkdir()

    def save_short():
        compare.write(short_primary / 'run.json', short_manifest)
        compare.write(short_primary / 'summary.json', {'scope': 'full', 'measure_id': '003',
                                                       'evidence': {'run_manifest_sha256': compare.sha(short_primary / 'run.json')}})

    data = {'source_assets_sha256': compare.sha(assets), 'source_input_hash': 'fixture', 'models': [model]}
    save_short()
    assert short.primary(short_primary, data)['measure_id'] == '003'
    short_manifest['conditions']['decode_on_cpu'] = False; save_short()
    refused(lambda: short.primary(short_primary, data), 'A metal-decoder primary run entered the short supplement.',
            expected='three-round 30/650 CPU-decoder')
    short_manifest['conditions']['decode_on_cpu'] = True

    rows = [{'file': f'{stratum}-{index}.wav', 'pcm_path': str(binary), 'pcm_sha256': identity['sha256'],
             'seconds': seconds, 'stratum': stratum, 'reference': 'reference'}
            for stratum, seconds in [('6-10s', 6), ('10-20s', 12), ('20-30s', 24)] for index in range(10)]
    compare.write(assets, {'subset': rows, 'models': [model], 'input_hash': 'fixture'})
    repeat_manifest = {'scope': 'full', 'measure_id': 'fixture', 'types': ['q4_k'], 'assets_sha256': compare.sha(assets),
                       'input_hash': 'fixture', 'builds': {side: build for side in ['baseline', 'candidate']},
                       'execution_environments': {side: {} for side in ['baseline', 'candidate']},
                       'models': {'q4_k': {'sha256': identity['sha256']}},
                       'conditions': {'python': platform.python_version(), 'python_executable': sys.executable, 'rounds': 3,
                                      'latency_count': 30, 'quality_count': 650, 'decode_on_cpu': True, 'threads': 4,
                                      'runner_sha256': {name: compare.sha(compare.HERE / name)
                                                        for name in ['compare.py', 'worker.py']},
                                      'latency_definition': 'Fixture timer.', 'aggregation': 'Median per input, then percentiles.'}}
    repeat_primary = root / 'repeat-primary'; repeat_primary.mkdir()

    def save_repeat():
        compare.write(repeat_primary / 'run.json', repeat_manifest)
        compare.write(repeat_primary / 'summary.json', {'scope': 'full', 'measure_id': 'fixture',
                                                        'evidence': {'run_manifest_sha256': compare.sha(repeat_primary / 'run.json')}})

    save_repeat()
    for side in ['baseline', 'candidate']:
        for rnd in range(3):
            directory = repeat_primary / 'q4_k' / side / f'latency-{rnd}'; directory.mkdir(parents=True)
            compare.write(directory / 'result.json', {'type': 'q4_k', 'round': rnd, 'side': side,
                                                      'started_utc': f'2026-01-01T00:00:{rnd:02}Z'})
    arguments = SimpleNamespace(primary_run=repeat_primary, assets=assets, runner=compare.HERE, out=root / 'unused')
    assert len(repeat.primary(arguments)[2]) == 6
    repeat_manifest['conditions']['decode_on_cpu'] = False; save_repeat()
    refused(lambda: repeat.primary(arguments), 'A metal-decoder primary run entered the latency repeat.',
            expected='three-round 30/650 CPU-decoder')
    repeat_manifest['conditions']['decode_on_cpu'] = True


class Links(HTMLParser):
    def __init__(self):
        super().__init__(); self.hrefs = []

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag in ['script', 'img', 'iframe', 'link'] and any(k in values for k in ['src', 'href']):
            raise AssertionError('Standalone reports must not fetch external assets.')
        if tag == 'a' and 'href' in values:
            self.hrefs.append(values['href'])


def review_regressions(root):
    run = root / 'run'; run.mkdir()
    build = {'source_verified': True, 'source_commit': 'fixture', 'speech': {'path': str(root / 'speech'), 'sha256': 'a' * 64},
             'stage_tool': {'path': str(root / 'stage'), 'sha256': 'b' * 64},
             'commands': [['cmake', '-S', '/private/source', '-B', '/private/build', '-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=ON', '-DGGML_METAL=ON'],
                          ['cmake', '--build', '/private/build', '--target', 'speech-cli', 'fastconformer-timing', '-j4']]}
    manifest = {'measure_id': '001', 'scope': 'full', 'types': ['q4_k'], 'builds': {side: build for side in ['baseline', 'candidate']},
                'conditions': {'rounds': 3, 'quality_policy': 'exact'}, 'execution_environments': {'baseline': {}, 'candidate': {}},
                'models': {}, 'assets_sha256': 'assets-fixture', 'input_hash': 'input-fixture'}
    compare.write(run / 'run.json', manifest)
    records = [{'file': f'{i}.wav', 'reference': 'a', 'text': 'a', 'seconds': 1., 'audio_seconds': 8.,
                'stratum': '6-10s', 'tokens': [], 'segments': []} for i in range(2)]
    for side in manifest['builds']:
        for phase, rounds in [('latency', 3), ('quality', 1)]:
            for index in range(rounds):
                directory = run / 'q4_k' / side / f'{phase}-{index}'; directory.mkdir(parents=True)
                rows = [dict(row) for row in records]
                if side == 'candidate' and phase == 'latency' and index == 1:
                    rows[0]['text'] = 'broken'
                compare.write(directory / 'result.json', {'round': index, 'records': rows, 'memory': {'peak_memory_footprint': 100},
                                                         'load_seconds': .1, 'first_request': {'seconds': 2.}})
    compare.summarize(run)
    result = compare.read(run / 'summary.json'); profile = result['types']['q4_k']; count = profile['comparisons']['relative']
    assert count['text_changed'] == 0 and count['latency_text_changed'] == 1
    assert [r['text_changed'] for r in count['latency_rounds']] == [0, 1, 0]
    assert count['latency_text_comparisons'] == 6 and 'cumulative' not in profile['comparisons']
    campaign = compare.read(compare.HERE / 'campaign.json')
    fixture = {**result, 'title': 'Fixture', 'reason': 'Fixture', 'status': 'conditional'}
    page = render.report_html(fixture, '001-fixture.json', campaign)
    assert 'timestamps disabled' in page and 'cumulative gain is not measured' in page
    report_dir = root / 'reports'; report_dir.mkdir()
    compare.write(report_dir / '001-example.json', {'measure_id': '001', 'title': 'Example', 'status': 'pending'})
    checks = root / 'checks.json'
    options = SimpleNamespace(measure='001', title=None, run=str(run), checks=str(checks), verdict='accepted', reason='Fixture', artifact_root=str(root))
    with patch.object(compare, 'REPORTS', report_dir):
        for status in ['refused', 'skipped', 'unavailable', 'failed']:
            compare.write(checks, {'checks': [{'name': 'Required stage', 'status': status, 'required': True}]})
            refused(lambda: compare.record(options), f'{status} was accepted.', expected='every required check passed')
        compare.write(checks, {'checks': [{'name': 'Required stage', 'status': 'passed', 'required': True}]})
        refused(lambda: compare.record(options), 'Latency-only corruption was accepted.', expected='changed latency text')
        before = (report_dir / '001-example.json').read_bytes()
        compare.write(run / 'summary.json', {**result, 'measure_id': '002a'})
        refused(lambda: compare.record(options), 'A different measure was written.', expected='measure_id must match')
        assert (report_dir / '001-example.json').read_bytes() == before
        compare.write(run / 'summary.json', {**result, 'scope': 'screen'})
        refused(lambda: compare.record(options), 'A nine-input screen was accepted.', expected='Acceptance requires a full run')
        path = run / 'q4_k/candidate/latency-1/result.json'; clean = compare.read(path); clean['records'][0]['text'] = 'a'; compare.write(path, clean)
        compare.summarize(run)
        with patch.object(compare.subprocess, 'run'):
            compare.record(options)
        accepted = compare.read(report_dir / '001-example.json')
        assert accepted['status'] == 'accepted' and 'cumulative' not in accepted['types']['q4_k']['comparisons']
        flags = accepted['builds']['baseline']['build_flags']
        assert flags['configure'] == ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=ON', '-DGGML_METAL=ON']
        assert flags['build'] == ['--target', 'speech-cli', 'fastconformer-timing', '-j4']
        assert '/private/source' not in json.dumps(accepted) and '/private/build' not in json.dumps(accepted)
        page = render.report_html(accepted, '001-example.json', campaign)
        assert '-DCMAKE_BUILD_TYPE=Release' in page
        with patch.object(render, 'REPORTS', report_dir):
            render.main()
        assert 'not measured' in (report_dir / 'index.html').read_text()
    manifest['fixed_anchor'] = {'measured_side': 'baseline', 'source_commit': 'fixture', 'speech_sha256': 'a' * 64,
                                'environment': {}, 'build_manifest_sha256': 'explicit-fixture'}
    compare.write(run / 'run.json', manifest); compare.summarize(run)
    explicit = compare.read(run / 'summary.json')
    assert explicit['fixed_anchor']['measured_side'] == 'baseline'
    assert explicit['types']['q4_k']['comparisons']['cumulative'] == explicit['types']['q4_k']['comparisons']['relative']
    assert set(explicit['builds']) == {'baseline', 'candidate'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--previous'); args = parser.parse_args()
    unequal_lengths = [{'reference': 'abcde', 'text': 'abXde'}, {'reference': 'a' * 95, 'text': 'a' * 95}]
    value = compare.score(unequal_lengths)
    assert value['literal_edits'] == 1 and value['literal_characters'] == 100 and value['literal_cer'] == .01
    value = compare.score([{'reference': 'Ａ。 ', 'text': 'A'}])
    assert value['literal_edits'] == 3 and value['repository_edits'] == 0
    a = [{'file': 'one.wav', 'reference': 'a', 'text': 'a', 'seconds': 2., 'tokens': [{'text': 'a', 'start': 0., 'end': 1.}], 'segments': []},
         {'file': 'two.wav', 'reference': 'a', 'text': 'a', 'seconds': 4., 'tokens': [], 'segments': []}]
    b = [dict(r) for r in a]; b[0] = {**b[0], 'seconds': 1., 'tokens': [{'text': 'a', 'start': .08, 'end': 1.}]}
    counts, changed = compare.paired(a, b)
    assert counts['text_changed'] == 0 and counts['public_tokens_changed'] == 1 and len(changed) == 1
    assert counts['median_latency_ratio'] == .75
    refused(lambda: compare.paired(a, [b[0], b[0]]), 'Duplicate files were accepted.')
    refused(lambda: compare.paired(a, b[:1]), 'Missing paired input was accepted.')
    refused(lambda: compare.environment(['HOME=elsewhere']), 'An unrelated environment variable was accepted.')
    environment_launch_checks()
    diagnostic_environment_checks()
    assert compare.environment(['GGML_METAL_MM_TILE_N=64'])['GGML_METAL_MM_TILE_N'] == '64'
    for key in ['SPEECH_HAAS_CPU_REPACK', 'SPEECH_HAAS_FFN_F16', 'SPEECH_HAAS_TDT_RESIDENT_STATE',
                'SPEECH_HAAS_TDT_JOINT_REUSE', 'SPEECH_HAAS_TDT_CPU_PLAN', 'SPEECH_HAAS_ENCODER_QKV',
                'GGML_METAL_MM_SILU', 'GGML_METAL_ADD_SOFTMAX', 'SPEECH_HAAS_TDT_BLANK_BATCH',
                'SPEECH_HAAS_KDA_QKV', 'SPEECH_HAAS_POSITION_CACHE', 'GGML_METAL_GLU',
                'SPEECH_HAAS_ENCODER_GRAPH_CACHE', 'SPEECH_HAAS_ENCODER_GRAPH_CACHE_TRACE',
                'GGML_METAL_DW_SILU', 'SPEECH_HAAS_DEPTHWISE_PREPACK', 'GGML_METAL_RESIDUAL_NORM',
                'GGML_METAL_HEAD_STORE', 'SPEECH_HAAS_TDT_BUFFER_REUSE', 'SPEECH_HAAS_FRONTEND_PARALLEL',
                'SPEECH_HAAS_FRONTEND_VDSP', 'SPEECH_HAAS_AUTO_THREADS', 'SPEECH_HAAS_KDA_CONSTANTS',
                'GGML_CPU_NORM_MUL_ADD_FUSION', 'SPEECH_HAAS_WORKER_UNSYNC', 'SPEECH_HAAS_WORKER_UNTIE',
                'SPEECH_HAAS_TDT_DEVICE_ARGMAX', 'SPEECH_HAAS_REL_FA', 'SPEECH_HAAS_KEYWORD_CACHE',
                'SPEECH_HAAS_RAW_MEL_PREFIX', 'GGML_METAL_BIND_CACHE', 'SPEECH_HAAS_KDA_PREFIX_CACHE',
                'GGML_METAL_MM_GLU', 'SPEECH_HAAS_KEYWORD_PREFIX']:
        for value in ['0', '1']:
            assert compare.environment([f'{key}={value}'])[key] == value
        refused(lambda: compare.environment([f'{key}=2']), 'An invalid binary switch was accepted.', expected='0 or 1')
        for value in ['0', '1']:
            with patch.dict(compare.os.environ, {key: value}):
                refused(lambda: compare.environment([]), 'An inherited switch was not recorded.', expected='inherited')
                assert compare.environment([f'{key}=0'])[key] == '0'
                assert compare.environment([f'{key}=1'])[key] == '1'
                assert worker.execution_environment({key: '0'})[key] == '0'
                assert worker.execution_environment({key: '1'})[key] == '1'
    subset = [{'file': f'{group}-{i}', 'seconds': seconds + i / 10, 'stratum': group}
              for group, seconds in [('6-10s', 6), ('10-20s', 10), ('20-30s', 20)] for i in range(10)]
    selected = compare.screen_inputs(list(reversed(subset)))
    assert len(selected) == 9 and {group: sum(r['stratum'] == group for r in selected) for group in ['6-10s', '10-20s', '20-30s']} == {'6-10s': 3, '10-20s': 3, '20-30s': 3}
    assert [r['file'] for r in selected] == [f'{group}-{i}' for group in ['6-10s', '10-20s', '20-30s'] for i in [0, 4, 9]]
    assert compare.percentile([1., 4., 9.], .9) == 8.
    with tempfile.TemporaryDirectory(prefix='haas-campaign-check-') as temporary:
        performance_build_checks(Path(temporary) / 'performance-builds')
        decoder_device_run_checks(Path(temporary) / 'decoder-device')
        from worker_check import check as check_worker_threads
        check_worker_threads(Path(temporary) / 'worker-threads')
        cpu_decoder_supplement_gates(Path(temporary) / 'cpu-decoder-supplements')
        review_regressions(Path(temporary))
        from short_check import check as check_short_inputs
        check_short_inputs(Path(temporary) / 'short-prefixes')
        from latency_repeat_check import check as check_latency_repeat
        check_latency_repeat(Path(temporary) / 'latency-repeat')
    campaign = compare.read(compare.HERE / 'campaign.json')
    html = render.report_html({'measure_id': '999', 'title': '<script>bad()</script>', 'status': 'pending', 'reason': '<b>not markup</b>'}, '999.json', campaign)
    assert '<script>' not in html and '&lt;script&gt;' in html
    diagnostic = {'title': 'Instrumented diagnostic', 'method': 'Separate three-input measurement.',
                  'headers': ['Interval', 'Milliseconds'], 'rows': [['<untrusted>', 0.25]], 'notes': ['Not the primary full run.']}
    html = render.report_html({'measure_id': '999', 'title': 'Fixture', 'status': 'conditional', 'reason': 'Fixture',
                               'diagnostics': [diagnostic]}, '999.json', campaign)
    assert 'Instrumented diagnostic' in html and '&lt;untrusted&gt;' in html and '<untrusted>' not in html
    for path in compare.REPORTS.glob('*.html'):
        page = path.read_text(); links = Links(); links.feed(page)
        assert '<html lang="en">' in page and '<meta name="viewport"' in page
        for href in links.hrefs:
            if not href.startswith(('https://', '#')):
                assert (path.parent / href.split('#')[0]).is_file(), (path, href)
        assert '/Users/' not in page
    entries = [compare.read(p) for p in compare.REPORTS.glob('*.json')]
    ids = [e['measure_id'] for e in entries if e.get('kind') != 'preliminary' and not e.get('parent_measure_id')]
    assert sorted(ids) == [f'{n:03}' for n in range(1, 35)]
    if args.previous:
        previous = Path(args.previous)
        expected = compare.read(previous / 'result.json')
        for item in expected['results']:
            records = compare.read(previous / 'runs' / f"quality-{item['type']}-cpu-t4-r0.json")['records']
            actual = compare.score(records)
            assert actual['literal_edits'] == item['quality']['literal_edits']
            assert actual['repository_edits'] == item['quality']['repository_cer_edits']
        print('Previous 650-input scores reproduced for all four final types.')
    print('Arithmetic, pairing, refusal gates, decoder-device selection, environment selection and standalone report checks passed.')


if __name__ == '__main__':
    main()
