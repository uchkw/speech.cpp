"""Check campaign arithmetic, pairing, acceptance gates and standalone reports without a GPU.

Usage: python3 experiments/haas-mac/check.py [--previous PREVIOUS_EXPERIMENT_DIRECTORY]
"""
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
import compare
import render


def refused(operation, message, expected=None):
    try:
        operation()
    except ValueError as error:
        if expected is not None:
            assert expected in str(error), (expected, str(error))
        return
    raise AssertionError(message)


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
    assert compare.environment(['GGML_METAL_MM_TILE_N=64'])['GGML_METAL_MM_TILE_N'] == '64'
    for key in ['SPEECH_HAAS_CPU_REPACK', 'SPEECH_HAAS_FFN_F16', 'SPEECH_HAAS_FRONTEND_PARALLEL']:
        for value in ['0', '1']:
            assert compare.environment([f'{key}={value}'])[key] == value
        refused(lambda: compare.environment([f'{key}=2']), 'An invalid binary switch was accepted.', expected='0 or 1')
        with patch.dict(compare.os.environ, {key: '1'}):
            refused(lambda: compare.environment([]), 'An inherited switch was not recorded.', expected='inherited')
    subset = [{'file': f'{group}-{i}', 'seconds': seconds + i / 10, 'stratum': group}
              for group, seconds in [('6-10s', 6), ('10-20s', 10), ('20-30s', 20)] for i in range(10)]
    selected = compare.screen_inputs(list(reversed(subset)))
    assert len(selected) == 9 and {group: sum(r['stratum'] == group for r in selected) for group in ['6-10s', '10-20s', '20-30s']} == {'6-10s': 3, '10-20s': 3, '20-30s': 3}
    assert [r['file'] for r in selected] == [f'{group}-{i}' for group in ['6-10s', '10-20s', '20-30s'] for i in [0, 4, 9]]
    assert compare.percentile([1., 4., 9.], .9) == 8.
    with tempfile.TemporaryDirectory(prefix='haas-campaign-check-') as temporary:
        review_regressions(Path(temporary))
    campaign = compare.read(compare.HERE / 'campaign.json')
    html = render.report_html({'measure_id': '999', 'title': '<script>bad()</script>', 'status': 'pending', 'reason': '<b>not markup</b>'}, '999.json', campaign)
    assert '<script>' not in html and '&lt;script&gt;' in html
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
    print('Arithmetic, pairing, refusal gates, environment selection and standalone report checks passed.')


if __name__ == '__main__':
    main()
