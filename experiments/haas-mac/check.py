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

sys.dont_write_bytecode = True
import compare
import render


def refused(operation, message):
    try:
        operation()
    except ValueError:
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
    assert compare.percentile([1., 4., 9.], .9) == 8.
    with tempfile.TemporaryDirectory(prefix='haas-campaign-check-') as temporary:
        root = Path(temporary); run = root / 'run'; run.mkdir()
        compare.write(root / '001-example.json', {'measure_id': '001', 'title': 'Example', 'status': 'pending'})
        compare.write(run / 'summary.json', {'scope': 'full', 'types': {}})
        original = compare.REPORTS; compare.REPORTS = root
        try:
            for status in ['refused', 'skipped', 'unavailable', 'failed']:
                compare.write(root / 'checks.json', {'checks': [{'name': 'Required stage', 'status': status, 'required': True}]})
                options = SimpleNamespace(measure='001', title=None, run=str(run), checks=str(root / 'checks.json'),
                                          verdict='accepted', reason='Should fail', artifact_root=str(root))
                refused(lambda: compare.record(options), f'{status} was accepted as a required pass.')
        finally:
            compare.REPORTS = original
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
