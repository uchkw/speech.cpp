"""Render the fork's English Haas ledger as standalone local HTML.

Usage: python3 experiments/haas-mac/render.py
JSON reports are authoritative. Rendering never changes a verdict or reads large artifacts.
"""
from collections import Counter
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'reports/haas-mac'
CSS = '''
:root{color-scheme:light;font:16px/1.55 system-ui,sans-serif;color:#172738;background:#f3f6f9}
*{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:auto;padding:34px 22px 64px}
a{color:#075ca8;text-underline-offset:3px}a:focus-visible,summary:focus-visible{outline:3px solid #c36b00;outline-offset:4px}
h1{font-size:clamp(1.8rem,4vw,2.7rem);line-height:1.18;margin:12px 0 18px}h2{font-size:1.35rem;margin:30px 0 12px}
h3{font-size:1.05rem;margin:22px 0 10px}p{max-width:90ch}small,.muted{color:#506174}.eyebrow{font-size:.8rem;letter-spacing:.12em;text-transform:uppercase;color:#506174}
.panel{background:white;border:1px solid #d5dfe8;border-radius:10px;padding:20px;margin:18px 0}.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;vertical-align:top;padding:11px 10px;border-bottom:1px solid #dce4eb}th{font-size:.85rem;background:#f2f6fa}td.num,th.num{text-align:right;white-space:nowrap}
caption{text-align:left;font-weight:600;padding:8px 0}code{font: .87em ui-monospace,monospace;overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.83rem}
.status{display:inline-block;border:1px solid;border-radius:5px;padding:2px 8px;font-size:.8rem;font-weight:650;text-transform:uppercase}
.pending,.not_applicable{color:#405269;background:#edf1f5}.in_progress{color:#174c7b;background:#e8f2ff}.accepted,.validated_mac{color:#16572f;background:#e7f5eb}.rejected{color:#8a2631;background:#fff0f0}.conditional,.pending_official_reference{color:#754700;background:#fff4d8}.blocked{color:#663c83;background:#f4eafa}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px}.metric strong{display:block;font-size:1.55rem;line-height:1.2}.metric span{font-size:.85rem;color:#506174}
details{margin:14px 0}summary{cursor:pointer;font-weight:600}.warning{border-left:4px solid #bc7600;padding-left:15px}svg{max-width:100%;height:auto;display:block}footer{margin-top:36px;font-size:.85rem;color:#506174}
@media(max-width:600px){main{padding:22px 12px}.panel{padding:14px}th,td{padding:9px 7px}h1{overflow-wrap:anywhere}}
'''


def text(value):
    return escape(str(value), quote=True)


def badge(status):
    return f'<span class="status {text(status)}">{text(status.replace("_", " "))}</span>'


def number(value, digits=2):
    return '—' if value is None else f'{value:,.{digits}f}'


def table(headers, rows, caption=None):
    head = ''.join(f'<th scope="col">{text(h)}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
    label = f'<caption>{text(caption)}</caption>' if caption else ''
    return f'<div class="scroll" tabindex="0" role="region" aria-label="{text(caption or "Results table")}"><table>{label}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def document(title, body):
    return f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>{text(title)}</title><style>{CSS}</style></head><body><main>{body}<footer>Local experiment report · JSON is the source of truth · No external assets or telemetry</footer></main></body></html>\n'


def source_link(build, repository):
    commit = build.get('source_commit')
    if not build.get('source_verified') or not commit:
        return '<strong>Source commit unverified</strong>'
    return f'<a href="{text(repository)}/commit/{text(commit)}"><code>{text(commit[:12])}</code></a>'


def latency_chart(sides):
    maximum = max(v['latency']['p50_ms'] for v in sides.values())
    height = 45 + 38 * len(sides)
    bars = []
    for i, (side, values) in enumerate(sides.items()):
        value = values['latency']['p50_ms']; y = 22 + i * 38
        color = '#1765a5' if side == 'candidate' else '#637d94'
        bars.append(f'<text x="0" y="{y + 17}" fill="#172738" font-size="13">{text(side)}</text><rect x="110" y="{y}" width="{300 * value / maximum:.1f}" height="24" rx="3" fill="{color}"/><text x="{120 + 300 * value / maximum:.1f}" y="{y + 17}" fill="#172738" font-size="13">{value:.2f} ms</text>')
    return f'<svg viewBox="0 0 520 {height}" role="img" aria-label="Median warm request latency in milliseconds, lower is faster"><title>Median warm request latency, lower is faster</title>{"".join(bars)}</svg>'


def report_html(entry, filename, campaign, trials=()):
    title = f"{entry['measure_id']} · {entry['title']}"
    body = f'<nav><a href="index.html">← All measures</a></nav><p class="eyebrow">Haas on Apple Silicon</p><h1>{text(title)}</h1>{badge(entry["status"])}<p>{text(entry["reason"])}</p>'
    if entry.get('prior_evidence_note'):
        body += f'<p class="warning">{text(entry["prior_evidence_note"])}</p>'
    body += f'<p><a href="{text(filename)}">Machine-readable report JSON</a> · <a href="{text(campaign["tracking_issue"])}">Tracking issue</a></p>'
    conditions = {**campaign, **entry.get('conditions', {})}
    if entry.get('types'):
        body += f'<p><strong>Measured scope:</strong> {text(conditions["latency_count"])} inputs × {text(conditions["rounds"])} latency rounds per side; {text(conditions["quality_count"])} quality inputs per side. CPU decoding with {text(conditions["threads"])} threads.</p>'
    if trials:
        body += '<h2>Recorded variants</h2><p>A rejected variant does not close the whole measure.</p>'
        body += table(['Trial', 'Verdict', 'Reason'], [[f'<a href="{text(p.with_suffix(".html").name)}">{text(e["measure_id"])} · {text(e["title"])}</a>', badge(e['status']), text(e['reason'])] for p, e in trials])
    if not entry.get('types'):
        body += '<section class="panel"><h2>Evidence</h2><p>No new measurement is recorded. Pending work is not a measured failure or a speed-up.</p></section>'
    for kind, profile in entry.get('types', {}).items():
        sides = profile['sides']
        body += f'<section class="panel"><h2>{text(kind.upper())}</h2>{latency_chart(sides)}'
        rows = []
        for side, v in sides.items():
            q = v['quality']; l = v['latency']
            rows.append([text(side), number(l['p50_ms']), number(l['p90_ms']), number(l['p99_ms']),
                         number(q['literal_cer'] * 100, 4) + '%', number(q['repository_cer'] * 100, 4) + '%',
                         number(v['peak_memory_bytes'] / 2**30, 3)])
        body += table(['Build', 'p50 ms', 'p90 ms', 'p99 ms', 'Literal CER', 'Repository CER', 'Peak GiB'], rows, 'Warm latency and recognition error')
        changes = profile.get('comparisons', {})
        if changes:
            rows = [[text(label), number(v['latency_reduction_percent']) + '%', str(v['text_changed']),
                     str(v['public_tokens_changed']), str(v['segments_changed']),
                     f"{v['literal_better']} / {v['literal_worse']}", f"{v['repository_better']} / {v['repository_worse']}"]
                    for label, v in changes.items()]
            body += table(['Reference', 'Latency reduction', 'Texts changed', 'Tokens/times changed', 'Segments changed',
                           'Literal better / worse', 'Repository better / worse'], rows, conditions.get('comparison_reference', 'Candidate against the previous accepted build and the fixed anchor'))
            body += '<p class="muted">Positive latency reduction is faster. Relative and cumulative gains are direct paired comparisons, not added percentages. A matching CER does not mean matching transcripts.</p>'
        rows = [[text(side), text(stratum), number(v['p50_ms']), number(v['p90_ms']), number(v['p99_ms']), str(v['count'])]
                for side, values in sides.items() for stratum, v in values.get('strata', {}).items()]
        if rows:
            body += table(['Build', 'Audio length', 'p50 ms', 'p90 ms', 'p99 ms', 'Inputs'], rows, 'Latency by input length')
        body += '<details><summary>Load, first request and stage timings</summary>'
        body += table(['Build', 'Load to ready ms', 'First request ms', 'RTFx', 'Stage evidence'],
                      [[text(side), number(v.get('load_ms')), number(v.get('first_request_ms')), number(v['latency'].get('rtfx')),
                        text(v.get('stage_status', 'unavailable'))] for side, v in sides.items()])
        for side, v in sides.items():
            for name, times in v.get('stages', {}).items():
                body += table(['Stage', 'Median ms'], [[text(k.removesuffix('_ms').replace('_', ' ')), number(value)] for k, value in times.items()], f'{side}: {name}')
        body += '</details></section>'
    if entry.get('product_summary'):
        product = entry['product_summary']
        body += '<h2>Product timing screen</h2><p>' + text(product['method']) + '</p>'
        body += table(['Type', 'Median candidate / baseline latency ratio'],
                      [[text(kind), number(ratio, 4)] for kind, ratio in product['median_shape_latency_ratio'].items()],
                      'Ratios above one are slower; Q4_K is an unchanged-path control')
    if entry.get('builds'):
        rows = [[text(side), source_link(build, campaign['repository']), f'<code>{text(build.get("ggml_commit", "unverified"))}</code>',
                 f'<code>{text(build["speech"]["sha256"])}</code>', f'<code>{text(json.dumps(entry.get("execution_environments", {}).get(side, {})))}</code>'] for side, build in entry['builds'].items()]
        body += '<h2>Build identity</h2>' + table(['Build', 'Source commit', 'ggml commit', 'Binary SHA-256', 'Explicit environment'], rows)
    body += '<h2>Checks</h2>'
    checks = entry.get('checks', [])
    if checks:
        body += table(['Check', 'Required', 'Status', 'Evidence'], [[text(c['name']), 'yes' if c.get('required', True) else 'no',
                         text(c['status']), text(c.get('evidence', c.get('log', ''))) + ('<br>' + text(c['reason']) if c.get('reason') else '')] for c in checks])
    else:
        body += '<p>No check result is recorded. Refused, skipped and unavailable checks are never passes.</p>'
    body += '<h2>Conditions and limitations</h2><p>' + text(conditions['latency_definition']) + '</p><p>' + text(conditions['aggregation']) + '</p>'
    body += '<p><strong>Literal CER:</strong> ' + text(campaign['literal_cer']) + '</p><p><strong>Repository CER:</strong> ' + text(campaign['repository_cer']) + '</p>'
    body += '<ul>' + ''.join(f'<li>{text(item)}</li>' for item in conditions['limitations']) + '</ul>'
    details = {k: entry[k] for k in ['scope', 'conditions', 'models', 'builds', 'implementation_checkpoint', 'product_summary', 'input_hash', 'assets_sha256', 'evidence'] if k in entry}
    body += '<details><summary>Reproduction metadata and local evidence paths</summary><p>Large logs, models and audio are not included in the versioned report.</p><pre>' + text(json.dumps(details, ensure_ascii=False, indent=2)) + '</pre></details>'
    return document(title, body)


def main():
    campaign = json.loads((ROOT / 'experiments/haas-mac/campaign.json').read_text())
    records = [(p, json.loads(p.read_text())) for p in sorted(REPORTS.glob('*.json'))]
    trials = {}
    for path, entry in records:
        if entry.get('parent_measure_id'):
            trials.setdefault(entry['parent_measure_id'], []).append((path, entry))
    for path, entry in records:
        if entry['status'] != 'pending' or entry['measure_id'] in trials:
            path.with_suffix('.html').write_text(report_html(entry, path.name, campaign, trials.get(entry['measure_id'], [])))
    measures = [(p, e) for p, e in records if e.get('kind') != 'preliminary' and not e.get('parent_measure_id')]
    counts = Counter(e['status'] for _, e in measures)
    body = '<p class="eyebrow">Independent fork · Sequential measurements</p><h1>Haas on Apple Silicon</h1><p>' + text(campaign['purpose']) + '</p>'
    body += f'<p><a href="{text(campaign["tracking_issue"])}">Tracking issue</a> · <a href="{text(campaign["repository"])}">Source repository</a></p>'
    body += '<div class="grid">' + ''.join(f'<div class="panel metric"><strong>{counts[status]}</strong><span>{text(status.replace("_", " ").title())}</span></div>' for status in ['accepted', 'validated_mac', 'in_progress', 'conditional', 'rejected', 'blocked', 'pending']) + '</div>'
    for path, entry in records:
        if entry.get('kind') == 'preliminary':
            body += f'<div class="panel warning"><strong>Preliminary evidence</strong><p>{text(entry["reason"])}</p><a href="{text(path.with_suffix(".html").name)}">{text(entry["title"])}</a></div>'
    rows = []
    for path, entry in measures:
        gains = []
        for kind, p in entry.get('types', {}).items():
            comparisons = p.get('comparisons', {})
            if comparisons:
                gains.append(f"{text(kind)}: {number(comparisons['relative']['latency_reduction_percent'])}% / {number(comparisons['cumulative']['latency_reduction_percent'])}%")
        target = path.name if entry['status'] == 'pending' and entry['measure_id'] not in trials else path.with_suffix('.html').name
        reason = entry['reason'] if entry['measure_id'] not in trials else f"{len(trials[entry['measure_id']])} recorded variant(s). The measure is {entry['status'].replace('_', ' ')}."
        rows.append([text(entry['measure_id']), f'<a href="{text(target)}">{text(entry["title"])}</a>',
                     badge(entry['status']), '<br>'.join(gains) or '—', text(reason)])
    body += table(['#', 'Measure', 'Verdict', 'Relative / cumulative latency reduction', 'Reason'], rows, f'{len(measures)} measures')
    body += '<h2>How to read these results</h2><p>' + text(campaign['aggregation']) + '</p><p>Every verdict belongs to its recorded build, model, inputs and settings. No pending entry claims a result. Changes to precision are judged with both error measures; higher precision does not guarantee lower recognition error.</p>'
    (REPORTS / 'index.html').write_text(document('Haas on Apple Silicon · Measurement ledger', body))
    print(REPORTS / 'index.html')


if __name__ == '__main__':
    main()
