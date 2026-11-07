"""Render a static, standalone model-comparison page from one public aggregate summary JSON.
Usage: python3 experiments/haas-mac/quantization_report.py --summary summary.json --out report.html
Only the named aggregate JSON is read; measurements, specifications, audio and worker output are never opened.
"""
import argparse
import json
import math
from pathlib import Path

from render import document, table, text

PROFILES = ['q4_k_cpu', 'q5_k_cpu', 'q6_k_cpu', 'q8_0_cpu', 'f16_cpu', 'f16_metal']
APP_PROFILES = ['q4_k_cpu', 'q5_k_cpu', 'q6_k_cpu', 'q8_0_cpu', 'f16_metal']
COHORTS = [('fleurs', 'Public'), ('private', 'Private')]
STRATA = {'fleurs': ['6-10s', '10-20s', '20-30s'], 'private': ['2-<5s', '5-<10s', '10-<20s', '20-30s']}
NAMES = {'q4_k_cpu': 'Q4_K · CPU (existing Voimem model)', 'q5_k_cpu': 'Q5_K · CPU', 'q6_k_cpu': 'Q6_K · CPU',
         'q8_0_cpu': 'Q8_0 · CPU (bare speech alias)', 'f16_cpu': 'F16 · CPU (CLI-controlled)', 'f16_metal': 'F16 · Metal'}
PINS = [('Source commit', 'source_commit'), ('ggml commit', 'ggml_commit'), ('Speech binary SHA-256', 'speech_sha256'),
        ('Build manifest SHA-256', 'build_manifest_sha256'), ('Model revision', 'model_revision'),
        ('CPU quality runner commit', 'cpu_quality_runner_commit'), ('CPU quality worker SHA-256', 'cpu_quality_worker_sha256'),
        ('Placement runner commit', 'placement_runner_commit'), ('Placement worker SHA-256', 'placement_worker_sha256')]
TOKEN_COUNTS = [('inputs', 'Inputs compared', 0, ''), ('text_changed_inputs', 'Inputs with changed text', 0, ''),
                ('segments_changed_inputs', 'Inputs with changed segments', 0, ''),
                ('token_array_changed_inputs', 'Inputs with changed token arrays', 0, ''),
                ('token_count_changed_inputs', 'Inputs with changed token count', 0, ''), ('changed_token_texts', 'Token texts changed', 0, ''),
                ('changed_token_starts', 'Token starts changed', 0, ''), ('changed_token_ends', 'Token ends changed', 0, ''),
                ('max_token_time_delta_seconds', 'Largest token time difference', 3, ' s')]


def number(value):
    """The value itself only when it is a finite int or float; bools and strings never pass (type() is exact, so bool is excluded)."""
    return value if type(value) in (int, float) and math.isfinite(value) else None


def count(value):
    value = number(value)
    return f'{int(value):,}' if value is not None and value >= 0 and value == int(value) else 'unavailable'


def cell(value, digits=2, suffix='', scale=1.0):
    value = number(value)
    return 'unavailable' if value is None else f'{value * scale:,.{digits}f}{suffix}'


def cohort_block(summary, profile, cohort):
    return summary.get('profiles', {}).get(profile, {}).get('latency', {}).get(cohort, {})


def edits(quality, prefix):
    return f'{count(quality.get(prefix + "_edits"))} / {count(quality.get(prefix + "_characters"))}'


def meets_gate(summary, profile):
    for cohort, _ in COHORTS:
        mine, base = cohort_block(summary, profile, cohort), cohort_block(summary, 'q4_k_cpu', cohort)
        if any(number(mine.get(f)) is None or number(base.get(f)) is None or number(mine[f]) > number(base[f])
               for f in ('p50_ms', 'p90_ms')):
            return False
    return True


def leaders(pool, empty, note):
    pool = [(value, k) for value, k in pool if number(value) is not None]
    if not pool:
        return empty
    floor = min(value for value, _ in pool)
    return ' = '.join(f'{NAMES[k]} ({note(value)})' for value, k in pool if value == floor)


def latency_table(summary, cohort):
    private = cohort == 'private'
    counts = summary.get('cohorts', {}).get(cohort, {})
    caption = (f'{"Private" if private else "Public"} timed cohort · {count(counts.get("count"))} inputs '
               f'× {count(counts.get("rounds"))} rounds per profile')
    head = ['Profile (model · decoder)', 'p50 ms', 'p90 ms', 'p99 ms', 'Paired reduction vs Q4_K', 'RTFx', 'Peak MiB']
    if private:
        head += ['Exact same vs Q4_K', 'Normalized same vs Q4_K', 'Private CER']
    rows = []
    for k in PROFILES:
        m = cohort_block(summary, k, cohort)
        row = [text(NAMES[k]), cell(m.get('p50_ms')), cell(m.get('p90_ms')), cell(m.get('p99_ms')),
               cell(m.get('paired_reduction_vs_q4_percent'), 1, '%'), cell(m.get('rtfx'), 1), cell(m.get('peak_footprint_bytes'), 1, scale=1 / 2**20)]
        if private:
            agree = m.get('round0_agreement_vs_q4', {})
            row += [f'{count(agree.get("literal_same"))} / {count(agree.get("count"))}',
                    f'{count(agree.get("normalized_same"))} / {count(agree.get("count"))}', 'unavailable']
        rows.append(row)
    return table(head, rows, caption)


def body_html(summary, source_name):
    gold, pub, priv = (summary.get('cohorts', {}).get(key, {}) for key in ('fleurs_gold', 'fleurs', 'private'))
    device, threads, chip = (summary.get('hardware', {}).get(key) for key in ('model_device', 'threads', 'chip'))
    gate = {k: meets_gate(summary, k) for k in APP_PROFILES}
    cer = {key: [(summary.get('profiles', {}).get(k, {}).get('quality', {}).get(key), k) for k in APP_PROFILES if gate[k]]
           for key in ('repository_cer', 'literal_cer')}
    fastest = {cohort: [(cohort_block(summary, k, cohort).get('p50_ms'), k) for k in APP_PROFILES] for cohort, _ in COHORTS}
    cli_cpu = [k for k in PROFILES if k.endswith('_cpu')]
    cli_fastest = {cohort: [(cohort_block(summary, k, cohort).get('p50_ms'), k) for k in cli_cpu] for cohort, _ in COHORTS}
    picks = [['Lowest normalized public CER, at p50 and p90 equal or better than Q4_K on both timed cohorts',
              leaders(cer['repository_cer'], 'No profile meets the latency gate', lambda v: f'{v:.3%}')],
             ['Lowest literal public CER at the same latency gate', leaders(cer['literal_cer'], 'No profile meets the latency gate', lambda v: f'{v:.3%}')],
             ['Fastest app placement by public cohort median', leaders(fastest['fleurs'], 'unavailable', lambda v: f'{v:.2f} ms')],
             ['Fastest app placement by private cohort median', leaders(fastest['private'], 'unavailable', lambda v: f'{v:.2f} ms')],
             ['Fastest with explicit CLI CPU decoding, public p50', leaders(cli_fastest['fleurs'], 'unavailable', lambda v: f'{v:.2f} ms')],
             ['Fastest with explicit CLI CPU decoding, private p50', leaders(cli_fastest['private'], 'unavailable', lambda v: f'{v:.2f} ms')]]
    body = (f'<p class="eyebrow">Public aggregate comparison</p><h1>Haas model quantization on Apple Silicon</h1>'
            f'<p>Six fixed profiles of one model on {text(chip or "this device")}; Q4_K is the existing Voimem model baseline and the bare speech alias is Q8_0. '
            f'Cohorts: FLEURS Japanese gold {count(gold.get("count"))} inputs · public timed {count(pub.get("count"))} × {count(pub.get("rounds"))} rounds · '
            f'private timed {count(priv.get("count"))} × {count(priv.get("rounds"))} rounds · machine-readable source: <a href="{text(source_name)}">{text(source_name)}</a>.</p>'
            '<section class="panel"><h2>Actionable choices</h2>' + table(['Decision', 'Measured pick (ties included)'], [[text(a), text(b)] for a, b in picks])
            + '<p class="muted">Computed from unrounded values; absent gate or quality metrics skip a profile rather than counting it as zero.</p></section>')
    body += ('<h2>Public timed cohort</h2>' + latency_table(summary, 'fleurs') + '<h2>Private timed cohort</h2>' + latency_table(summary, 'private')
             + '<p class="muted">Warm latency starts immediately before the first PCM16 send and ends at the terminal result; it excludes disk reading, load and warmup. '
             'Each clip contributes its median over three rounds before the cohort percentile; this is whole worker-request time, not decoder-only and not whole app.</p>'
             + '<h2>Public recognition error</h2>' + table(
                 ['Profile (model · decoder)', 'Literal CER', 'Literal edits / characters', 'Normalized CER', 'Normalized edits / characters'],
                 [[text(NAMES[k]), cell(q.get('literal_cer'), 3, '%', 100), edits(q, 'literal'), cell(q.get('repository_cer'), 3, '%', 100), edits(q, 'repository')]
                  for k in PROFILES for q in [summary.get('profiles', {}).get(k, {}).get('quality', {})]],
                 f'FLEURS Japanese gold cohort · {count(gold.get("count"))} inputs · timestamps {"enabled" if gold.get("timestamps") else "disabled"}')
             + '<p class="muted">Literal CER compares Unicode code points, including punctuation and whitespace; repository normalized CER applies NFKC, then removes '
             'Unicode punctuation, symbols, separators and whitespace — no case folding or alternate spellings. Both are total edits over total reference characters.</p>')
    tokens = summary.get('f16_placement_public_output')
    if tokens is not None:
        rows = [[text(label), cell(tokens.get(key), digits, suffix)] for key, label, digits, suffix in TOKEN_COUNTS]
        body += ('<section class="panel warning"><h2>F16 public token timing</h2><p>The two F16 profiles are separate measured placements; on the FLEURS Japanese gold cohort '
                 'their token timing differs while transcripts stay stable — measured counts only:</p>' + table(['Measured count', 'Value'], rows) + '</section>')
    commands = []
    for k in PROFILES:
        model = summary.get('models', {}).get(summary.get('profiles', {}).get(k, {}).get('type'), {})
        model_id = model.get('model_id')
        flag = '--decode-on-cpu=false' if summary.get('profiles', {}).get(k, {}).get('decoder') == 'metal' else '--decode-on-cpu'
        command = f'speech asr {model_id} --device {device} --threads {threads} {flag} input.wav' if model_id else 'unavailable'
        commands.append([text(NAMES[k]), f'<code>{text(model_id or "unavailable")}</code>',
                         cell(model.get('size_bytes'), 1, scale=1 / 2**20), f'<code>{text(command)}</code>'])
    order = ' > '.join(part.upper() for part in summary.get('precision_order_high_to_low', [])) or 'unavailable'
    windows = f'speech.exe asr {summary.get("models", {}).get("q4_k", {}).get("model_id")} --device cpu --threads {threads} --decode-on-cpu input.wav'
    body += ('<section class="panel"><h2>Choosing and running a model</h2>'
             f'<p>In the app: Settings → Recognition → Model → Save settings persists the fully qualified model value. Prepare loads it now, the next recording loads '
             f'the new model before capture, and a file import with a speech region prepares the selected model too — no app restart is required. Device stays auto at {text(threads or "default")} threads.</p>'
             '<p>The four quantized formats default to CPU decode; F16 defaults to Metal on this accelerator, and F16 CPU still requires the explicit CLI flag.</p>'
             + table(['Profile', 'Fully qualified model ID', 'Weights MiB', 'macOS command (copyable)'], commands)
             + f'<p>Windows CPU equivalent with an explicit full model name: <code>{text(windows)}</code> — Windows metrics: unmeasured.</p>'
             f'<p class="warning">Weight-format precision {text(order)} ranks stored-weight precision only; it is a separate selection preference from the public recognition error above.</p></section>')
    pins = [[text(label), f'<code>{text(summary.get("build", {}).get(key) or "unavailable")}</code>'] for label, key in PINS]
    body += ('<section class="panel"><h2>Measurement identity</h2>'
             f'<p>Device {text(device or "unavailable")} · chip {text(chip or "unavailable")} · {text(summary.get("hardware", {}).get("os") or "unlisted OS")} · CPU decode pinned to {text(threads or "unavailable")} threads.</p>'
             + table(['Pin from build', 'Value'], pins) + '</section>')
    body += '<details><summary>Duration strata, load and first request, aggregate vs paired deltas</summary>'
    strata = [[text(label), text(NAMES[k]), text(name), count(s.get('count')), cell(s.get('p50_ms')), cell(s.get('p90_ms')), cell(s.get('p99_ms'))]
              for cohort, label in COHORTS for k in PROFILES
              for name in STRATA[cohort] for s in [cohort_block(summary, k, cohort).get('strata', {}).get(name, {})]]
    body += table(['Cohort', 'Profile', 'Duration stratum', 'Inputs', 'p50 ms', 'p90 ms', 'p99 ms'], strata, 'Fixed duration strata recorded in the summary')
    rows = [[text(NAMES[k]), text(label), cell(m.get('median_load_ms')), cell(m.get('median_first_request_ms')), cell(d.get('p50_ms')), cell(d.get('p90_ms')),
             cell(d.get('p99_ms')), cell(m.get('paired_reduction_vs_q4_percent'), 1, '%')]
            for cohort, label in COHORTS for k in PROFILES
            for m in [cohort_block(summary, k, cohort)] for d in [m.get('aggregate_delta_vs_q4_ms', {})]]
    body += table(['Profile', 'Cohort', 'Median load ms', 'Median first request ms', 'Aggregate p50 delta ms', 'Aggregate p90 delta ms', 'Aggregate p99 delta ms',
                   'Paired reduction vs Q4_K'], rows, 'Aggregate deltas subtract Q4_K percentiles; the paired reduction is a median of per-input ratios, so the two can disagree') + '</details>'
    limits = ['Peak MiB is the maximum process footprint across the entire cohort run, not the footprint of one duration stratum or input.',
              f'p99 rests on only {count(pub.get("count"))} public and {count(priv.get("count"))} private inputs — a tiny sample; treat it as indicative.',
              'The private cohort has no gold reference: private CER is unavailable, agreement appears as exact and normalized counts against Q4_K, and free text from private sections is ignored.',
              'Measured scope is the whole worker request with no keywords or VAD context — not decoder-only, and nothing here is a whole-app claim.',
              'Unrecognized summary keys are ignored by this page.']
    return body + '<h2>Conditions and limitations</h2><ul>' + ''.join(f'<li>{text(item)}</li>' for item in limits) + '</ul>'


def main():
    parser = argparse.ArgumentParser(description='Render the public quantization comparison page.')
    parser.add_argument('--summary', required=True, help='Public aggregate summary JSON')
    parser.add_argument('--out', required=True, help='HTML output path')
    args = parser.parse_args()
    source = Path(args.summary)
    Path(args.out).write_text(document('Haas model quantization · Public comparison', body_html(json.loads(source.read_text()), source.name)))
    print(args.out)


if __name__ == '__main__':
    main()
