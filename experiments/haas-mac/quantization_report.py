#!/usr/bin/env python3
"""Render the public FLEURS Japanese quantization summary as one static HTML report.

Usage: python3.9 quantization_report.py --summary summary.json --out report.html
"""
import argparse, html, json, math
from pathlib import Path

PROFILE_IDS = ("q4_k_cpu", "q5_k_cpu", "q6_k_cpu", "q8_0_cpu", "f16_cpu", "f16_metal")
DATASET = "FLEURS Japanese test"
CHIP = "Apple M2 Ultra"
DEFAULT_PLACEMENT = {pid: pid != "f16_cpu" for pid in PROFILE_IDS}
STRATA = ("6-10s", "10-20s", "20-30s")
HEX = set("0123456789abcdef")
UNAVAILABLE = '<span class="u">unavailable</span>'
CSS = (
    "body{font-family:system-ui,sans-serif;max-width:62rem;margin:1rem auto;padding:0 1rem;color:#1c1c1c}"
    "h1{font-size:1.3rem;margin:0 0 .3rem} h2{font-size:1.05rem;margin:1.4rem 0 .3rem;border-bottom:1px solid #ddd}"
    ".tw{overflow-x:auto} table{border-collapse:collapse;font-size:.85rem;margin-top:.4rem}"
    "th,td{border:1px solid #ccc;padding:.28rem .5rem;text-align:left;white-space:nowrap} th{background:#f4f4f4}"
    "code{font-family:ui-monospace,monospace;font-size:.85em;word-break:break-all}"
    ".u{color:#888;font-style:italic} .note{color:#555;font-size:.88rem} ul{margin:.4rem 0;padding-left:1.2rem}"
)

def esc(value):
    return html.escape(str(value), quote=True)

def finite(value):
    """Return value when it is a finite JSON int/float (bools excluded), else None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value

def num(value, digits=2, suffix=""):
    """Format a finite number with fixed digits, or mark the metric unavailable."""
    value = finite(value)
    return UNAVAILABLE if value is None else f"{value:.{digits}f}{suffix}"

def table(headers, rows):
    """Render one table; headers are escaped, cells are already safe HTML."""
    head = "<tr>" + "".join(f'<th scope="col">{esc(h)}</th>' for h in headers) + "</tr>"
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="tw"><table>{head}{body}</table></div>'

def load(path):
    """Read summary.json and validate the fixed dataset, machine and six known profiles."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read summary {path}: {exc}")
    if not isinstance(data, dict) or not isinstance(data.get("dataset"), str):
        raise SystemExit("summary must be a JSON object with a dataset string")
    if data["dataset"] != DATASET:
        raise SystemExit(f"unsupported dataset {data['dataset']!r}, expected {DATASET!r}")
    machine = data.get("machine") or {}
    if not isinstance(machine, dict) or machine.get("chip") != CHIP:
        raise SystemExit(f"machine.chip must be exactly {CHIP!r}")
    for key in ("gold_count", "timed_count", "rounds", "threads"):
        if finite(data.get(key)) is None: raise SystemExit(f"missing or non-finite {key}")
    for key in ("cpu_cores", "gpu_cores", "memory_gb"):
        if finite(machine.get(key)) is None: raise SystemExit(f"missing or non-finite machine.{key}")
    profiles = {}
    for entry in data.get("profiles") or []:
        pid = entry.get("id") if isinstance(entry, dict) else None
        if pid not in PROFILE_IDS or pid in profiles:
            raise SystemExit(f"unknown or duplicate profile id: {pid!r}")
        weight, _, decoder = pid.rpartition("_")
        if (entry.get("weight_type") != weight or entry.get("decoder") != decoder
                or entry.get("model_id") != f"haas-ja-v1.0:{weight}"
                or entry.get("default_placement") is not DEFAULT_PLACEMENT[pid]):
            raise SystemExit(f"profile {pid}: invalid weight_type/decoder/model_id/default_placement")
        profiles[pid] = entry
    missing = [pid for pid in PROFILE_IDS if pid not in profiles]
    if missing:
        raise SystemExit(f"missing profiles: {', '.join(missing)}")
    return data, profiles

def main():
    parser = argparse.ArgumentParser(description="Render the public FLEURS quantization report.")
    parser.add_argument("--summary", required=True, type=Path, help="input summary.json")
    parser.add_argument("--out", required=True, type=Path, help="output HTML file")
    args = parser.parse_args()
    data, profiles = load(args.summary)
    machine = data["machine"]
    scope = (f"Dataset: {esc(data['dataset'])}. Gold utterances: {num(data['gold_count'], 0)}. "
             f"Timed subset: {num(data['timed_count'], 0)} inputs × {num(data['rounds'], 0)} rounds. "
             f"Hardware: {esc(machine['chip'])} — {num(machine['cpu_cores'], 0)} CPU cores, "
             f"{num(machine['gpu_cores'], 0)} GPU core(s) as MTL0, {num(machine['memory_gb'], 0)} GB, "
             f"{num(data['threads'], 0)} CPU threads.")

    # Choices are recomputed: eligible rows sit at default placement and are no
    # slower than the Q4_K CPU baseline on both p50 and p90.
    base = profiles["q4_k_cpu"].get("latency") or {}
    bp50, bp90 = finite(base.get("p50_ms")), finite(base.get("p90_ms"))
    eligible = []
    for pid in PROFILE_IDS:
        lat = profiles[pid].get("latency") or {}
        p50, p90 = finite(lat.get("p50_ms")), finite(lat.get("p90_ms"))
        ready = profiles[pid].get("default_placement") is True and None not in (bp50, bp90, p50, p90)
        if ready and p50 <= bp50 and p90 <= bp90:
            eligible.append(pid)

    def pick(candidates, section, key):
        """Winner id(s) holding the lowest finite value; equal best scores tie."""
        scored = [(finite((profiles[p].get(section) or {}).get(key)), p) for p in candidates]
        scored = [pair for pair in scored if pair[0] is not None]
        if not scored:
            return "unavailable", None
        best = min(pair[0] for pair in scored)
        return ", ".join(p for value, p in scored if value == best), best

    cpu_rows = [pid for pid in PROFILE_IDS if profiles[pid].get("decoder") == "cpu"]
    lit_w, lit_v = pick(eligible, "quality", "literal_cer")
    rep_w, rep_v = pick(eligible, "quality", "repository_cer")
    fast_w, fast_v = pick(cpu_rows, "latency", "p50_ms")
    choice_rows = [[esc("Lowest literal CER among eligible profiles (default placement, p50 and p90 not above Q4_K CPU)"), esc(lit_w),
                    num(lit_v * 100 if lit_v is not None else None, 3, "%")],
                   [esc("Lowest repository CER among the same eligible profiles"), esc(rep_w),
                    num(rep_v * 100 if rep_v is not None else None, 3, "%")],
                   [esc("Fastest p50 among CPU-decoder rows"), esc(fast_w), num(fast_v, 2, " ms")]]
    profile_rows = []
    for pid in PROFILE_IDS:
        p = profiles[pid]
        q, lat = p.get("quality") or {}, p.get("latency") or {}
        tag = f"{p['weight_type'].upper()} / {p['decoder'].upper()} ({'default' if p.get('default_placement') is True else 'override'})"
        lit, le, ln = (finite(q.get(k)) for k in ("literal_cer", "literal_edits", "literal_characters"))
        rep, re_, rn = (finite(q.get(k)) for k in ("repository_cer", "repository_edits", "repository_characters"))
        peak, size, paired = (finite(lat.get("peak_footprint_bytes")), finite(p.get("model_size_bytes")),
                              finite(lat.get("paired_reduction")))
        profile_rows.append([
            esc(tag), esc(p["model_id"]),
            num(lit * 100 if lit is not None else None, 3, "%") + (f" ({le}/{ln})" if None not in (lit, le, ln) else ""),
            num(rep * 100 if rep is not None else None, 3, "%") + (f" ({re_}/{rn})" if None not in (rep, re_, rn) else ""),
            num(lat.get("p50_ms")), num(lat.get("p90_ms")), num(lat.get("p99_ms")), num(lat.get("rtfx")),
            num(peak / 1048576 if peak is not None else None), num(size / 1048576 if size is not None else None),
            f"{paired * 100:+.2f}%" if paired is not None else UNAVAILABLE,
            num(lat.get("load_ms")), num(lat.get("first_warm_ms")),
        ])
    strata_rows = []
    for pid in PROFILE_IDS:
        st = profiles[pid].get("strata") or {}
        for name in STRATA:
            s = st.get(name) or {}
            strata_rows.append([esc(pid), esc(name), num(s.get("p50_ms")), num(s.get("p90_ms")),
                                num(s.get("p99_ms")), num(s.get("count"), 0)])
    threads = int(data["threads"])
    commands = []
    for pid in PROFILE_IDS:
        p = profiles[pid]
        if p["decoder"] == "cpu":
            placement = "default placement" if DEFAULT_PLACEMENT[pid] else "explicit override"
            commands.append((f"{pid} on CPU ({placement})",
                             f"speech asr {p['model_id']} --device gpu --threads {threads} --decode-on-cpu input.wav"))
        else:
            commands.append((f"{pid} on Metal (default placement)",
                             f"speech asr {p['model_id']} --device MTL0 --threads {threads} --decode-on-cpu=false input.wav"))
    commands.append(("Windows CPU example (qualified, unmeasured)",
                     f"speech.exe asr {profiles['q4_k_cpu']['model_id']} --device cpu --threads {threads} --decode-on-cpu input.wav"))
    commands_html = "".join(f"<li>{esc(label)}:<br><code>{esc(cmd)}</code></li>" for label, cmd in commands)
    tk = data.get("f16_token_timing") or {}
    token_items = (
        ("Gold inputs compared", "count", 0), ("Inputs with changed text", "text_changes", 0),
        ("Inputs with changed segments", "segment_changes", 0),
        ("Inputs with changed public token arrays", "token_array_changes", 0),
        ("Aligned token entries differing", "different_aligned_tokens", 0),
        ("Maximum aligned token time delta (ms)", "maximum_time_delta_ms", 2))
    token_rows = [[esc(label), num(tk.get(key), digits)] for label, key, digits in token_items]
    evidence = data.get("evidence") or {}
    ev_spec = (("quality_cpu", ("input_sha256", "manifest_sha256")),
               ("quality_metal", ("manifest_sha256",)), ("latency", ("manifest_sha256",)))
    prov = [("catalog_revision", data.get("catalog_revision"), 40), ("source_commit", data.get("source_commit"), 40),
            ("ggml_commit", data.get("ggml_commit"), 40), ("speech_sha256", data.get("speech_sha256"), 64)]
    prov += [(f"evidence.{name}.{key}", (evidence.get(name) or {}).get(key), 64) for name, keys in ev_spec for key in keys]
    prov += [(f"model_sha256[{pid}]", profiles[pid].get("model_sha256"), 64) for pid in PROFILE_IDS]
    prov_rows = [[esc(label), esc(value) if isinstance(value, str) and len(value) == length and set(value) <= HEX else UNAVAILABLE]
                 for label, value, length in prov]
    notes = (
        "Literal CER is the sum of Unicode codepoint edits divided by total reference codepoints, including punctuation, symbols and whitespace.",
        "Repository CER applies NFKC, then removes Unicode punctuation, symbols, separators and other whitespace. No case folding or alternate spellings are applied.",
        "Latency starts before the first PCM chunk preparation/serialization/send and ends on the terminal event. File reads, model load and warmup are excluded.",
        "Cohort percentiles are taken over per-input medians of the rounds. RTFx is total audio seconds divided by the sum of per-input medians. Small-n p99 is indicative only.",
        "The precision order F16 > Q8_0 > Q6_K > Q5_K > Q4_K ranks stored-weight precision only.",
        "Paired reduction vs Q4_K CPU is signed percent (positive means faster). Equal best scores tie. Missing or non-finite metrics cannot win and display as unavailable.",
    )
    notes_html = "".join(f"<li>{esc(note)}</li>" for note in notes)
    page = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>FLEURS Japanese quantization report</title><style>{CSS}</style></head>
<body>
<h1>FLEURS Japanese quantization report</h1>
<p>{scope}</p>
<h2>Recomputed choices</h2>
{table(["Choice", "Winner (profile ids)", "Value"], choice_rows)}
<h2>Profiles</h2>
{table(["Profile", "Model id", "Literal CER", "Repository CER", "p50 (ms)", "p90 (ms)", "p99 (ms)", "RTFx",
        "Peak (MiB)", "Weight (MiB)", "vs Q4_K", "Load (ms)", "First warm (ms)"], profile_rows)}
<p class="note">CER cells append edits over reference codepoints. The "vs Q4_K" column is the paired reduction in percent, where positive means faster.</p>
<p class="note">Peak is the maximum gross process footprint across the three timing workers. Load and first warm are fresh-worker startup observations and are excluded from the warm latency percentiles.</p>
<h2>Latency by duration stratum</h2>
{table(["Profile", "Stratum", "p50 (ms)", "p90 (ms)", "p99 (ms)", "n"], strata_rows)}
<h2>Reproduce</h2><ul>{commands_html}</ul>
<p class="note">The model id alone follows default placement: quantized formats decode on CPU, F16 decodes on Metal. F16 on CPU needs the explicit override.</p>
<h2>F16 token timing</h2>
<p class="note">This compares F16 CPU decoding with F16 Metal decoding on the timestamp-enabled public gold inputs ({num(data['gold_count'], 0)} utterances).</p>
{table(["Metric", "Value"], token_rows)}
<h2>Method notes</h2><ul>{notes_html}</ul>
<h2>Provenance</h2>
{table(["Identifier", "Value (hex)"], prov_rows)}
<p class="note">Generated from <a href="summary.json">summary.json</a>. Static page with no scripts or external assets.</p>
</body></html>
"""
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
