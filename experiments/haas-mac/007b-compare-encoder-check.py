#!/usr/bin/env python3
"""Offline synthetic fixture for 007b-compare-encoder.py.

Builds tiny fake emitter output roots and JSON Lines reports in a temporary
directory and drives the real comparator CLI.  The passing case checks the
original/OFF bit-exact gate and known OFF/ON metrics (signed zeros, an
all-zero reference, a perfect match, a changed argmax); the relative case
checks that a relative tensor record resolves against the working directory;
every other case asserts a nonzero exit with an actionable message,
including early, duplicate and mixed completion markers.  Everything is
synthetic; no model, audio or trial data is read.
"""

import json
import math
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

COMPARATOR = Path(__file__).resolve().parent / "007b-compare-encoder.py"
SIDES = ("original", "off", "on")
FINAL = {"passed": True, "official_reference_checked": False, "recognition_checked": False, "timing": False}
TINY_KEY = "synthetic-1/normal/projected.f32"
TINY_ROWS = [[1.0, 2.0]]

# ROWS_A has a first-maximum tie, a -0.0 reference and, versus ROWS_A_ON, two
# bit changes, one zero-reference mismatch and one changed argmax frame;
# ROWS_B compares equal to itself; ROWS_C is an all-zero reference.
ROWS_A = [[1.0, 3.0, 2.0, 3.0, 0.0],
          [1.0, 0.99, 0.1, 0.05, -0.5],
          [2.0, -1.0, 0.25, -0.0, 1.0]]
ROWS_A_ON = [[1.0, 3.0, 2.0, 3.0, 0.0],
             [1.0, 0.99, 0.1, 0.05, -0.5],
             [2.0, -1.0, 0.25, 0.25, 3.0]]
ROWS_B = [[0.5, -0.25, 0.75, 0.125], [-1.0, 0.0, 0.0625, 2.0]]
ROWS_C = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
ROWS_C_ON = [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]


def pack(rows):
    """Pack row-major samples as explicitly little-endian float32."""
    flat = [value for row in rows for value in row]
    return struct.pack("<%df" % len(flat), *flat)


def f32(value):
    """Round to the float32 value the comparator reads back."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def first_max(row):
    return row.index(max(row))


def write_side(base, name, tensors, relative=False):
    """Write one side's tensors and a report that mimics the emitter."""
    root = base / name
    for key, rows in tensors.items():
        target = root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(pack(rows))
    records = [{"tensor": (name + "/" + key) if relative else str(root / key),
                "frames": len(rows), "width": len(rows[0]),
                "channel_argmax": [first_max(row) for row in rows]}
               for key, rows in tensors.items()]
    records.append({"frames": 1, "retained": False, "flash_attention": 0, "expected": 0})
    records.append({"mask_layer": 0, "max_abs": 0.0, "zero": 0, "subnormal": 0})
    records.append(dict(FINAL))
    (base / (name + ".jsonl")).write_text("".join(json.dumps(record) + "\n" for record in records))


def tiny(base, rows=TINY_ROWS):
    for side in SIDES:
        write_side(base, side, {TINY_KEY: rows})


def edit(base, side, change):
    """Rewrite one side's report after mutating its records."""
    path = base / (side + ".jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    change(records)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def mutated(side, change):
    def setup(base):
        tiny(base)
        edit(base, side, change)
    return setup


def run(base, report=None, relative=False):
    """Run the comparator CLI; relative mode matches the emitter's working directory."""
    command = [sys.executable, "-B", str(COMPARATOR)]
    for side in SIDES:
        names = (side, side + ".jsonl") if relative else (str(base / side), str(base / (side + ".jsonl")))
        command += ["--%s-dir" % side, names[0], "--%s-jsonl" % side, names[1]]
    if report is not None:
        command += ["--report", str(report)]
    return subprocess.run(command, capture_output=True, text=True, cwd=str(base) if relative else None)


def strict_json(text):
    def reject(name):
        raise ValueError("nonstandard JSON constant %s" % name)
    return json.loads(text, parse_constant=reject)


def check_passing(base):
    """The passing case reports the known gate result and OFF/ON metrics as strict JSON."""
    original = {"synthetic-2/normal/projected.f32": ROWS_A,
                "synthetic-2/retained/layer-0.f32": ROWS_B,
                "input-0/normal/projected.f32": ROWS_C}
    write_side(base, "original", original)
    write_side(base, "off", original)
    write_side(base, "on", {"synthetic-2/normal/projected.f32": ROWS_A_ON,
                            "synthetic-2/retained/layer-0.f32": ROWS_B,
                            "input-0/normal/projected.f32": ROWS_C_ON})
    report_path = base / "report.json"
    result = run(base, report=report_path)
    assert result.returncode == 0, result.stderr
    report = strict_json(report_path.read_text())
    assert report["original_vs_off_bit_exact"] is True and report["total_bit_changes"] == 0
    assert report["tensor_count"] == 3 and "passed" not in report and report["on_threshold"] is None
    rows = {row["key"]: row for row in report["off_vs_on"]}
    signal = noise = 0.0  # independent double sums, compare.h style
    for reference, candidate in zip([f32(v) for row in ROWS_A for v in row],
                                    [f32(v) for row in ROWS_A_ON for v in row]):
        signal += reference * reference
        noise += (candidate - reference) ** 2
    a = rows["synthetic-2/normal/projected.f32"]
    assert (a["frames"], a["width"], a["samples"], a["bit_changes"]) == (3, 5, 15, 2)
    assert a["max_abs_error"] == 2.0 and a["max_relative_error_nonzero_reference"] == 2.0
    assert (a["zero_reference_elements"], a["zero_reference_mismatches"]) == (2, 1)
    assert a["nmse"]["status"] == "ok" and math.isclose(a["nmse"]["value"], noise / signal, rel_tol=1e-12)
    assert a["snr_db"]["status"] == "ok"
    assert math.isclose(a["snr_db"]["value"], 10.0 * math.log10(signal / noise), rel_tol=1e-12)
    assert a["argmax_changed_frames"] == 1 and a["argmax_changed_frame_indices"] == [2]
    b = rows["synthetic-2/retained/layer-0.f32"]
    assert b["bit_changes"] == 0 and b["max_abs_error"] == 0.0
    assert b["nmse"] == {"value": 0.0, "status": "ok"}
    assert b["snr_db"] == {"value": None, "status": "perfect_zero_error"}
    assert b["max_relative_error_nonzero_reference"] == 0.0 and b["argmax_changed_frames"] == 0
    c = rows["input-0/normal/projected.f32"]
    assert c["nmse"] == {"value": None, "status": "undefined_all_zero_reference_power"}
    assert c["snr_db"] == {"value": None, "status": "undefined_all_zero_reference_power"}
    assert c["max_relative_error_nonzero_reference"] is None and c["max_abs_error"] == 1.0
    assert c["argmax_changed_frame_indices"] == [0] and c["zero_reference_mismatches"] == 1


def check_signed_zero_gate(base):
    """A single +0.0 versus -0.0 change fails the gate despite perfect OFF/ON metrics."""
    write_side(base, "original", {TINY_KEY: [[1.0, 0.0, 2.0]]})
    write_side(base, "off", {TINY_KEY: [[1.0, -0.0, 2.0]]})
    write_side(base, "on", {TINY_KEY: [[1.0, -0.0, 2.0]]})
    result = run(base)
    assert result.returncode != 0, "+0.0 versus -0.0 must fail the original/OFF gate"
    report = strict_json(result.stdout)
    assert report["original_vs_off_bit_exact"] is False and report["total_bit_changes"] == 1
    assert report["tensors_with_bit_changes"] == [TINY_KEY]
    row = report["off_vs_on"][0]
    assert row["bit_changes"] == 0 and row["nmse"] == {"value": 0.0, "status": "ok"}
    assert row["snr_db"] == {"value": None, "status": "perfect_zero_error"}


def check_relative_records(base):
    """A relative tensor record resolves against the comparator's working directory."""
    for side in SIDES:
        write_side(base, side, {TINY_KEY: TINY_ROWS}, relative=True)
    result = run(base, relative=True)
    assert result.returncode == 0, result.stderr


def truncate_tensor(base):
    tiny(base)
    path = base / "off" / TINY_KEY
    path.write_bytes(path.read_bytes()[:-4])


def nan_on(base):
    tiny(base)
    (base / "on" / TINY_KEY).write_bytes(pack([[float("nan"), 2.0]]))


def inf_off(base):
    tiny(base)
    (base / "off" / TINY_KEY).write_bytes(pack([[float("inf"), 2.0]]))


def out_of_root_path(base):
    tiny(base)
    edit(base, "on", lambda records: records[0].update(tensor=str(base / "elsewhere" / TINY_KEY)))


def drop_tensor(base):
    tensors = {TINY_KEY: TINY_ROWS, "input-0/normal/projected.f32": [[3.0, 4.0]]}
    for side in SIDES:
        write_side(base, side, tensors)
    edit(base, "on", lambda records: records.pop(0))


NEGATIVES = [
    ("missing_completion_marker", "completion", mutated("on", lambda r: r.pop())),
    ("early_completion_marker", "final", mutated("on", lambda r: r.insert(0, dict(FINAL)))),
    ("duplicate_completion_marker", "final", mutated("on", lambda r: r.append(dict(FINAL)))),
    ("mixed_completion_marker", "mix", mutated("on", lambda r: r[-1].update(tensor="mixed"))),
    ("duplicate_tensor_record", "duplicate", mutated("original", lambda r: r.insert(1, dict(r[0])))),
    ("tensor_key_set_mismatch", "key set", drop_tensor),
    ("inconsistent_shape", "dimensions", mutated("on", lambda r: r[0].update(width=4))),
    ("truncated_tensor", "bytes", truncate_tensor),
    ("nan_sample", "not finite", nan_on),
    ("inf_sample", "not finite", inf_off),
    ("argmax_metadata_mismatch", "argmax", mutated("on", lambda r: r[0].update(channel_argmax=[0]))),
    ("out_of_root_tensor_path", "outside", out_of_root_path),
]


def rejection(keyword, setup):
    def case(base):
        setup(base)
        result = run(base)
        assert result.returncode != 0, "expected a nonzero exit, stdout:\n%s" % result.stdout
        assert keyword in result.stderr.lower(), "stderr missing %r: %s" % (keyword, result.stderr.strip())
    return case


def main():
    assert COMPARATOR.exists(), "comparator not found next to this file: %s" % COMPARATOR
    cases = [("passing_metrics", check_passing),
             ("signed_zero_gate", check_signed_zero_gate),
             ("relative_tensor_paths", check_relative_records)]
    cases += [(name, rejection(keyword, setup)) for name, keyword, setup in NEGATIVES]
    failures = []
    with tempfile.TemporaryDirectory(prefix="007b-compare-check-") as temp:
        for name, case in cases:
            base = Path(temp) / name
            base.mkdir()
            try:
                case(base)
                print("PASS %s" % name)
            except AssertionError as error:
                failures.append(name)
                print("FAIL %s: %s" % (name, error))
    if failures:
        print("%d of %d cases failed: %s" % (len(failures), len(cases), ", ".join(failures)))
        return 1
    print("all %d cases passed" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
