#!/usr/bin/env python3
"""Compare three 007b-encoder-check runs: original, candidate OFF and candidate ON.

Each run is one output root plus the JSON Lines report the emitter printed:
tensor records ({"tensor": path, "frames": F, "width": W, "channel_argmax":
[...]}), mask and execution records, and a single {"passed": true} completion
marker that must be the final nonempty object in the file.  A relative tensor
path is exactly what the emitter printed, so it resolves against the working
directory and must still fall inside that side's declared root.

original versus OFF must stay bit-exact over the little-endian float32
patterns (+0.0 versus -0.0 counts); any mismatch exits nonzero.  OFF versus
ON is report-only double-precision energy metrics: no universal threshold
exists for these raw tensors, and the flash-attention NMSE<5e-4, official CPU
F32 SNR>90dB and Metal F32/F16 SNR>25dB bounds are not applied here.

Exit codes: 0 on success, 1 on invalid input or a bit mismatch, 2 on usage.
"""

import argparse
import array
import json
import math
import sys
from pathlib import Path

SIDES = ("original", "off", "on")


class CompareError(Exception):
    """A validation failure that must abort the comparison with a nonzero status."""


def tensor_key(record, root, where):
    """Return the recorded tensor path as its key relative to the side root."""
    recorded = record.get("tensor")
    if not isinstance(recorded, str) or not recorded:
        raise CompareError("%s: tensor path must be a nonempty string, got %r" % (where, recorded))
    path = Path(recorded)
    if not path.is_absolute():
        path = Path.cwd() / path
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        raise CompareError("%s: tensor path %s resolves to %s, outside the declared root %s; "
                           "point the emitter at a directory inside the root" % (where, recorded, path, root))


def tensor_record(record, tensors, root, where):
    """Validate one tensor record and store it under its relative key."""
    frames = record.get("frames")
    width = record.get("width")
    for value, what in ((frames, "frames"), (width, "width")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise CompareError("%s: %s must be a positive integer, got %r" % (where, what, value))
    argmax = record.get("channel_argmax")
    if not isinstance(argmax, list):
        raise CompareError("%s: channel_argmax must be a list, got %r" % (where, argmax))
    if len(argmax) != frames:
        raise CompareError("%s: channel_argmax has %d entries but frames is %d" % (where, len(argmax), frames))
    for frame, channel in enumerate(argmax):
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < width:
            raise CompareError("%s: channel_argmax[%d]=%r is not an integer channel in [0, %d)"
                               % (where, frame, channel, width))
    key = tensor_key(record, root, where)
    if key in tensors:
        raise CompareError("%s: duplicate tensor record for key %s" % (where, key))
    tensors[key] = {"frames": frames, "width": width, "argmax": argmax}


def parse_side(side, root, report_path):
    """Read one report and return its tensor records keyed by relative path."""
    try:
        lines = Path(report_path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise CompareError("side %s: cannot read report %s (%s)" % (side, report_path, error))
    records = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        where = "side %s, line %d of %s" % (side, number, report_path)
        try:
            record = json.loads(line)
        except ValueError as error:
            raise CompareError("%s: not valid JSON (%s)" % (where, error))
        if not isinstance(record, dict):
            raise CompareError("%s: record is not a JSON object" % where)
        records.append((where, record))
    if not records:
        raise CompareError("side %s: report %s is empty" % (side, report_path))
    tensors = {}
    completed = False
    for index, (where, record) in enumerate(records):
        if "passed" in record:
            mixed = sorted(field for field in ("tensor", "frames", "width", "channel_argmax",
                                               "retained", "mask_layer") if field in record)
            if mixed:
                raise CompareError("%s: completion record must not mix tensor or run fields %s" % (where, mixed))
            if record["passed"] is not True:
                raise CompareError("%s: completion record is not passed:true (%r)" % (where, record["passed"]))
            if index != len(records) - 1:
                raise CompareError("%s: completion marker must be the final nonempty record; early, "
                                   "repeated or appended markers are rejected" % where)
            completed = True
        elif "tensor" in record:
            tensor_record(record, tensors, root, where)
        elif not set(record) & {"retained", "mask_layer"}:
            # Execution ("retained") and mask ("mask_layer") records are context only.
            raise CompareError("%s: unrecognized record with fields %s" % (where, sorted(record)))
    if not completed:
        raise CompareError("side %s: report %s ends without its final passed:true completion record"
                           % (side, report_path))
    if not tensors:
        raise CompareError("side %s: report %s contains no tensor records" % (side, report_path))
    return tensors


def read_tensor(side, root, key, frames, width):
    """Read one tensor file as explicitly little-endian float32 values and bit patterns."""
    path = root.joinpath(*key.split("/"))
    try:
        data = path.read_bytes()
    except OSError as error:
        raise CompareError("side %s: cannot read tensor file %s (%s)" % (side, path, error))
    expected = 4 * frames * width
    if len(data) != expected:
        raise CompareError("side %s: tensor %s has %d bytes but frames %d width %d need exactly %d"
                           % (side, path, len(data), frames, width, expected))
    values = array.array("f", data)
    bits = array.array("I", data)
    if sys.byteorder != "little":
        values.byteswap()
        bits.byteswap()
    for index, value in enumerate(values):
        if not math.isfinite(value):
            raise CompareError("side %s: tensor %s sample %d is not finite (%r)" % (side, key, index, value))
    return values, bits


def check_argmax(side, key, declared, values, frames, width):
    """Reject metadata whose declared argmax differs from the first per-frame sample maximum."""
    for frame in range(frames):
        row = values[frame * width:(frame + 1) * width]
        if declared[frame] != row.index(max(row)):
            raise CompareError("side %s: tensor %s channel_argmax[%d] is %d but the samples first "
                               "maximize at %d" % (side, key, frame, declared[frame], row.index(max(row))))


def nmse_snr(signal, noise):
    """NMSE and SNR from double-precision powers, null where undefined."""
    if signal == 0.0:
        return ({"value": None, "status": "undefined_all_zero_reference_power"},
                {"value": None, "status": "undefined_all_zero_reference_power"})
    if noise == 0.0:
        return {"value": 0.0, "status": "ok"}, {"value": None, "status": "perfect_zero_error"}
    return {"value": noise / signal, "status": "ok"}, {"value": 10.0 * math.log10(signal / noise), "status": "ok"}


def off_on_row(key, frames, width, off, on, argmax_changed):
    """Report-only OFF versus ON metrics for one tensor, compare.h-style double sums."""
    off_values, off_bits = off
    on_values, on_bits = on
    signal = noise = max_abs = 0.0
    max_rel = None
    zero_refs = zero_mismatches = 0
    for reference, candidate in zip(off_values, on_values):
        error = candidate - reference
        magnitude = abs(error)
        if magnitude > max_abs:
            max_abs = magnitude
        signal += reference * reference
        noise += error * error
        if reference == 0.0:
            zero_refs += 1
            if error != 0.0:
                zero_mismatches += 1
        elif max_rel is None or magnitude / abs(reference) > max_rel:
            max_rel = magnitude / abs(reference)
    nmse, snr_db = nmse_snr(signal, noise)
    return {
        "key": key,
        "frames": frames,
        "width": width,
        "samples": frames * width,
        "bit_changes": sum(a != b for a, b in zip(off_bits, on_bits)),
        "max_abs_error": max_abs,
        "max_relative_error_nonzero_reference": max_rel,
        "zero_reference_elements": zero_refs,
        "zero_reference_mismatches": zero_mismatches,
        "nmse": nmse,
        "snr_db": snr_db,
        "argmax_changed_frames": len(argmax_changed),
        "argmax_changed_frame_indices": argmax_changed,
    }


def compare_all(sides):
    """Compare every agreed tensor and build the machine-readable report."""
    keys = set(sides["off"]["tensors"])
    for side in ("original", "on"):
        missing = sorted(keys - set(sides[side]["tensors"]))
        extra = sorted(set(sides[side]["tensors"]) - keys)
        if missing or extra:
            raise CompareError("side %s tensor key set differs from off: missing %s, extra %s"
                               % (side, missing, extra))
    rows = []
    changed_keys = []
    total_bit_changes = 0
    for key in sorted(keys):
        meta = {side: sides[side]["tensors"][key] for side in SIDES}
        frames = meta["original"]["frames"]
        width = meta["original"]["width"]
        for side in ("off", "on"):
            if (meta[side]["frames"], meta[side]["width"]) != (frames, width):
                raise CompareError("sides disagree on dimensions for tensor %s: original frames %d width %d, "
                                   "%s declares frames %d width %d"
                                   % (key, frames, width, side, meta[side]["frames"], meta[side]["width"]))
        # One tensor triplet is held at a time and dropped before the next key.
        data = {side: read_tensor(side, sides[side]["root"], key, frames, width) for side in SIDES}
        for side in SIDES:
            check_argmax(side, key, meta[side]["argmax"], data[side][0], frames, width)
        bit_changes = sum(a != b for a, b in zip(data["original"][1], data["off"][1]))
        total_bit_changes += bit_changes
        if bit_changes:
            changed_keys.append(key)
        argmax_changed = [frame for frame in range(frames)
                          if meta["off"]["argmax"][frame] != meta["on"]["argmax"][frame]]
        rows.append(off_on_row(key, frames, width, data["off"], data["on"], argmax_changed))
    return {
        "tensor_count": len(rows),
        "original_vs_off_bit_exact": not changed_keys,
        "tensors_with_bit_changes": changed_keys,
        "total_bit_changes": total_bit_changes,
        "off_vs_on": rows,
        "on_threshold": None,
        "note": "OFF versus ON is report-only: the FA scalar NMSE<5e-4, official CPU F32 SNR>90dB and "
                "Metal F32/F16 SNR>25dB bounds are not applied to these raw tensors",
    }


def main(argv):
    parser = argparse.ArgumentParser(
        prog="007b-compare-encoder",
        description="Compare original, candidate OFF and candidate ON 007b-encoder-check output roots.")
    for side in SIDES:
        parser.add_argument("--%s-dir" % side, required=True, help="output root directory of the %s run" % side)
        parser.add_argument("--%s-jsonl" % side, required=True, help="JSON Lines report of the %s run" % side)
    parser.add_argument("--report", default="-", help="path for the JSON report, or '-' for stdout")
    args = parser.parse_args(argv)
    sides = {}
    for side in SIDES:
        root = Path(getattr(args, side + "_dir")).resolve()
        sides[side] = {"root": root, "tensors": parse_side(side, root, Path(getattr(args, side + "_jsonl")))}
    report = compare_all(sides)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.report == "-":
        sys.stdout.write(text)
    else:
        Path(args.report).write_text(text, encoding="utf-8")
    sys.stderr.write("tensors: %d, original/OFF bit-exact: %s\n"
                     % (report["tensor_count"], report["original_vs_off_bit_exact"]))
    return 0 if report["original_vs_off_bit_exact"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except (CompareError, OSError) as error:
        sys.stderr.write("error: %s\n" % error)
        sys.exit(1)
