"""Checks the native runner's raw projections against the same normal artifact's Python correctness run.

Usage: python check_native.py MANIFEST PYTHON_CHECK_DIR NATIVE_RUN_DIR
This reads arrays and hashes artifacts; run it within the campaign's assigned compute lease.
"""

import argparse
from pathlib import Path

from contract import (load_manifest, read_json, require, sha256, verify_source_snapshot, write_json)


def main():
    import numpy as np
    from numerics import array, difference

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("python_check", type=Path)
    parser.add_argument("native_run", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    verify_source_snapshot(manifest)
    checked = read_json(args.python_check / "check.json")
    native = read_json(args.native_run / "run.json")
    result = read_json(args.native_run / "result.json")
    require(checked["manifest_sha256"] == native["manifest_sha256"] == sha256(args.manifest), "Manifest identities differ.")
    require(checked["compute_units"] == native["compute_units"], "The compute-unit configurations differ.")
    require(native["model"] == manifest["models"][native["precision"]]["normal"], "The native run uses another artifact.")
    # The package the native run compiled at run time, the package the Python check predicted and
    # the export's inventory must be one and the same content; re-hashing the path alone proves
    # nothing after both runs have ended.
    require(native["model_identity"] == checked["model_identities"][native["precision"] + "/normal"],
            "The compiled package differs from the package the Python check predicted.")
    require(native["model_identity"] == manifest["model_identities"][native["precision"]]["normal"],
            "The compiled package differs from the export's inventory.")
    expected_order = []
    for round_number in range(native["rounds"] + 1):
        shift = 0 if round_number == 0 else (round_number * 7) % len(manifest["cases"])
        order = manifest["cases"][shift:] + manifest["cases"][:shift]
        expected_order += [(row["id"], round_number) for row in order]
    require([(row["id"], row["round"]) for row in result["records"]] == expected_order, "Native predictions are incomplete or reordered.")
    rows = []
    for case in manifest["cases"]:
        projected = args.python_check / native["precision"] / "normal" / case["id"] / "joint_encoded.npy"
        raw = args.native_run / (case["id"] + ".joint.f32")
        expected = array(projected, (case["u"], 640))
        actual = np.fromfile(raw, dtype="<f4").reshape(case["u"], 640)
        # The two file hashes are what the TDT admission later binds, so the projection it feeds to
        # the decoder is verifiably the one this parity proved bit-exact.
        rows.append({"id": case["id"], **difference(actual, expected),
                     "python_npy_sha256": sha256(projected), "native_raw_sha256": sha256(raw)})
    # The record carries the manifest's own hash so a later admission binds this parity to that manifest.
    write_json(args.native_run / "output-check.json", {"manifest_sha256": sha256(args.manifest),
               "precision": native["precision"], "compute_units": native["compute_units"], "rows": rows,
               "all_exact": all(row["bit_exact"] for row in rows),
               "scope": "Native/Python normal-artifact parity, not official full KDA/TDT validation."})
    require(all(row["bit_exact"] for row in rows), "Native outputs differ from the Python normal artifact; retain and investigate the raw difference.")


if __name__ == "__main__":
    main()
