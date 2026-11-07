"""Checks Core ML artifacts against the official F32 dumps, keeping isolated and composed errors separate.

Usage: python check.py MANIFEST --units CPU_ONLY|CPU_AND_GPU|CPU_AND_NE|ALL --out NEW_OUTPUT_DIRECTORY
This compiles and executes Core ML models; run only under an assigned campaign compute lease.
"""

import argparse
import gc
from pathlib import Path
import time

from contract import (COMPUTE_UNITS, disk_preflight, load_manifest, model_identity, output_directory, require,
                      sha256, verify_reference_files, verify_source_snapshot, write_json)


def execute(args, destination):
    from export import environment
    manifest = load_manifest(args.manifest)
    verify_source_snapshot(manifest)
    disk = disk_preflight("check", destination)
    current_environment = environment(args.manifest.resolve().parent / "uv.lock")
    require(current_environment == manifest["environment"], "Use the same frozen conversion environment for stage checks.")
    import coremltools as ct
    import numpy as np
    from numerics import array, difference, output_arrays, stage

    units = getattr(ct.ComputeUnit, args.units)
    records, identities, saved = [], {}, {}
    write_json(destination / "run.json", {"manifest_sha256": sha256(args.manifest), "compute_units": args.units,
               "environment": current_environment, "disk_preflight": disk,
               "checker_sha256": sha256(__file__), "source_snapshot_verified": True})
    for precision in ("f32", "f16"):
        for kind in ("normal", "diagnostic", "joint"):
            path = manifest["models"][precision][kind]
            key = precision + "/" + kind
            # Identity is hashed and checked outside every prediction timer: the package must be
            # the exact one the export recorded, whatever path it now sits at.
            identity = model_identity(path)
            require(identity == manifest["model_identities"][precision][kind],
                    "The converted package differs from the export's inventory: " + key)
            identities[key] = identity
            model = ct.models.MLModel(path, compute_units=units)
            for case in manifest["cases"]:
                directory = Path(case["reference_dir"])
                verify_reference_files(case, directory)
                want_encoded = array(directory / "encoded.npy", (case["u"], 1024))
                want_joint = array(directory / "joint_encoded.npy", (case["u"], 640))
                raw_key = "encoded" if kind == "joint" else "features"
                raw = case[raw_key + "_raw"]
                require(sha256(raw) == case[raw_key + "_sha256"], "Prepared input changed.")
                shape = (1, case["u"], 1024) if kind == "joint" else (1, 128, case["p"])
                data = np.fromfile(raw, dtype="<f4").reshape(shape)
                inputs = {"encoded_input": data} if kind == "joint" else {
                    "features": data, "length": np.array([case["t"]], dtype=np.int32)}
                start = time.perf_counter()
                output = model.predict(inputs)
                elapsed = time.perf_counter() - start
                out = destination / precision / kind / case["id"]
                out.mkdir(parents=True)
                # Raw padded outputs remain available even when the shape or finite-value check fails.
                for name, value in output.items():
                    np.save(out / (name + ".raw.npy"), value)
                actual = output_arrays(output, case, diagnostic=kind == "diagnostic", joint=kind == "joint")
                for name, value in actual.items():
                    np.save(out / (name + ".npy"), value)
                record = {"id": case["id"], "precision": precision, "kind": kind, "compute_units": args.units,
                          "python_prediction_seconds": elapsed, "timing_scope": "Correctness run, not benchmark evidence",
                          "joint_composed" if kind != "joint" else "joint_isolated": difference(actual["joint_encoded"], want_joint)}
                write_json(out / "raw-metrics.json", record)
                if kind == "joint":
                    stage(actual["joint_encoded"], want_joint, 90 if precision == "f32" else 40,
                          "isolated joint " + case["id"])
                elif precision == "f32":
                    stage(actual["joint_encoded"], want_joint, 90, "F32 composed joint " + case["id"])
                if kind == "diagnostic":
                    record["encoder"] = stage(actual["encoded"], want_encoded, 90 if precision == "f32" else 25,
                                              "encoder " + case["id"])
                    record["normal_vs_diagnostic"] = difference(actual["joint_encoded"], saved[(precision, case["id"])])
                if kind == "normal":
                    saved[(precision, case["id"])] = actual["joint_encoded"].copy()
                records.append(record)
                write_json(out / "metrics.json", record)
            del model
            gc.collect()
    write_json(destination / "check.json", {"schema": 1, "manifest_sha256": sha256(args.manifest),
               "compute_units": args.units, "model_identities": identities, "records": records,
               "isolated_stage_screens_passed": True,
               "f16_composed_acceptance": "Requires review of raw composed error and full TDT decisions; no bound is inferred from isolated joint.",
               "normal_diagnostic_acceptance": "Inspect each normal_vs_diagnostic result; diagnostic placement is not normal placement.",
               "official_scope": "encoder and joint.enc only", "full_tdt_checked": False,
               "actual_ane_execution": "Not established by the compute-unit option or this check."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--units", choices=COMPUTE_UNITS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    destination = output_directory(args.out)
    try:
        execute(args, destination)
    except Exception as error:
        write_json(destination / "failure.json", {"stage": "check", "type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
