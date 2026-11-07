"""Validates the independent experiment's identities and shape contract without importing ML packages."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess


HERE = Path(__file__).resolve().parent
PINS = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
COMPUTE_UNITS = ("CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL")

GIB = 1024 ** 3

# Planning estimates in GiB for what one invocation of a step still adds to the volume it runs on,
# compared with that volume's live free space immediately before the invocation. They are planning
# constants, not measurements and not guarantees: package sizes are not pinned exactly and the
# converted models' sizes follow from the shapes converted, so they are deliberately high, and the
# reserve covers the system and the smaller writes around each step. The pinned archive,
# 1232384000 bytes, and its reference dumps are already on the disk before the environment step; no
# step budgets them again.
DISK_PLAN = {
    # The converter environment and its download cache.
    "environment": {"environment": 8, "cache": 4, "reserve": 8},
    # Six .mlpackage directories: the normal and diagnostic encoders in F32 and F16 carry the
    # weights and the isolated joint in each precision is far smaller, plus the temporary work of
    # restoring the archive and tracing and converting, plus the run's raw inputs and records.
    "export": {"packages": 8, "conversion_temp": 4, "run_data": 1, "reserve": 8},
    # A check invocation compiles every package it loads into a system temporary directory and
    # writes the raw padded outputs and records of all cases, kinds and precisions.
    "check": {"compiled_temp": 2, "check_outputs": 1, "reserve": 8},
    # A native invocation compiles the package to a temporary copy and retains one compiled copy in
    # its result directory, so the possible two-copy peak is budgeted; plan, predict and trace are
    # separate invocations that each preflight this row again.
    "native": {"compiled_temp": 2, "compiled_kept": 2, "native_outputs": 1, "reserve": 8},
    # An actual Instruments capture, collected in a separate later step from the runner's signpost
    # mode and retained only when collected.
    "instruments": {"instruments_trace": 4, "reserve": 8},
}

# The sources a later gate must still match at the export's snapshot: the gate checkers, every
# helper they import, and the pins.
CHECKER_SOURCES = ("check.py", "check_native.py", "check_tdt.py", "contract.py", "export.py",
                   "numerics.py", "pins.json")

# The dump files whose content the export binds per case, hashed as they were used.
REFERENCE_FILES = ("meta.json", "features.npy", "encoded.npy", "joint_encoded.npy", "audio.npy")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def output_directory(path):
    """Creates a new artifact directory under an outputs component, never in the tracked experiment source."""
    path = Path(path).resolve()
    require("outputs" in path.parts and not path.is_relative_to(HERE), "Artifacts must be outside source, under outputs/.")
    path.mkdir(parents=True, exist_ok=False)
    return path


def disk_decision(stage, free_bytes):
    """Compares one invocation's planning estimate with the free space a volume reports."""
    require(stage in DISK_PLAN, "Unknown disk plan step: " + stage)
    planning = {name: gib * GIB for name, gib in DISK_PLAN[stage].items()}
    total = sum(planning.values())
    require(free_bytes >= total, "The disk plan for " + stage + " needs " + str(total) + " bytes ("
            + str(DISK_PLAN[stage]) + " GiB each) but the volume reports " + str(free_bytes)
            + " free; free space on the volume or lower the plan's assumptions.")
    return {"step": stage, "planning_gib": DISK_PLAN[stage], "planning_bytes": planning, "total_bytes": total,
            "free_bytes": free_bytes}


def disk_preflight(stage, path):
    """Runs disk_decision() on the volume holding an existing directory, before the invocation starts."""
    require(Path(path).exists(), "Preflight an existing directory: " + str(path))
    return disk_decision(stage, shutil.disk_usage(Path(path)).free)


def model_identity(path):
    """Streams every file of a package directory into a relative-path to SHA-256 inventory."""
    path = Path(path)
    require(path.is_dir(), "Not a package directory: " + str(path))
    identity = {str(item.relative_to(path)): sha256(item) for item in sorted(path.rglob("*")) if item.is_file()}
    require(identity, "The package holds no files: " + str(path))
    return identity


def verify_source_snapshot(manifest, directory=HERE):
    """Binds a later gate to the export's snapshot of the checker sources and the helpers they import."""
    snapshot = manifest["source_sha256"]
    require(set(CHECKER_SOURCES) <= set(snapshot), "The export's source snapshot lacks a checker source.")
    for name in sorted(snapshot):
        current = Path(directory) / name
        require(current.is_file() and sha256(current) == snapshot[name],
                "The source differs from the export's snapshot: " + name)


def verify_reference_files(case, directory):
    """Re-hashes the exact dump files the export used before a later gate trusts their arrays."""
    directory = Path(directory)
    for name in REFERENCE_FILES:
        require((directory / name).is_file() and sha256(directory / name) == case["reference_files"][name],
                "The reference file differs from the one the export used: " + name)


def selected_cases(shape_set):
    require(shape_set in ("fixed", "enumerated"), "Choose fixed or enumerated shapes.")
    return [dict(row) for row in PINS["cases"] if shape_set == "enumerated" or row["p"] == 1008]


def validate_meta(case, meta):
    """Keeps the reference's valid lengths separate from the padded tensor dimensions."""
    require(meta["reference_scope"] == "frontend-encoder-joint.enc", "An encoder-only official dump is required.")
    require(meta["model"] == PINS["checkpoint"], "The dump checkpoint differs from the pin.")
    require(meta["dtype"] == "float32" and meta["device"] == "cpu", "The reference must be CPU F32.")
    require(meta["nemo"] == PINS["converter_candidate"]["nemo-toolkit"], "The official NeMo version differs.")
    observed = (meta["frames"], meta["encoded_frames"], meta["encoder_input_frames"], meta["encoder_output_frames"])
    require(observed == tuple(case[key] for key in ("t", "u", "p", "q")), "Valid or padded lengths differ from the case pin.")
    require(0 < case["t"] <= case["p"] and 0 < case["u"] <= case["q"], "Invalid frame interval.")
    require(meta["sample_rate"] == 16000 and meta["samples"] > 0, "Invalid reference audio extent.")
    for name, shape in {"features": [case["t"], 128], "encoded": [case["u"], 1024],
                        "joint_encoded": [case["u"], 640]}.items():
        require(meta["arrays"][name] == shape, "Invalid reference shape: " + name)


def reference_source(path):
    path = Path(path).resolve()
    commit = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    require(commit == PINS["reference_commit"], "Use the pinned reference checkout.")
    subprocess.run(["git", "-C", str(path), "diff", "--exit-code", "HEAD", "--", "reference/fastconformer"], check=True)
    return path / "reference" / "fastconformer"


def load_manifest(path):
    manifest = read_json(path)
    require(manifest["schema"] == 1 and manifest["experiment"] == "034", "Unknown experiment manifest.")
    require(manifest["pins"] == PINS, "The manifest's source or checkpoint pins differ.")
    expected = selected_cases(manifest["shape_set"])
    actual = [{key: row[key] for key in ("id", "t", "u", "p", "q")} for row in manifest["cases"]]
    require(actual == expected, "Missing, duplicate, reordered, or altered experiment cases.")
    return manifest
