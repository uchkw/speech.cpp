"""Freezes the TDT harness's build receipt and admits the checked candidate projection to the harness.

Usage: python check_tdt.py freeze --worktree WORKTREE --executable HARNESS --receipt NEW_RECEIPT_JSON
       python check_tdt.py admit --worktree WORKTREE --receipt RECEIPT_JSON --manifest MANIFEST
                                --python-check DIR --native-run DIR --executable HARNESS
                                --model MODEL.gguf [--device cpu|gpu] --out NEW_OUTPUT_DIRECTORY
Freeze runs once at build time; admit runs under the campaign lease and is the only door through
which the recipe reaches the unchanged C++ harness. Standard library only.
"""

import argparse
import json
from pathlib import Path
import subprocess

from contract import (COMPUTE_UNITS, HERE, PINS, load_manifest, output_directory, read_json, require,
                      sha256, verify_reference_files, verify_source_snapshot, write_json)

# The tracked runtime and build inputs the harness build consumes: the root CMake project, its
# top-level CMake and version inputs, the cmake modules, the runtime sources with their public
# headers, and the checks it compiles against. A pathspec that names nothing in either tree
# contributes no difference, and the experiment's own sources stay outside the set, so a
# descendant that changed nothing here is still allowed through.
RUNTIME_PATHS = ("CMakeLists.txt", "VERSION", "CMakePresets.json", "cmake", "src", "include", "checks")
HARNESS_SOURCES = ("CMakeLists.txt", "tdt-check.cpp")
GGUF_SHA256_LENGTH = 64


def git(worktree, *arguments):
    return subprocess.run(["git", "-C", str(worktree), *arguments], check=True, capture_output=True,
                          text=True).stdout.strip()


def git_ok(worktree, *arguments):
    return subprocess.run(["git", "-C", str(worktree), *arguments], capture_output=True).returncode == 0


def write_text(path, value):
    """Writes one fresh evidence file; the exclusive open keeps every record new."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(value)


def runtime_state(worktree):
    """Collects the runtime identity of the worktree the harness is built from."""
    worktree = Path(worktree).resolve()
    require((worktree / "CMakeLists.txt").is_file() and (worktree / "src").is_dir(),
            "Not a speech.cpp worktree: " + str(worktree))
    return {"head": git(worktree, "rev-parse", "HEAD"),
            "base_is_ancestor": git_ok(worktree, "merge-base", "--is-ancestor", PINS["runtime_base"], "HEAD"),
            "runtime_clean": git_ok(worktree, "diff", "--exit-code", "HEAD", "--", *RUNTIME_PATHS),
            "runtime_at_base": git_ok(worktree, "diff", "--quiet", PINS["runtime_base"], "HEAD", "--", *RUNTIME_PATHS),
            "ggml_gitlink": git(worktree, "ls-tree", "HEAD", "ggml").split()[2],
            "ggml_working": git(worktree, "-C", "ggml", "rev-parse", "HEAD"),
            "ggml_clean": git_ok(worktree, "-C", "ggml", "diff", "--exit-code", "HEAD")}


def verify_runtime_state(state, receipt=None):
    """Checks a collected runtime state against the pins and, at admission, the frozen receipt."""
    require(state["base_is_ancestor"],
            "The worktree's HEAD is not a descendant of the pinned runtime base " + PINS["runtime_base"] + ".")
    require(state["runtime_clean"], "The runtime and CMake inputs carry uncommitted changes.")
    require(state["runtime_at_base"],
            "The tracked runtime and build inputs at HEAD differ from the pinned runtime base "
            + PINS["runtime_base"] + "; only changes outside them, such as this experiment's own sources, may descend from it.")
    require(state["ggml_gitlink"] == PINS["ggml_base"], "The ggml gitlink differs from the pin.")
    require(state["ggml_working"] == state["ggml_gitlink"] and state["ggml_clean"],
            "The ggml working tree is not clean at the gitlink.")
    if receipt is not None:
        for key in ("head", "ggml_gitlink", "ggml_working"):
            require(state[key] == receipt["runtime"][key], "The runtime moved since the receipt was frozen: " + key)


def validate_admission(manifest, checked, native, parity, manifest_sha256, model_sha256, pinned_gguf_sha256):
    """Validates the records the TDT comparison is admitted on; pure, so fixtures can exercise it."""
    require(manifest_sha256 == checked["manifest_sha256"] == native["manifest_sha256"],
            "The Python check or the native run used a different manifest than the one supplied.")
    require(parity.get("manifest_sha256") == manifest_sha256,
            "The parity record belongs to a different manifest than the one supplied.")
    require(native["compute_units"] in COMPUTE_UNITS, "Unknown compute-unit provenance.")
    require(checked["compute_units"] == native["compute_units"], "The compute-unit provenance differs.")
    require(native["precision"] in ("f32", "f16"), "Unknown precision provenance.")
    require(native["mode"] == "predict", "Parity must bind a prediction run.")
    require(parity["all_exact"] is True and parity["precision"] == native["precision"] and
            parity["compute_units"] == native["compute_units"],
            "A completed successful parity record for this configuration is required.")
    require(native["model_identity"] == checked["model_identities"][native["precision"] + "/normal"],
            "The compiled package differs from the package the Python check predicted.")
    require(native["model_identity"] == manifest["model_identities"][native["precision"]]["normal"],
            "The compiled package differs from the export's inventory.")
    require(isinstance(pinned_gguf_sha256, str) and len(pinned_gguf_sha256) == GGUF_SHA256_LENGTH,
            "The expected GGUF SHA-256 is missing from pins.json's tdt_model pin.")
    require(model_sha256 == pinned_gguf_sha256, "The model file differs from the pinned GGUF SHA-256.")
    expected = [row["id"] for row in manifest["cases"]]
    require([row["id"] for row in parity["rows"]] == expected,
            "The parity record's cases are missing, duplicated, swapped or incomplete.")
    require(all(row["bit_exact"] for row in parity["rows"]), "The parity record is not bit-exact for every case.")
    return {"precision": native["precision"], "compute_units": native["compute_units"], "cases": expected}


def verify_coverage(stdout, pairs):
    """Requires the harness to have decoded both sides and reported one true decision per pair."""
    lines = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    decodings = [row for row in lines if "source" in row]
    decisions = [row for row in lines if "exact_tdt_decisions" in row]
    require(len(decodings) == 2 * len(pairs) and len(decisions) == len(pairs),
            "The harness output does not cover every pair on both sides.")
    for index, (features, projected) in enumerate(pairs):
        pair = decodings[2 * index: 2 * index + 2]
        require([row["source"] for row in pair] == ["ggml_normal_encoder", "coreml_normal_encoder"] and
                pair[0]["input"] == str(features) and pair[1]["input"] == str(projected),
                "The harness decoded another input or order for pair " + str(index))
        require(decisions[index]["exact_tdt_decisions"] is True and decisions[index]["input"] == str(features),
                "A TDT decision is missing, false or reordered for pair " + str(index))


def freeze(args):
    for name in HARNESS_SOURCES:
        require((HERE / name).is_file(), "Missing harness source: " + name)
    require(Path(args.executable).is_file(), "Build the harness before freezing its receipt.")
    state = runtime_state(args.worktree)
    verify_runtime_state(state)
    write_json(args.receipt, {"schema": 1, "experiment": "034", "executable_sha256": sha256(args.executable),
                              "harness_sources": {name: sha256(HERE / name) for name in HARNESS_SOURCES},
                              "runtime": state})


def admit(args, destination):
    manifest = load_manifest(args.manifest)
    verify_source_snapshot(manifest)
    receipt = read_json(args.receipt)
    require(receipt.get("schema") == 1 and receipt.get("experiment") == "034", "Unknown build receipt.")
    verify_runtime_state(runtime_state(args.worktree), receipt)
    require(sha256(args.executable) == receipt["executable_sha256"],
            "The executable differs from the frozen build receipt.")
    for name, recorded in receipt["harness_sources"].items():
        require(sha256(HERE / name) == recorded, "The harness source differs from the receipt: " + name)
    checked = read_json(Path(args.python_check) / "check.json")
    native = read_json(Path(args.native_run) / "run.json")
    parity = read_json(Path(args.native_run) / "output-check.json")
    provenance = validate_admission(manifest, checked, native, parity, sha256(args.manifest), sha256(args.model),
                                    PINS["tdt_model"]["sha256"])
    pairs = []
    for case, row in zip(manifest["cases"], parity["rows"]):
        verify_reference_files(case, case["reference_dir"])
        # The candidate is the exact Python-normal NPY whose file hash the successful parity record
        # bound to the checked native raw projection; the pairs come from these validated records
        # alone, so nothing on the command line can substitute for them.
        projected = Path(args.python_check) / native["precision"] / "normal" / case["id"] / "joint_encoded.npy"
        require(sha256(projected) == row["python_npy_sha256"],
                "The projection file differs from the one parity bound: " + case["id"])
        require(sha256(Path(args.native_run) / (case["id"] + ".joint.f32")) == row["native_raw_sha256"],
                "The native raw projection differs from the parity record: " + case["id"])
        pairs.append((Path(case["reference_dir"]) / "features.npy", projected))
    command = [str(Path(args.executable).resolve()), str(Path(args.model).resolve()), args.device]
    command += [str(path) for pair in pairs for path in pair]
    completed = subprocess.run(command, capture_output=True, text=True)
    # The harness prints its token/frame/duration/progress evidence to stdout even when it returns
    # failure, so the raw streams and the exit metadata are kept in the fresh result directory
    # before either assertion runs; a failed comparison therefore keeps its evidence, and the
    # admission record never needs a second copy of what the raw logs already hold.
    write_text(destination / "harness-stdout.log", completed.stdout)
    write_text(destination / "harness-stderr.log", completed.stderr)
    write_json(destination / "harness-run.json", {"command": command, "exit_code": completed.returncode})
    require(completed.returncode == 0, "The TDT harness rejected the comparison: " + completed.stderr.strip())
    verify_coverage(completed.stdout, pairs)
    write_json(destination / "admission.json", {"schema": 1, "manifest_sha256": sha256(args.manifest),
               "receipt_sha256": sha256(args.receipt), "executable_sha256": receipt["executable_sha256"],
               "model_sha256": sha256(args.model), "device": args.device, "command": command,
               "provenance": provenance,
               "evidence": {"run": "harness-run.json", "stdout": "harness-stdout.log",
                            "stderr": "harness-stderr.log"},
               "scope": "Unboosted CPU-4 core TDT admission; not worker E2E or official Mac KDA validation."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze", help="freeze the build receipt at build time")
    for name in ("worktree", "executable"):
        freezing.add_argument("--" + name, required=True, type=Path)
    freezing.add_argument("--receipt", required=True, type=Path)
    admitting = commands.add_parser("admit", help="verify every record and run the harness")
    for name in ("worktree", "receipt", "manifest", "python-check", "native-run", "executable", "model", "out"):
        admitting.add_argument("--" + name, required=True, type=Path)
    admitting.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args)
        return
    destination = output_directory(args.out)
    try:
        admit(args, destination)
    except Exception as error:
        write_json(destination / "failure.json", {"stage": "tdt-admission", "type": type(error).__name__,
                                                  "error": str(error)})
        raise


if __name__ == "__main__":
    main()
