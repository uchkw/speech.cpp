"""Checks mixed full and encoder-only FastConformer dump discovery and explicit stage selection.

usage: python3 checks/smoke/fastconformer_reference_smoke.py <build> <F32.gguf> <encoder dump case> <new work dir> [--list-only]

The fixture links one official encoder case and writes synthetic producer metadata for intentionally incomplete full
dumps. It never treats those fixtures as full-model reference outputs. --list-only checks the runner's planning without
opening model tensors or running any C++ stage. A full run also checks projection inputs and the full-dump diagnostics
on the CPU; it does not execute a prediction network or decode a transcript. The work directory retains every log.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("case", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    meta = json.loads((args.case / "meta.json").read_text())
    assert meta["reference_scope"] == "frontend-encoder-joint.enc"
    name = meta["model"]["repository"].rsplit("/", 1)[-1]
    full = {"prediction_steps": 0, "joint_evaluations": 0, "text": ""}
    runner = Path(__file__).resolve().parents[1] / "run.py"
    roots = {}

    def fixture(kind, cases):
        reference = args.work / kind / "reference"
        out = reference / "fastconformer" / "out"
        for case_name, metadata, missing in cases:
            folder = out / name / case_name
            folder.mkdir(parents=True)
            (folder / "meta.json").write_text(json.dumps(metadata) + "\n")
            for path in args.case.glob("*.npy"):
                if path.name not in missing:
                    (folder / path.name).symlink_to(path.resolve())
        roots[kind] = (reference, out)

    fixture("mixed", [("encoder-case", meta, set()), ("full-case", full, {"joint_encoded.npy"})])
    fixture("encoder", [("encoder-case", meta, set())])
    fixture("full", [("full-case", full, {"joint_encoded.npy"})])
    fixture("missing-projection", [("encoder-case", meta, {"joint_encoded.npy"})])
    fixture("unknown", [("unknown-case", {"reference_scope": "unknown"}, set())])
    fixture("invalid-full", [("invalid-case", {**full, "prediction_steps": "invalid"}, set())])
    fixture("empty", [])
    results = []

    def run(label, command, code, required, forbidden=()):
        result = subprocess.run([str(part) for part in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.work / (label + ".log")).write_text(result.stdout)
        assert result.returncode == code, f"{label}: exit {result.returncode}, expected {code}: {result.stdout}"
        assert all(text in result.stdout for text in required), f"{label}: missing {required}: {result.stdout}"
        assert not any(text in result.stdout for text in forbidden), f"{label}: unexpected {forbidden}: {result.stdout}"
        results.append({"case": label, "returncode": result.returncode})
        print("pass", label, flush=True)

    def plan(kind, stage, code=0, required=()):
        check = "fastconformer-" + stage + "-check"
        command = [sys.executable, runner, "fastconformer", "--models", args.model, "--reference", roots[kind][0],
                   "--devices", "cpu", "--only", check, "--list", "--logs", args.work / "runner-logs"]
        run("list-" + kind + "-" + stage, command, code, required or [check + " "], ["skip "] if code == 0 else [])

    plan("mixed", "joint-encoder")
    plan("mixed", "transducer")
    plan("encoder", "joint-encoder")
    plan("missing-projection", "joint-encoder")
    plan("full", "joint-encoder", 1, ["requested stage", "no compatible frontend-encoder-joint.enc"])
    plan("encoder", "transducer", 1, ["requested stage", "no compatible full"])
    plan("empty", "joint-encoder", 1, ["requested stage", "no compatible"])
    plan("unknown", "joint-encoder", 1, ["unknown FastConformer reference_scope"])
    plan("invalid-full", "transducer", 1, ["no recognized FastConformer dump producer metadata"])
    run("list-encoder-all", [sys.executable, runner, "fastconformer", "--models", args.model,
                            "--reference", roots["encoder"][0], "--devices", "cpu", "--list",
                            "--logs", args.work / "runner-logs"], 0,
        ["fastconformer-joint-encoder-check ", "skip fastconformer-transducer-check", "skip fastconformer-times-check",
         "no compatible full"], ["skip fastconformer-joint-encoder-check"])
    if not args.list_only:
        suffix = ".exe" if sys.platform == "win32" else ""

        def stage(kind, stage_name, code, required, forbidden=()):
            check = args.build / ("fastconformer-" + stage_name + "-check" + suffix)
            run("stage-" + kind + "-" + stage_name, [check, args.model, roots[kind][1], "cpu"], code, required, forbidden)

        stage("mixed", "joint-encoder", 0, ["encoder-case", "encoder projection"], ["full-case"])
        stage("missing-projection", "joint-encoder", 1, ["joint_encoded.npy"])
        stage("mixed", "transducer", 1, ["pred_labels.npy"])
        stage("encoder", "transducer", 1, ["no compatible full", "encoder-only"])
        stage("encoder", "times", 1, ["no compatible full", "encoder-only"])
        stage("full", "joint-encoder", 1, ["no compatible frontend-encoder-joint.enc"])
        stage("unknown", "joint-encoder", 1, ["unknown FastConformer reference_scope"])
        stage("invalid-full", "transducer", 1, ["no recognized FastConformer dump producer metadata"])
    (args.work / "results.json").write_text(json.dumps({"list_only": args.list_only, "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
