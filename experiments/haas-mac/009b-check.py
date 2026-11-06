#!/usr/bin/env python3
"""Run the original/off/on KDA checks and preserve every strict comparison result.

Usage: /usr/bin/python3 009b-check.py --original CHECK --candidate CHECK
       --model MODEL.gguf|synthetic --out DIR [--device cpu|gpu] [--threads 4]
This is a correctness driver. Each check invocation holds the shared campaign lock.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference(want, got):
    if not want.exists() or not got.exists():
        return {"exact": False, "status": "missing-output"}
    a, b = want.read_bytes(), got.read_bytes()
    result = {"exact": a == b, "bytes": [len(a), len(b)], "sha256": [sha256(want), sha256(got)]}
    if not a or not b or (want.suffix == ".f32" and (len(a) % 4 or len(b) % 4)):
        result.update(exact=False, status="empty-or-invalid-output")
        return result
    if want.suffix == ".f32" and len(a) == len(b) and len(a) % 4 == 0:
        aa, bb = struct.iter_unpack("=f", a), struct.iter_unpack("=f", b)
        maximum = signal = noise = 0.0
        finite = True
        for (x,), (y,) in zip(aa, bb):
            finite = finite and math.isfinite(x) and math.isfinite(y)
            if math.isfinite(x) and math.isfinite(y):
                maximum = max(maximum, abs(x - y))
                signal += x * x
                noise += (x - y) ** 2
        result.update(finite=finite, max_abs=maximum,
                      snr_db=None if noise == 0 or signal == 0 else 10 * math.log10(signal / noise),
                      zero_noise=noise == 0)
        result["exact"] = result["exact"] and finite
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lock", default="/private/tmp/speech-haas-mac-perf.lock")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.out.exists():
        parser.error("--out already exists; choose a new directory to preserve earlier evidence")
    args.out.mkdir(parents=True)
    result = {"python": sys.version, "python_executable": sys.executable,
              "device": args.device, "threads": args.threads, "model": args.model,
              "runs": {}, "comparisons": {}}
    if args.model != "synthetic":
        result["model_sha256"] = sha256(Path(args.model))
    base_env = os.environ.copy()
    inherited = {k: v for k, v in base_env.items()
                 if k.startswith(("SPEECH_HAAS_", "GGML_"))}
    result["inherited_experiment_environment"] = inherited
    allowed = {"GGML_CPU_NORM_MUL_ADD_FUSION", "GGML_CPU_NORM_MUL_ADD_FUSION_TRACE"}
    unexpected = {k: v for k, v in inherited.items() if k not in allowed}
    if unexpected:
        raise RuntimeError("unset unrelated SPEECH_HAAS_/GGML_ experiment variables before this independent check")
    for name, binary, enabled in (("original", args.original, "0"), ("off", args.candidate, "0"), ("on", args.candidate, "1")):
        directory = args.out / name
        directory.mkdir()
        env = base_env.copy()
        flags = {"GGML_CPU_NORM_MUL_ADD_FUSION": enabled, "GGML_CPU_NORM_MUL_ADD_FUSION_TRACE": "1"}
        env.update(flags)
        expected = 3 if args.device == "cpu" and enabled == "1" else 0
        command = ["/usr/bin/lockf", "-k", args.lock, str(binary.resolve()),
                   "--synthetic" if args.model == "synthetic" else str(Path(args.model).resolve()),
                   str(directory.resolve()), args.device, str(args.threads), str(expected)]
        with (directory / "stdout.log").open("wb") as stdout, (directory / "stderr.log").open("wb") as stderr:
            process = subprocess.run(command, env=env, stdout=stdout, stderr=stderr, check=False)
        result["runs"][name] = {"command": command, "environment": flags, "returncode": process.returncode,
                                 "binary_sha256": sha256(binary)}
        fixture = directory / "synthetic-kda-F32.gguf"
        if fixture.exists():
            result["runs"][name]["synthetic_model_sha256"] = sha256(fixture)
        (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for left, right in (("original", "off"), ("off", "on"), ("original", "on")):
        result["comparisons"][left + "-" + right] = {
            filename: difference(args.out / left / filename, args.out / right / filename)
            for filename in ("states.f32", "predictions.f32", "logits.f32", "inputs.f32", "direct.f32", "decisions.txt")}
    result["passed"] = all(run["returncode"] == 0 for run in result["runs"].values()) and all(
        item["exact"] for comparison in result["comparisons"].values() for item in comparison.values())
    result["official_stage_reference"] = "not-run"
    (args.out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"passed": result["passed"], "result": str(args.out / "result.json")}, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
