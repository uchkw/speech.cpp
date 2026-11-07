"""Restores and checks official Haas stages, then prepares fixed or enumerated Core ML artifacts.

Usage: python export.py --reference-source CHECKOUT --dumps DUMP_MODEL_DIR --checkpoint FILE
                       --lock FROZEN_UV_LOCK --shape-set fixed|enumerated --out NEW_OUTPUT_DIRECTORY
This executes CPU model work and conversion; run only under an assigned campaign compute lease.
"""

import argparse
import importlib.metadata
import inspect
from pathlib import Path
import platform
import shutil
import sys
import warnings

from contract import (HERE, PINS, REFERENCE_FILES, disk_preflight, model_identity, output_directory, read_json,
                      reference_source, require, selected_cases, sha256, validate_meta, write_json)


def environment(lock):
    require(platform.system() == "Darwin" and platform.machine() == "arm64", "Use the isolated Darwin arm64 environment.")
    require(platform.python_version() == PINS["converter_candidate"]["python"], "Use the pinned Python version.")
    packages = {name: importlib.metadata.version(name) for name in
                ("coremltools", "torch", "nemo-toolkit", "numpy", "lightning", "protobuf", "soundfile", "sentencepiece")}
    for name in ("coremltools", "torch", "nemo-toolkit"):
        require(packages[name] == PINS["converter_candidate"][name], "Unexpected converter package: " + name)
    require(Path(lock).is_file(), "Resolve and freeze the separate environment before conversion.")
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "lock_sha256": sha256(lock), "candidate_pyproject_sha256": sha256(HERE / "env" / "pyproject.toml")}


def execute(args, destination):
    disk = disk_preflight("export", destination)
    recorded_environment = environment(args.lock)
    source = reference_source(args.reference_source)
    write_json(destination / "run.json", {"pins": PINS, "shape_set": args.shape_set, "environment": recorded_environment,
               "disk_preflight": disk,
               "reference_source": str(source), "dumps": str(args.dumps.resolve()), "checkpoint": str(args.checkpoint.resolve())})
    sys.path.insert(0, str(source))
    import coremltools as ct
    import numpy as np
    import torch
    from haas_checkpoint import CLASSES, load_checkpoint, restore_stages
    from numerics import array, difference, stage

    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(0)
    checkpoint = load_checkpoint(args.checkpoint)
    stages = restore_stages(checkpoint)
    require(stages["preprocessor"].featurizer.pad_value == 0, "The official padding value changed.")
    provenance = {"config_sha256": checkpoint.config_sha256, "checkpoint_dtypes": checkpoint.dtypes,
                  "official_sources": {key: sha256(inspect.getfile(value)) for key, value in CLASSES.items()}}
    del checkpoint
    require(not any(name in sys.modules for name in ("haas_asr", "fla", "triton")), "The encoder export imported prediction kernels.")

    class Wrapper(torch.nn.Module):
        def __init__(self, diagnostic):
            super().__init__()
            self.encoder = stages["encoder"]
            self.joint = stages["joint"]
            self.diagnostic = diagnostic

        def forward(self, features, length):
            encoded, valid = self.encoder(audio_signal=features, length=length.to(torch.int64))
            projected = self.joint.project_encoder(encoded.transpose(1, 2))
            if self.diagnostic:
                return projected, valid.to(torch.int32), encoded
            return projected, valid.to(torch.int32)

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.joint = stages["joint"]

        def forward(self, encoded):
            return self.joint.project_encoder(encoded)

    cases, inputs, reference, replay = [], [], [], []
    input_directory = destination / "inputs"
    input_directory.mkdir()
    with torch.inference_mode():
        for case in selected_cases(args.shape_set):
            directory = Path(args.dumps).resolve() / case["id"]
            meta = read_json(directory / "meta.json")
            validate_meta(case, meta)
            require(provenance["official_sources"] == {name: row["source_sha256"] for name, row in
                    meta["environment"]["official_modules"].items()}, "The official module sources differ from the recorded reference.")
            feature = array(directory / "features.npy", (case["t"], 128))
            want_encoded = array(directory / "encoded.npy", (case["u"], 1024))
            want_joint = array(directory / "joint_encoded.npy", (case["u"], 640))
            features = np.zeros((1, 128, case["p"]), dtype=np.float32)
            features[0, :, :case["t"]] = feature.T
            signal = torch.from_numpy(array(directory / "audio.npy", (meta["samples"],)))[None]
            ordinary_features, ordinary_length = stages["preprocessor"](input_signal=signal, length=torch.tensor([signal.shape[1]]))
            require(int(ordinary_length[0]) == case["t"] and tuple(ordinary_features.shape) == features.shape,
                    "Reconstructed frontend extent differs: " + case["id"])
            stage(ordinary_features.numpy(), features, 90, "frontend reconstruction " + case["id"])
            require(np.count_nonzero(ordinary_features.numpy()[0, :, case["t"]:]) == 0, "Official padding is not zero.")
            item = (torch.from_numpy(features), torch.tensor([case["t"]], dtype=torch.int32))
            encoded, valid = stages["encoder"](audio_signal=item[0], length=item[1].to(torch.int64))
            require(tuple(encoded.shape) == (1, 1024, case["q"]) and int(valid[0]) == case["u"], "Official encoder extent differs.")
            projected = stages["joint"].project_encoder(encoded.transpose(1, 2))
            isolated = stages["joint"].project_encoder(torch.from_numpy(want_encoded)[None])
            replay.append({"id": case["id"],
                           "encoder": stage(encoded.numpy()[0, :, :case["u"]].T, want_encoded, 90,
                                            "encoder replay " + case["id"]),
                           "joint_isolated": stage(isolated.numpy()[0], want_joint, 90,
                                                   "isolated projection replay " + case["id"]),
                           "joint_composed": stage(projected.numpy()[0, :case["u"]], want_joint, 90,
                                                   "composed projection replay " + case["id"])})
            raw_features = input_directory / (case["id"] + ".features.f32")
            raw_encoded = input_directory / (case["id"] + ".encoded.f32")
            features.astype("<f4", copy=False).tofile(raw_features)
            want_encoded.astype("<f4", copy=False).tofile(raw_encoded)
            cases.append({**case, "reference_dir": str(directory),
                          "reference_files": {name: sha256(directory / name) for name in REFERENCE_FILES},
                          "features_raw": str(raw_features), "features_sha256": sha256(raw_features),
                          "encoded_raw": str(raw_encoded), "encoded_sha256": sha256(raw_encoded),
                          "seconds": meta["seconds"], "samples": meta["samples"]})
            inputs.append(item)
            reference.append((projected.detach().clone(), valid.to(torch.int32), encoded.detach().clone(), want_encoded))
        write_json(destination / "f32-replay.json", replay)

        models = {"f32": {}, "f16": {}}
        identities = {"f32": {}, "f16": {}}
        trace_reports = []
        for kind in ("normal", "diagnostic", "joint"):
            module = Projection().eval() if kind == "joint" else Wrapper(kind == "diagnostic").eval()
            examples = [(torch.from_numpy(row[3])[None],) for row in reference] if kind == "joint" else inputs
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                traced = torch.jit.trace(module, examples[-1], check_inputs=examples, strict=True)
            warnings_text = [str(warning.message) for warning in seen]
            for case, example, ordinary in zip(cases, examples, reference):
                actual = traced(*example)
                if kind == "joint":
                    stage(actual.numpy(), module(*example).numpy(), 90, "traced isolated joint " + case["id"])
                else:
                    require(torch.equal(actual[1], ordinary[1]), "Trace froze the valid length: " + case["id"])
                    stage(actual[0].numpy(), ordinary[0].numpy(), 90, "traced composed joint " + case["id"])
                    if kind == "diagnostic":
                        stage(actual[2].numpy(), ordinary[2].numpy(), 90, "traced encoder " + case["id"])
            trace_reports.append({"kind": kind, "warnings": warnings_text, "checked_cases": [case["id"] for case in cases]})
            write_json(destination / ("trace-" + kind + ".json"), trace_reports[-1])
            shapes = sorted({(1, row["u"], 1024) if kind == "joint" else (1, 128, row["p"]) for row in cases})
            shape = shapes[0] if len(shapes) == 1 else ct.EnumeratedShapes(shapes=shapes, default=shapes[-1])
            model_inputs = [ct.TensorType(name="encoded_input" if kind == "joint" else "features", shape=shape, dtype=np.float32)]
            model_outputs = [ct.TensorType(name="joint_encoded", dtype=np.float32)]
            if kind != "joint":
                model_inputs.append(ct.TensorType(name="length", shape=(1,), dtype=np.int32))
                model_outputs.append(ct.TensorType(name="encoded_length", dtype=np.int32))
            if kind == "diagnostic":
                model_outputs.append(ct.TensorType(name="encoded", dtype=np.float32))
            for precision, compute in (("f32", ct.precision.FLOAT32), ("f16", ct.precision.FLOAT16)):
                converted = ct.convert(traced, convert_to="mlprogram", inputs=model_inputs, outputs=model_outputs,
                                       compute_precision=compute, minimum_deployment_target=ct.target.macOS13,
                                       skip_model_load=True)
                path = destination / (kind + "-" + precision + ".mlpackage")
                converted.save(str(path))
                models[precision][kind] = str(path)
                identities[precision][kind] = model_identity(path)
                del converted
            del traced
        write_json(destination / "trace-check.json", trace_reports)

    snapshot = destination / "source"
    snapshot.mkdir()
    source_hashes = {}
    for path in sorted(HERE.glob("*.py")) + [HERE / "pins.json"]:
        shutil.copyfile(path, snapshot / path.name)
        source_hashes[path.name] = sha256(path)
    shutil.copyfile(args.lock, destination / "uv.lock")
    write_json(destination / "manifest.json", {"schema": 1, "experiment": "034", "pins": PINS,
               "shape_set": args.shape_set, "environment": recorded_environment, "provenance": provenance,
               "cases": cases, "models": models, "model_identities": identities,
               "source_sha256": source_hashes,
               "scope": "Official encoder and joint.enc only; no official Mac KDA/TDT validation."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference-source", "dumps", "checkpoint", "lock", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--shape-set", choices=("fixed", "enumerated"), required=True)
    args = parser.parse_args()
    destination = output_directory(args.out)
    try:
        execute(args, destination)
    except Exception as error:
        write_json(destination / "failure.json", {"stage": "export", "error": str(error), "type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
