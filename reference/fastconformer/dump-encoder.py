"""Dumps Haas's official NeMo frontend, encoder and encoder projection on the CPU in float32.

usage: uv run --project haas-encoder python dump-encoder.py --check-imports
       uv run --project haas-encoder python dump-encoder.py --checkpoint <Haas-ja-v1.0.nemo> <out dir> <audio.wav>...

The checkpoint is obtained separately from the revision in pins.py. Each input is mono WAVE at the model's
sample rate. Its folder is <out dir>/Haas-ja-v1.0/<audio stem> and holds the same audio, features, subsampled,
layers and encoded arrays as dump.py, plus joint_encoded [T, joint_hidden] from RNNTJoint.project_encoder().
meta.json identifies the scope as encoder-only: no prediction network, decoding, tokens or transcript is run.
The official classes and forward methods are used unchanged, with the archive's configuration and state dict.
"""

import argparse
import hashlib
import importlib.metadata
import inspect
import io
import json
from pathlib import Path
import platform
import sys
import tarfile

import numpy as np
import soundfile
import torch
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder, RNNTJoint
from omegaconf import OmegaConf

from pins import MODELS, NEMO, verify_checkpoint


MODEL = "Haas-ja-v1.0"
CPU_SEED = 0
CLASSES = {
    "preprocessor": AudioToMelSpectrogramPreprocessor,
    "encoder": ConformerEncoder,
    "joint": RNNTJoint,
}


def sha256(path):
    """Returns a file's SHA-256 without retaining its contents."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def environment():
    """Records the pinned environment and the files defining the official stage classes."""
    forbidden = [name for name in sys.modules if name in ("haas_asr", "fla", "triton")]
    if forbidden:
        raise RuntimeError(f"The encoder-only reference imported prediction dependencies: {forbidden}")
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "prediction_modules_imported": forbidden,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("nemo-toolkit", "torch", "lightning", "numpy", "soundfile")
        },
        "lock_sha256": sha256(Path(__file__).parent / "haas-encoder" / "uv.lock"),
        "script_sha256": sha256(__file__),
        "official_modules": {
            name: {"class": cls.__module__ + "." + cls.__name__, "source_sha256": sha256(inspect.getfile(cls))}
            for name, cls in CLASSES.items()
        },
    }


def restore_stages(path):
    """Restores only the official frontend, encoder and joint from the pinned archive on the CPU."""
    verify_checkpoint(path, MODELS[MODEL])
    print(json.dumps({"event": "checkpoint_verified", "sha256": MODELS[MODEL]["sha256"]}), flush=True)
    with tarfile.open(path, "r:*") as archive:
        members = {member.name.removeprefix("./"): member for member in archive.getmembers()}
        config_bytes = archive.extractfile(members["model_config.yaml"]).read()
        config = OmegaConf.load(io.StringIO(config_bytes.decode("utf-8")))
        with archive.extractfile(members["model_weights.ckpt"]) as source:
            state = torch.load(source, map_location="cpu", weights_only=True)
    stages = {}
    for name, cls in CLASSES.items():
        if config[name]._target_ != "nemo.collections.asr.modules." + cls.__name__:
            raise ValueError(f"{name} is not the official {cls.__name__}: {config[name]._target_}")
        module = cls.from_config_dict(config[name]).cpu().float().eval()
        prefix = name + "."
        weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(weights, strict=True)
        stages[name] = module
        print(json.dumps({"event": "stage_loaded", "stage": name, "tensors": len(weights)}), flush=True)
    checkpoint_dtypes = sorted({str(value.dtype) for value in state.values() if isinstance(value, torch.Tensor)})
    return config, stages, hashlib.sha256(config_bytes).hexdigest(), checkpoint_dtypes


def dump_audio(path, output, config, stages, provenance, threads):
    """Records the official encoder's stage outputs for one WAVE file, retaining only valid frames."""
    samples, rate = soundfile.read(path, dtype="float32")
    sample_rate = int(config.preprocessor.sample_rate)
    if rate != sample_rate or samples.ndim != 1 or not samples.size or not np.isfinite(samples).all():
        raise ValueError(f"{path} must contain nonempty finite mono audio at {sample_rate} Hz")
    out = output / MODEL / path.stem
    if out.exists():
        raise FileExistsError(f"{out} already exists; choose a new output folder")
    print(json.dumps({"event": "dump_started", "audio": str(path)}), flush=True)
    signal = torch.from_numpy(samples)[None]
    features, lengths = stages["preprocessor"](input_signal=signal, length=torch.tensor([signal.shape[1]]))
    frames = int(lengths[0])
    hooked = {"layers": []}
    encoder = stages["encoder"]
    hooks = [encoder.pre_encode.register_forward_hook(lambda module, inputs, result: hooked.update(subsampled=result[0]))]
    hooks += [layer.register_forward_hook(lambda module, inputs, result: hooked["layers"].append(result)) for layer in encoder.layers]
    try:
        encoded, length = encoder(audio_signal=features, length=torch.tensor([frames]))
    finally:
        for hook in hooks:
            hook.remove()
    valid = int(length[0])
    projected = stages["joint"].project_encoder(encoded.transpose(1, 2))
    saved = {
        "audio": samples,
        "features": features[0, :, :frames].T,
        "subsampled": hooked["subsampled"][0, :valid],
        "layers": torch.stack([layer[0, :valid] for layer in hooked["layers"]]),
        "encoded": encoded[0, :, :valid].T,
        "joint_encoded": projected[0, :valid],
    }
    if not torch.equal(saved["layers"][-1], saved["encoded"]):
        raise ValueError("The official encoder output differs from its last layer")
    arrays = {name: np.ascontiguousarray(value.detach().numpy() if isinstance(value, torch.Tensor) else value)
              for name, value in saved.items()}
    if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("An official stage output is not finite float32")
    meta = {
        "reference_scope": "frontend-encoder-joint.enc",
        "nemo": NEMO,
        "model": MODELS[MODEL],
        "audio": path.name,
        "audio_sha256": sha256(path),
        "samples": int(samples.size),
        "seconds": samples.size / sample_rate,
        "sample_rate": sample_rate,
        "frames": frames,
        "encoded_frames": valid,
        "encoder_input_frames": int(features.shape[2]),
        "encoder_output_frames": int(encoded.shape[2]),
        "device": "cpu",
        "dtype": "float32",
        "cpu_seed": CPU_SEED,
        "threads": threads,
        "arrays": {name: list(value.shape) for name, value in arrays.items()},
        **provenance,
    }
    out.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(out / (name + ".npy"), value)
    with open(out / "meta.json", "w", encoding="utf-8") as destination:
        json.dump(meta, destination, ensure_ascii=False, indent=1)
    print(json.dumps({"output": str(out), "seconds": meta["seconds"], "frames": frames, "encoded_frames": valid}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-imports", action="store_true", help="report official imports without opening weights")
    parser.add_argument("--checkpoint", type=Path, help="the pinned .nemo archive already downloaded")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("out_dir", type=Path, nargs="?")
    parser.add_argument("audio", type=Path, nargs="*")
    args = parser.parse_args()
    provenance = {"environment": environment()}
    if args.check_imports:
        if args.checkpoint or args.out_dir or args.audio:
            parser.error("--check-imports takes no checkpoint, output directory or audio")
        print(json.dumps(provenance, indent=1))
        return
    if not args.checkpoint or not args.out_dir or not args.audio or args.threads < 1:
        parser.error("give --checkpoint, an output directory, WAVE files and a positive --threads")
    torch.set_num_threads(args.threads)
    torch.random.default_generator.manual_seed(CPU_SEED)
    with torch.inference_mode():
        config, stages, config_sha, checkpoint_dtypes = restore_stages(args.checkpoint)
        provenance.update(config_sha256=config_sha, checkpoint_dtypes=checkpoint_dtypes)
        for path in args.audio:
            dump_audio(path, args.out_dir, config, stages, provenance, args.threads)


if __name__ == "__main__":
    main()
