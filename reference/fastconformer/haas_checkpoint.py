"""Reads Haas's pinned archive and restores its official frontend, encoder and joint without a prediction network."""

from dataclasses import dataclass
import hashlib
import io
import json
import tarfile

import torch
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor, ConformerEncoder, RNNTJoint
from omegaconf import DictConfig, OmegaConf

from pins import MODELS, verify_checkpoint


MODEL = "Haas-ja-v1.0"
CLASSES = {
    "preprocessor": AudioToMelSpectrogramPreprocessor,
    "encoder": ConformerEncoder,
    "joint": RNNTJoint,
}


@dataclass
class HaasCheckpoint:
    """The configuration, original tensors and tokenizer stored together in the verified archive."""

    config: DictConfig
    tensors: dict[str, torch.Tensor]
    tokenizer: bytes
    config_sha256: str
    dtypes: list[str]


def load_checkpoint(path):
    """Reads the pinned archive on the CPU without constructing its model or downloading anything."""
    verify_checkpoint(path, MODELS[MODEL])
    print(json.dumps({"event": "checkpoint_verified", "sha256": MODELS[MODEL]["sha256"]}), flush=True)
    with tarfile.open(path, "r:*") as archive:
        members = {member.name.removeprefix("./"): member for member in archive.getmembers()}
        config_bytes = archive.extractfile(members["model_config.yaml"]).read()
        config = OmegaConf.load(io.StringIO(config_bytes.decode("utf-8")))
        with archive.extractfile(members["model_weights.ckpt"]) as source:
            state = torch.load(source, map_location="cpu", weights_only=True)
        tokenizer_path = config.tokenizer.model_path
        if not tokenizer_path.startswith("nemo:"):
            raise ValueError("The pinned checkpoint must contain its tokenizer as a nemo: artifact")
        tokenizer = archive.extractfile(members[tokenizer_path.removeprefix("nemo:")]).read()
    dtypes = sorted({str(value.dtype) for value in state.values() if isinstance(value, torch.Tensor)})
    return HaasCheckpoint(config, state, tokenizer, hashlib.sha256(config_bytes).hexdigest(), dtypes)


def restore_stages(checkpoint):
    """Strictly restores the official frontend, encoder and joint from the original checkpoint values in float32."""
    stages = {}
    for name, cls in CLASSES.items():
        config = checkpoint.config[name]
        if config._target_ != "nemo.collections.asr.modules." + cls.__name__:
            raise ValueError(f"{name} is not the official {cls.__name__}: {config._target_}")
        module = cls.from_config_dict(config).cpu().float().eval()
        prefix = name + "."
        weights = {key[len(prefix):]: value for key, value in checkpoint.tensors.items() if key.startswith(prefix)}
        module.load_state_dict(weights, strict=True)
        stages[name] = module
        print(json.dumps({"event": "stage_loaded", "stage": name, "tensors": len(weights)}), flush=True)
    return stages
