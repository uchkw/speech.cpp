"""The pinned checkpoints every script in this folder runs, and the version of NeMo their environment pins.

A checkpoint runs in the environment its pin names: pyproject.toml's here, or haas/pyproject.toml's for Haas, whose
prediction network needs Haas-ASR, flash-linear-attention and a CUDA GPU (`uv run --project haas python dump.py ...` on
Linux).
"""

import hashlib
import importlib.metadata
import importlib.util
import os

import torch
from huggingface_hub import hf_hub_download
from nemo.collections.asr.models import ASRModel

NEMO = importlib.metadata.version("nemo-toolkit")
# The environment the scripts run in: haas/'s, the one that holds Haas-ASR, or this folder's.
ENVIRONMENT = "haas" if importlib.util.find_spec("haas_asr") else None
MODELS = {
    "parakeet-tdt_ctc-0.6b-ja": {
        "repository": "nvidia/parakeet-tdt_ctc-0.6b-ja",
        "revision": "44edb27eea9317daf89333e75eb830db4b1cc298",
        "file": "parakeet-tdt_ctc-0.6b-ja.nemo",
        "size": 2490951680,
        "sha256": "a2ed2aab9c82cce4e9b699dca669184d7525c7b5e1eb8c3f7b789421bc3637e8",
    },
    "parakeet-tdt-0.6b-v3": {
        "repository": "nvidia/parakeet-tdt-0.6b-v3",
        "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "file": "parakeet-tdt-0.6b-v3.nemo",
        "size": 2509332480,
        "sha256": "3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d",
    },
    "reazonspeech-nemo-v2": {
        "repository": "reazon-research/reazonspeech-nemo-v2",
        "revision": "33693408be76b7cba9fd4a7546a0a8772430211b",
        "file": "reazonspeech-nemo-v2.nemo",
        "size": 2477946880,
        "sha256": "d196d43ad03466ca88beeda4bf5fafb07bab7202d4b663b8e4f12cb0a4381fae",
    },
    "Haas-ja-v1.0": {
        "repository": "third-intelligence/Haas-ja-v1.0",
        "revision": "660728d39019e5e065046154beca44ac9a66a540",
        "file": "Haas-ja-v1.0.nemo",
        "size": 1232384000,
        "sha256": "c6ed38698300a371a86eadab4b56d5458f47623f7fc4a5f1e938b5adbe8458aa",
        "environment": "haas",
    },
}


def checkpoint(pin):
    """The local path of a pinned .nemo checkpoint, downloaded on first use and checked against its pin."""
    path = hf_hub_download(pin["repository"], pin["file"], revision=pin["revision"])
    return verify_checkpoint(path, pin)


def verify_checkpoint(path, pin):
    """Checks a local checkpoint's size and digest without downloading or restoring a model."""
    size = os.path.getsize(path)
    if size != pin["size"]:
        raise SystemExit(f"{path} has {size} bytes, not the pinned {pin['size']}")
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    if digest.hexdigest() != pin["sha256"]:
        raise SystemExit(f"{path} has the SHA-256 {digest.hexdigest()}, not the pinned {pin['sha256']}")
    return path


def prediction_on_gpu(model):
    """Moves the prediction network of a Haas model on the CPU to the CUDA GPU, where it computes in float32 without
    TF32, its inputs going there and its outputs and state coming back to the CPU.

    fla's KimiDeltaAttention, which each of its layers is, decodes with Triton kernels alone: ShortConvolution's
    causal_conv1d_update, fused_recurrent_kda and FusedRMSNormGated. The rest of the model stays on the CPU, as every
    other checkpoint here runs, and the decoding with it, on CPU tensors.
    """
    if not torch.cuda.is_available():
        raise SystemExit("Haas's prediction network runs flash-linear-attention's Triton kernels, which need a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    network = model.decoder.prediction["dec_rnn"]
    network.cuda()
    forward = network.forward

    def on_gpu(y, state=None):
        g, state = forward(y.cuda(), None if state is None else tuple(s.cuda() for s in state))
        return g.cpu(), tuple(s.cpu() for s in state)

    network.forward = on_gpu


def restore(pin):
    """The official model of a pinned checkpoint, on the CPU in float32 and in evaluation, but for Haas's prediction
    network, which prediction_on_gpu() puts on the GPU."""
    if pin.get("environment") != ENVIRONMENT:
        where = "the environment of haas/ (uv run --project haas)" if pin.get("environment") else "this folder's environment"
        raise SystemExit(f"{pin['repository']} runs in {where}")
    torch.set_grad_enabled(False)
    if ENVIRONMENT == "haas":
        # Adds haas_asr to the packages whose classes NeMo builds from a configuration: the checkpoint's decoder is
        # haas_asr.prednet.KDATransducerDecoder.
        import haas_asr  # noqa: F401
    model = ASRModel.restore_from(checkpoint(pin), map_location="cpu").float().eval()
    if ENVIRONMENT == "haas":
        prediction_on_gpu(model)
    return model
