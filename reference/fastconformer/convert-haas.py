"""Converts the pinned original Haas archive to the existing F32 GGUF layout on a CPU, without KDA execution.

usage: python haas_sources.py <source folder>
       uv run --project haas-encoder python convert-haas.py --checkpoint <Haas-ja-v1.0.nemo> --sources <source folder> <out dir>

The official frontend, encoder and joint are restored strictly. The decoder's weights come directly from the original
checkpoint, and its constants from the config and verified official sources. The common writer is also used by the
full-model converter. conversion.json records the provenance; it does not claim a check of KDA or TDT inference.
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys

import sentencepiece
import torch

from gguf_writer import write_model
from haas_checkpoint import MODEL, load_checkpoint, restore_stages
from haas_decoder import decoder_settings
from haas_sources import Sources
from pins import MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    sources = Sources(args.sources)
    torch.set_num_threads(4)
    torch.random.default_generator.manual_seed(0)
    with torch.inference_mode():
        checkpoint = load_checkpoint(args.checkpoint)
        stages = restore_stages(checkpoint)
        tokenizer = sentencepiece.SentencePieceProcessor(model_proto=checkpoint.tokenizer)
        settings = decoder_settings(checkpoint, stages["joint"], tokenizer, sources)
        print(json.dumps({"event": "decoder_settings_verified", "settings": settings}), flush=True)
        path = write_model(MODEL, args.out_dir, checkpoint.config, stages["preprocessor"].featurizer,
                           stages["encoder"], checkpoint.tensors, tokenizer, settings)
    assert not any(name in sys.modules for name in ("haas_asr", "fla", "triton"))
    report = {
        "model": MODELS[MODEL], "file": str(path), "checkpoint_dtypes": checkpoint.dtypes,
        "config_sha256": checkpoint.config_sha256, "tokenizer_sha256": hashlib.sha256(checkpoint.tokenizer).hexdigest(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ("nemo-toolkit", "torch", "gguf", "sentencepiece", "numpy", "protobuf")},
        "lock_sha256": hashlib.sha256((Path(__file__).parent / "haas-encoder" / "uv.lock").read_bytes()).hexdigest(),
        "reference_files": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                            for name in ("convert-haas.py", "gguf_writer.py", "haas_checkpoint.py", "haas_decoder.py", "haas_sources.py", "pins.py")},
        "source_archives": sources.pins, "source_files": sources.files, "decoder_settings": settings,
        "environment_note": "The Darwin lock overrides NeMo's protobuf constraint; full NeMo compatibility is not checked.",
        "scope": "F32 weight conversion; no KDA or TDT reference inference",
    }
    (args.out_dir / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
