"""Derives Haas's conversion settings from its original config and pinned official source, without a decoder instance."""

import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
from nemo.collections.asr.parts.submodules.rnnt_decoding import AbstractRNNTDecoding, extract_punctuation_from_vocab

from haas_sources import assignment, defaults, definition, literal
from pins import NEMO


def calls(source, name, callee):
    """Finds calls to a particular constructor or kernel in a pinned function."""
    return [node for node in ast.walk(definition(source, name)) if isinstance(node, ast.Call) and ast.unparse(node.func) == callee]


def decoder_settings(checkpoint, joint, tokenizer, sources):
    """Checks the fixed Haas/FLA architecture and obtains the metadata the existing GGUF writer needs."""
    assert NEMO == "2.7.3"
    cfg, state = checkpoint.config, checkpoint.tensors
    dec, pred = cfg.decoder, cfg.decoder.prednet
    assert dec._target_ == "haas_asr.prednet.KDATransducerDecoder"
    assert dec.normalization_mode is None and dec.blank_as_pad and not dec.random_state_sampling
    assert cfg.decoding.strategy == "greedy_batch" and cfg.decoding.model_type == "tdt"
    assert cfg.loss.loss_name == "tdt"
    durations = [int(value) for value in cfg.loss.tdt_kwargs.durations]
    assert durations == list(cfg.decoding.durations) and not cfg.decoding.get("big_blank_durations")
    assert set(cfg.decoding.greedy) == {"max_symbols"}
    blank = tokenizer.vocab_size()
    assert blank == dec.vocab_size == cfg.joint.num_classes
    assert joint.encoder_hidden == cfg.model_defaults.enc_hidden
    assert joint.pred_hidden == cfg.model_defaults.pred_hidden == pred.pred_hidden
    assert joint.activation == "relu" and joint.temperature == 1.0 and not joint.is_adapter_available()
    assert joint.num_extra_outputs == len(durations)
    assert joint.joint_net[-1].out_features == blank + 1 + len(durations)
    assert state["decoder.prediction.embed.weight"].shape == (blank + 1, pred.pred_hidden)
    assert float(state["decoder.prediction.embed.weight"][blank].abs().max()) == 0

    haas = sources.read("haas-asr", "src/haas_asr/prednet.py")
    fla = sources.read("flash-linear-attention", "fla/layers/kda.py")
    norm = sources.read("fla-core", "fla/modules/fused_norm_gate.py")
    recurrent = sources.read("fla-core", "fla/ops/kda/fused_recurrent.py")
    fields = {node.target.id for node in definition(haas, "KDAConfig").body if isinstance(node, ast.AnnAssign)}
    assert {key.removeprefix("kda_") for key in pred if key.startswith("kda_")} == fields
    network_calls = calls(haas, "KDATransducerDecoder._predict_modules", "KDAPredictionNetwork")
    assert len(network_calls) == 1 and {kw.arg for kw in network_calls[0].keywords} == {"hidden_size", "num_layers", "config", "dropout"}
    layer_calls = calls(haas, "KDAPredictionNetwork.__init__", "kda_cls")
    assert len(layer_calls) == 1
    assert {kw.arg: ast.unparse(kw.value) for kw in layer_calls[0].keywords} == {
        "hidden_size": "hidden_size", "num_heads": "config.num_heads", "head_dim": "config.head_dim",
        "expand_v": "config.expand_v", "conv_size": "config.conv_size", "mode": "'chunk'",
        "use_short_conv": "True", "layer_idx": "i",
    }
    layer_defaults = defaults(fla, "KimiDeltaAttention.__init__")
    assert layer_defaults["num_v_heads"] is None and not layer_defaults["conv_bias"]
    assert not layer_defaults["allow_neg_eigval"] and not layer_defaults["safe_gate"] and layer_defaults["lower_bound"] is None
    assert pred.kda_expand_v == 1.0
    constructor = ast.unparse(definition(fla, "KimiDeltaAttention.__init__"))
    assert ast.unparse(assignment(constructor, "self.o_norm")) == "FusedRMSNormGated(self.head_v_dim, activation='sigmoid', eps=norm_eps)"
    assert defaults(norm, "FusedRMSNormGated.__init__")["elementwise_affine"]
    assert "self.register_parameter('bias', None)" in ast.unparse(definition(norm, "FusedRMSNormGated.__init__"))
    kernel_calls = calls(fla, "KimiDeltaAttention.forward", "fused_recurrent_kda")
    assert len(kernel_calls) == 1
    assert {kw.arg: ast.unparse(kw.value) for kw in kernel_calls[0].keywords}["use_qk_l2norm_in_kernel"] == "True"
    epsilons = []
    for node in ast.walk(definition(recurrent, "fused_recurrent_kda_fwd_kernel")):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.BinOp) and isinstance(node.value.op, ast.Div):
            target = ast.unparse(node.targets[0])
            if target in ("b_q", "b_k"):
                value = node.value
                assert ast.unparse(value.left) == target and ast.unparse(value.right.func) == "tl.sqrt"
                argument = value.right.args[0]
                assert isinstance(argument.op, ast.Add) and ast.unparse(argument.left) == f"tl.sum({target} * {target})"
                epsilons.append(ast.literal_eval(argument.right))
    assert len(epsilons) == 2 and epsilons[0] == epsilons[1]
    width = int(pred.kda_num_heads * pred.kda_head_dim)
    for layer in range(pred.pred_rnn_layers):
        prefix = f"decoder.prediction.dec_rnn.layers.{layer}."
        for axis in ("q", "k", "v"):
            assert state[prefix + axis + "_proj.weight"].shape == (width, pred.pred_hidden)
            assert state[prefix + axis + "_conv1d.weight"].shape == (width, 1, pred.kda_conv_size)
        assert state[prefix + "o_norm.weight"].shape == (pred.kda_head_dim,)

    boosting = sources.read("haas-asr", "src/haas_asr/boosting.py")
    vocabulary = sources.read("haas-asr", "src/haas_asr/vocabulary.py")
    audio = sources.read("haas-asr", "src/haas_asr/audio.py")
    assert literal(audio, "SAMPLE_RATE") == cfg.preprocessor.sample_rate
    frame = assignment(boosting, "SPEECH_FRAME")
    assert ast.unparse(frame.func) == "int" and len(frame.args) == 1
    assert isinstance(frame.args[0].op, ast.Mult) and ast.unparse(frame.args[0].right) == "SAMPLE_RATE"
    speech_frame = int(ast.literal_eval(frame.args[0].left) * literal(audio, "SAMPLE_RATE"))
    tree_defaults = BoostingTreeModelConfig()
    assert not tree_defaults.uniform_weights
    nemo = textwrap.dedent(inspect.getsource(AbstractRNNTDecoding.__init__))
    separators = assignment(nemo, "self.segment_seperators")
    assert ast.unparse(separators.func) == "self.cfg.get" and ast.literal_eval(separators.args[0]) == "segment_seperators"
    for cls in (AbstractRNNTDecoding, BoostingTreeModelConfig):
        path = Path(inspect.getfile(cls))
        sources.files["nemo/" + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "kind": "tdt", "prediction": "kda", "blank": blank, "layers": int(pred.pred_rnn_layers),
        "durations": durations, "max_symbols": int(cfg.decoding.greedy.max_symbols),
        "separators": list(cfg.decoding.get("segment_seperators", ast.literal_eval(separators.args[1]))),
        "punctuation": sorted(extract_punctuation_from_vocab([tokenizer.id_to_piece(i) for i in range(blank)])),
        "kda": {"num_heads": int(pred.kda_num_heads),
                "norm_eps": defaults(haas, "KDAPredictionNetwork.__init__")["norm_eps"],
                "gate_norm_eps": layer_defaults["norm_eps"], "qk_norm_eps": epsilons[0],
                "final_norm": bool(pred.kda_final_norm)},
        "vocabulary": {"context_score": literal(boosting, "CONTEXT_SCORE"),
                       "depth_scaling": literal(boosting, "DEPTH_SCALING"), "unk_score": float(tree_defaults.unk_score),
                       "alpha": defaults(vocabulary, "Vocabulary.__init__")["alpha"],
                       "speech_rms": literal(boosting, "SPEECH_RMS"), "speech_frame": speech_frame},
    }
