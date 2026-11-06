"""Writes FastConformer GGUF metadata and float32 tensors from checked official components and decoder settings."""

import os

import numpy as np
from gguf import GGUFValueType, GGUFWriter, LlamaFileType, naming_convention, size_label
from nemo.collections.asr.parts.preprocessing import features
from nemo.collections.asr.parts.submodules import multi_head_attention
from sentencepiece import sentencepiece_model_pb2

from pins import MODELS

ARCH = "fastconformer"
# Each layout this converter has written, with the first release of speech.cpp whose reader takes it; it writes the
# last.
RELEASES = {1: "0.7.0", 2: "0.8.0", 3: "0.9.3", 4: "0.9.4"}
LAYOUT = max(RELEASES)
# The ISO 639 two-letter codes of the languages each checkpoint transcribes, from its model card, which requests give as
# BCP 47 tags. None takes a language: parakeet-tdt-0.6b-v3 finds the language of the audio itself.
LANGUAGES = {
    "parakeet-tdt_ctc-0.6b-ja": ["ja"],
    "reazonspeech-nemo-v2": ["ja"],
    "parakeet-tdt-0.6b-v3": ["en", "es", "fr", "de", "bg", "hr", "cs", "da", "nl", "et", "fi", "el", "hu", "it", "lv",
                             "lt", "mt", "pl", "pt", "ro", "sk", "sl", "sv", "ru", "uk"],
    "Haas-ja-v1.0": ["ja"],
}
# The SPDX identifier of each checkpoint's license, from its model card.
LICENSES = {"parakeet-tdt_ctc-0.6b-ja": "CC-BY-4.0", "parakeet-tdt-0.6b-v3": "CC-BY-4.0", "reazonspeech-nemo-v2": "Apache-2.0",
            "Haas-ja-v1.0": "Apache-2.0"}
# The parts of each model's name under GGUF's naming convention (ggml's docs/gguf.md): its line, its size where the name
# gives one, what it was trained toward (ja, the language of parakeet-tdt_ctc-0.6b-ja) and its version. Where the name
# gives no size, the size label is counted from the parameters of the file's tensors.
NAMES = {
    "parakeet-tdt_ctc-0.6b-ja": {"basename": "parakeet-tdt_ctc", "size_label": "0.6B", "finetune": "ja", "version": None},
    "parakeet-tdt-0.6b-v3": {"basename": "parakeet-tdt", "size_label": "0.6B", "finetune": None, "version": "v3"},
    "reazonspeech-nemo-v2": {"basename": "reazonspeech-nemo", "size_label": None, "finetune": None, "version": "v2"},
    "Haas-ja-v1.0": {"basename": "Haas", "size_label": None, "finetune": "ja", "version": "v1.0"},
}
# speech.cpp's addition to NeMo's segments: the marks that end a segment wherever they stand, for a model whose
# languages are written without spaces. NeMo ends a segment at one of its separators only where a word ends, which
# text without spaces between its words never reaches.
UNSPACED_LANGUAGES = {"ja", "zh"}
UNSPACED_BREAKS = ["。", "？", "！", "?", "!"]


class Writer(GGUFWriter):
    """A GGUFWriter that writes an empty array of a given element type, which GGUFWriter refuses."""

    def _pack_val(self, val, vtype, add_vtype, sub_type=None):
        if vtype == GGUFValueType.ARRAY and sub_type is not None and len(val) == 0:
            return (self._pack("I", vtype) if add_vtype else b"") + self._pack("I", sub_type) + self._pack("Q", 0)
        return super()._pack_val(val, vtype, add_vtype, sub_type)


def write_model(model_name, out_dir, cfg, featurizer, encoder, state, tokenizer, settings):
    """Writes the existing GGUF layout from official weights, preserving its tensor mapping and batch-norm folding."""
    os.makedirs(out_dir, exist_ok=True)
    pin = MODELS[model_name]
    decoder = settings["kind"]
    prediction = settings["prediction"]
    durations = settings["durations"]
    enc = cfg.encoder
    assert enc.subsampling == "dw_striding" and not enc.get("causal_downsampling", False)
    assert enc.conv_norm_type == "batch_norm" and encoder.conv_context_size == [(enc.conv_kernel_size - 1) // 2] * 2
    # Attention over the whole utterance, or Longformer's local attention with as many frames on either side and one or
    # more global tokens from the first frame on, whose queries, keys and values are the layer's own.
    attention = encoder.self_attention_model
    assert encoder.att_context_style == "regular" and len(encoder.att_context_size_all) == 1
    context = list(encoder.att_context_size)
    if attention == "rel_pos":
        assert context == [-1, -1]
    else:
        assert attention == "rel_pos_local_attn" and context[0] == context[1] > 0, (attention, context)
        assert encoder.global_tokens == 1 or (encoder.global_tokens > 1 and encoder.global_tokens_spacing == 1)
        assert not encoder.global_attn_separate
    assert featurizer.exact_pad is False and featurizer.frame_splicing == 1 and featurizer.mag_power == 2.0
    assert featurizer.normalize == "per_feature" and featurizer.log and featurizer.log_zero_guard_type == "add"
    assert featurizer.pad_value == 0

    w = Writer(None, ARCH)

    def add_array(key, values, element):
        """An array with its element type given, not inferred from its first value."""
        w.add_key_value(key, list(values), GGUFValueType.ARRAY, sub_type=element)

    w.add_uint32("speech.layout", LAYOUT)
    w.add_string("speech.requires", RELEASES[LAYOUT])
    w.add_string("speech.task", "recognition")
    w.add_uint32("speech.sample_rate", int(featurizer.sample_rate))
    w.add_string("speech.language_use", "checked")

    # The frontend: FilterbankFeatures' parameters, and the normalization guard, CONSTANT in features.py.
    w.add_uint32("fastconformer.frontend.n_fft", int(featurizer.n_fft))
    w.add_uint32("fastconformer.frontend.hop_length", int(featurizer.hop_length))
    w.add_uint32("fastconformer.frontend.n_mels", int(featurizer.nfilt))
    w.add_float32("fastconformer.frontend.preemphasis", float(featurizer.preemph))
    w.add_float32("fastconformer.frontend.log_guard", float(featurizer.log_zero_guard_value))
    w.add_float32("fastconformer.frontend.std_guard", float(features.CONSTANT))

    # The encoder. pos_base is INF_VAL of multi_head_attention.py, the base of RelPositionalEncoding's
    # wavelengths; ff_factor is ConformerLayer's fc_factor, the weight of each half-step feed-forward.
    w.add_uint32("fastconformer.encoder.d_model", int(enc.d_model))
    w.add_uint32("fastconformer.encoder.num_layers", int(enc.n_layers))
    w.add_uint32("fastconformer.encoder.num_heads", int(enc.n_heads))
    w.add_uint32("fastconformer.encoder.conv_kernel", int(enc.conv_kernel_size))
    w.add_uint32("fastconformer.encoder.subsampling_factor", int(enc.subsampling_factor))
    w.add_float32("fastconformer.encoder.norm_eps", float(encoder.layers[0].norm_out.eps))
    w.add_float32("fastconformer.encoder.pos_base", float(multi_head_attention.INF_VAL))
    w.add_float32("fastconformer.encoder.xscale", float(encoder.xscale or 1.0))
    w.add_float32("fastconformer.encoder.ff_factor", float(encoder.layers[0].fc_factor))
    # Whether the linear layers of the feed-forward modules and the attention and the pointwise convolutions have biases;
    # the depthwise convolution has one either way once the batch norm is folded into it.
    use_bias = bool(encoder.layers[0].feed_forward1.use_bias)
    assert all(bool(m.use_bias) == use_bias for layer in encoder.layers
               for m in (layer.feed_forward1, layer.feed_forward2, layer.self_attn, layer.conv))
    w.add_bool("fastconformer.encoder.use_bias", use_bias)
    w.add_string("fastconformer.encoder.attention", attention)
    if attention == "rel_pos_local_attn":
        w.add_uint32("fastconformer.encoder.attention_context", int(context[0]))
        w.add_uint32("fastconformer.encoder.global_tokens", int(encoder.global_tokens))

    # The joint's classes are the tokenizer's ids, a blank after them, then one per duration.
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(tokenizer.serialized_model_proto())
    # Normal, user-defined, unknown and byte pieces only: SentencePiece writes a user-defined piece (parakeet-tdt-0.6b-v3's
    # <|en|> and the other tags) as it writes a normal one, and the C++ detokenizer handles no control pieces. A tokenizer
    # with byte fallback (Haas's) has a byte piece <0xXX> for each of the 256 bytes.
    allowed = (proto.SentencePiece.NORMAL, proto.SentencePiece.USER_DEFINED, proto.SentencePiece.UNKNOWN, proto.SentencePiece.BYTE)
    assert all(p.type in allowed for p in proto.pieces)
    byte_pieces = {int(p.piece[3:5], 16): i for i, p in enumerate(proto.pieces) if p.type == proto.SentencePiece.BYTE}
    assert all(proto.pieces[i].piece == f"<0x{b:02X}>" for b, i in byte_pieces.items())
    assert sorted(byte_pieces) == (list(range(256)) if proto.trainer_spec.byte_fallback else [])
    assert not proto.trainer_spec.treat_whitespace_as_suffix
    blank = settings["blank"]
    assert blank == len(proto.pieces) == tokenizer.vocab_size()
    w.add_string("fastconformer.decoder.kind", decoder)
    w.add_uint32("fastconformer.decoder.blank_id", blank)
    w.add_uint32("fastconformer.decoder.prediction_layers", int(settings["layers"]))
    w.add_string("fastconformer.decoder.prediction", prediction)
    if prediction == "kda":
        w.add_uint32("fastconformer.decoder.kda.num_heads", int(settings["kda"]["num_heads"]))
        # The epsilon of the blocks' layer norms and of the final one, and of the output norm of each KDA layer.
        w.add_float32("fastconformer.decoder.kda.norm_eps", float(settings["kda"]["norm_eps"]))
        w.add_float32("fastconformer.decoder.kda.gate_norm_eps", float(settings["kda"]["gate_norm_eps"]))
        # The epsilon under the square root of fused_recurrent_kda_fwd_kernel's normalization of q and k
        # (fla/ops/kda/fused_recurrent.py), a literal of the kernel: b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6).
        w.add_float32("fastconformer.decoder.kda.qk_norm_eps", settings["kda"]["qk_norm_eps"])
        w.add_bool("fastconformer.decoder.kda.final_norm", settings["kda"]["final_norm"])
    if decoder == "tdt":
        add_array("fastconformer.decoder.tdt.durations", durations, GGUFValueType.INT32)
        # The most tokens emitted on one frame before the decoding moves to the next.
        w.add_uint32("fastconformer.decoder.tdt.max_symbols", int(settings["max_symbols"]))
    else:
        w.add_uint32("fastconformer.decoder.rnnt.beam_size", int(settings["beam_size"]))
        # Whether the best of the finished hypotheses is the one with the highest score per label, the blank it starts
        # with counted.
        w.add_bool("fastconformer.decoder.rnnt.score_norm", bool(settings["score_norm"]))
        # The most labels a hypothesis takes, as a multiple of the encoder's frames.
        w.add_float32("fastconformer.decoder.rnnt.max_target_ratio", float(settings["max_target_ratio"]))
        # The most tokens greedy decoding emits on one frame before it moves to the next.
        w.add_uint32("fastconformer.decoder.rnnt.max_symbols", settings["max_symbols"])
    # The marks that end a segment where a word ends: the checkpoint's decoding segment_seperators, or NeMo's default
    # (".", "?", "!") when it sets none, as the decoding object takes them.
    add_array("fastconformer.segment.separators", list(settings["separators"]), GGUFValueType.STRING)
    unspaced = all(language in UNSPACED_LANGUAGES for language in LANGUAGES[model_name])
    add_array("fastconformer.segment.breaks", UNSPACED_BREAKS if unspaced else [], GGUFValueType.STRING)
    add_array("fastconformer.tokenizer.tokens", [p.piece for p in proto.pieces], GGUFValueType.STRING)
    add_array("fastconformer.tokenizer.byte_pieces", [byte_pieces[b] for b in sorted(byte_pieces)], GGUFValueType.INT32)
    w.add_uint32("fastconformer.tokenizer.unknown_id", int(tokenizer.unk_id()))
    w.add_string("fastconformer.tokenizer.unknown_surface", proto.trainer_spec.unk_surface)
    # Whether SentencePiece's decoder drops the leading "▁" of each piece until the text is no longer empty, which it
    # does when either normalizer option is set.
    w.add_bool("fastconformer.tokenizer.strip_leading_space",
               bool(proto.normalizer_spec.add_dummy_prefix or proto.normalizer_spec.remove_extra_whitespaces))
    # The marks before which the decoding removes one whitespace character. The C++ looks for a space only,
    # the one whitespace character a decoded text can hold when no piece holds another and the unknown surface
    # holds only spaces.
    assert not any(c.isspace() for p in proto.pieces for c in p.piece)
    assert all(c == " " or not c.isspace() for c in proto.trainer_spec.unk_surface)
    add_array("fastconformer.tokenizer.punctuation", settings["punctuation"], GGUFValueType.STRING)
    # The custom vocabulary of Haas-ASR (haas_asr.vocabulary and haas_asr.boosting), which its package offers for its
    # model alone: each word's spellings are encoded by the tokenizer and boosted while decoding by NeMo's boosting tree.
    vocabulary = "haas-asr" if MODELS[model_name].get("environment") == "haas" else "none"
    w.add_string("fastconformer.vocabulary.kind", vocabulary)
    if vocabulary == "haas-asr":
        # SentencePiece's encoding of a BPE model: the pieces' scores rank the merges, after the normalizer's precompiled
        # character map and its options have normalized the text.
        assert proto.trainer_spec.model_type == proto.trainer_spec.BPE
        # The C++ encoder takes normal, unknown and byte pieces alone, which the tokens, the unknown id and the byte pieces
        # say: SentencePiece treats a user-defined, control or unused piece otherwise in its encoding.
        assert all(p.type in (proto.SentencePiece.NORMAL, proto.SentencePiece.UNKNOWN, proto.SentencePiece.BYTE) for p in proto.pieces)
        add_array("fastconformer.tokenizer.scores", [float(p.score) for p in proto.pieces], GGUFValueType.FLOAT32)
        add_array("fastconformer.tokenizer.charsmap", list(proto.normalizer_spec.precompiled_charsmap), GGUFValueType.UINT8)
        w.add_bool("fastconformer.tokenizer.add_dummy_prefix", bool(proto.normalizer_spec.add_dummy_prefix))
        w.add_bool("fastconformer.tokenizer.remove_extra_whitespaces", bool(proto.normalizer_spec.remove_extra_whitespaces))
        w.add_bool("fastconformer.tokenizer.escape_whitespaces", bool(proto.normalizer_spec.escape_whitespaces))
        # The tree haas_asr compiles: ContextGraph's bonus per token and its depth scaling (CONTEXT_SCORE and
        # DEPTH_SCALING), and GPUBoostingTreeModel's score of a token that starts no word with BoostingTreeModelConfig's
        # defaults, whose uniform weights are off.
        w.add_float32("fastconformer.vocabulary.context_score", float(settings["vocabulary"]["context_score"]))
        w.add_float32("fastconformer.vocabulary.depth_scaling", float(settings["vocabulary"]["depth_scaling"]))
        w.add_float32("fastconformer.vocabulary.unk_score", float(settings["vocabulary"]["unk_score"]))
        # Vocabulary's boosting strength when none is given.
        w.add_float32("fastconformer.vocabulary.alpha", float(settings["vocabulary"]["alpha"]))
        # has_speech(): audio of which no frame of speech_frame samples is speech_rms loud is not boosted.
        w.add_float32("fastconformer.vocabulary.speech_rms", float(settings["vocabulary"]["speech_rms"]))
        w.add_uint32("fastconformer.vocabulary.speech_frame", int(settings["vocabulary"]["speech_frame"]))
    # The full reference can hold the prediction weights on a GPU.
    sd = {k: v.detach().float().cpu().numpy() for k, v in state.items()}

    def add(name, data):
        w.add_tensor(name, np.ascontiguousarray(data, dtype=np.float32))

    add("frontend.window", sd["preprocessor.featurizer.window"])
    add("frontend.filterbank", sd["preprocessor.featurizer.fb"][0])

    # The subsampling's nn.Sequential: Conv2d, ReLU, then (depthwise Conv2d, pointwise Conv2d, ReLU) per halving.
    s = "encoder.pre_encode."
    add("sub.conv.0.weight", sd[s + "conv.0.weight"])
    add("sub.conv.0.bias", sd[s + "conv.0.bias"])
    for i in range(1, int(np.log2(enc.subsampling_factor))):
        dw, pw = 3 * i - 1, 3 * i
        add(f"sub.conv.{i}.dw.weight", sd[s + f"conv.{dw}.weight"])
        add(f"sub.conv.{i}.dw.bias", sd[s + f"conv.{dw}.bias"])
        add(f"sub.conv.{i}.pw.weight", sd[s + f"conv.{pw}.weight"][:, :, 0, 0])
        add(f"sub.conv.{i}.pw.bias", sd[s + f"conv.{pw}.bias"])
    add("sub.out.weight", sd[s + "out.weight"])
    add("sub.out.bias", sd[s + "out.bias"])

    d = int(enc.d_model)
    for l in range(int(enc.n_layers)):
        a, o = f"encoder.layers.{l}.", f"blk.{l}."

        def linear(src, dst, bias=use_bias):
            add(dst + ".weight", sd[a + src + ".weight"])
            if bias:
                add(dst + ".bias", sd[a + src + ".bias"])

        def norm(src, dst):
            add(dst + ".weight", sd[a + src + ".weight"])
            add(dst + ".bias", sd[a + src + ".bias"])

        for ff in ("1", "2"):
            norm(f"norm_feed_forward{ff}", o + f"ff{ff}_norm")
            linear(f"feed_forward{ff}.linear1", o + f"ff{ff}_up")
            linear(f"feed_forward{ff}.linear2", o + f"ff{ff}_down")
        norm("norm_self_att", o + "attn_norm")
        for x in ("q", "k", "v", "out"):
            linear(f"self_attn.linear_{x}", o + f"attn_{x}")
        linear("self_attn.linear_pos", o + "attn_pos", bias=False)
        add(o + "attn_pos_bias_u", sd[a + "self_attn.pos_bias_u"])
        add(o + "attn_pos_bias_v", sd[a + "self_attn.pos_bias_v"])

        norm("norm_conv", o + "conv_norm")
        # GLU keeps the first half of the pointwise convolution's channels and gates them with the second.
        pw1 = sd[a + "conv.pointwise_conv1.weight"][:, :, 0]
        add(o + "conv_pw1_a.weight", pw1[:d])
        add(o + "conv_pw1_gate.weight", pw1[d:])
        if use_bias:
            add(o + "conv_pw1_a.bias", sd[a + "conv.pointwise_conv1.bias"][:d])
            add(o + "conv_pw1_gate.bias", sd[a + "conv.pointwise_conv1.bias"][d:])
        bn = a + "conv.batch_norm."
        scale = sd[bn + "weight"] / np.sqrt(sd[bn + "running_var"] + encoder.layers[l].conv.batch_norm.eps)
        dw_bias = sd[a + "conv.depthwise_conv.bias"] if use_bias else 0
        add(o + "conv_dw.weight", sd[a + "conv.depthwise_conv.weight"][:, 0, :] * scale[:, None])
        add(o + "conv_dw.bias", (dw_bias - sd[bn + "running_mean"]) * scale + sd[bn + "bias"])
        add(o + "conv_pw2.weight", sd[a + "conv.pointwise_conv2.weight"][:, :, 0])
        if use_bias:
            add(o + "conv_pw2.bias", sd[a + "conv.pointwise_conv2.bias"])
        norm("norm_out", o + "out_norm")

    # The prediction network: an embedding of the tokens and the blank, then LSTM layers whose gates are stacked as
    # PyTorch stacks them, input, forget, cell and output, or KDA blocks. A short convolution's kernel [channels, 1, taps]
    # is written [channels, taps], the oldest input's tap first, as ggml_ssm_conv() takes it.
    add("pred.embed.weight", sd["decoder.prediction.embed.weight"])
    for l in range(int(settings["layers"])):
        if prediction == "lstm":
            r = "decoder.prediction.dec_rnn.lstm."
            add(f"pred.lstm.{l}.ih.weight", sd[r + f"weight_ih_l{l}"])
            add(f"pred.lstm.{l}.hh.weight", sd[r + f"weight_hh_l{l}"])
            add(f"pred.lstm.{l}.bias", sd[r + f"bias_ih_l{l}"] + sd[r + f"bias_hh_l{l}"])
            continue
        r, o = f"decoder.prediction.dec_rnn.layers.{l}.", f"pred.kda.{l}."
        add(o + "norm.weight", sd[f"decoder.prediction.dec_rnn.norms.{l}.weight"])
        add(o + "norm.bias", sd[f"decoder.prediction.dec_rnn.norms.{l}.bias"])
        for x in ("q", "k", "v"):
            add(o + f"{x}.weight", sd[r + f"{x}_proj.weight"])
            add(o + f"{x}_conv.weight", sd[r + f"{x}_conv1d.weight"][:, 0, :])
        add(o + "f_a.weight", sd[r + "f_proj.0.weight"])
        add(o + "f_b.weight", sd[r + "f_proj.1.weight"])
        add(o + "b.weight", sd[r + "b_proj.weight"])
        add(o + "a_log", sd[r + "A_log"])
        add(o + "dt_bias", sd[r + "dt_bias"])
        add(o + "g_a.weight", sd[r + "g_proj.0.weight"])
        add(o + "g_b.weight", sd[r + "g_proj.1.weight"])
        add(o + "g_b.bias", sd[r + "g_proj.1.bias"])
        add(o + "o_norm.weight", sd[r + "o_norm.weight"])
        add(o + "o.weight", sd[r + "o_proj.weight"])
    if prediction == "kda" and settings["kda"]["final_norm"]:
        add("pred.norm.weight", sd["decoder.prediction.dec_rnn.final_norm.weight"])
        add("pred.norm.bias", sd["decoder.prediction.dec_rnn.final_norm.bias"])
    if prediction == "kda":
        # Every tensor of the KDA layers is written above.
        kda_parts = ["q_proj.weight", "k_proj.weight", "v_proj.weight", "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight",
                     "f_proj.0.weight", "f_proj.1.weight", "b_proj.weight", "A_log", "dt_bias", "g_proj.0.weight", "g_proj.1.weight",
                     "g_proj.1.bias", "o_norm.weight", "o_proj.weight"]
        expected = {f"layers.{l}.{part}" for l in range(int(settings["layers"])) for part in kda_parts}
        expected |= {f"norms.{l}.{part}" for l in range(int(settings["layers"])) for part in ("weight", "bias")}
        expected |= {f"final_norm.{part}" for part in ("weight", "bias")} if settings["kda"]["final_norm"] else set()
        present = {k[len("decoder.prediction.dec_rnn."):] for k in sd if k.startswith("decoder.prediction.dec_rnn.")}
        assert present == expected, sorted(present ^ expected)
    for x in ("enc", "pred"):
        add(f"joint.{x}.weight", sd[f"joint.{x}.weight"])
        add(f"joint.{x}.bias", sd[f"joint.{x}.bias"])
    add("joint.out.weight", sd["joint.joint_net.2.weight"])
    add("joint.out.bias", sd["joint.joint_net.2.bias"])

    # The model's identity and languages in the GGUF specification's general keys, which name the file. They follow the
    # tensors, whose parameters give a size label the name does not, and then go first after general.architecture, since
    # GGUFWriter writes the keys in the order they were added.
    parts = NAMES[model_name]
    label = parts["size_label"] or size_label(*w.get_total_parameter_count())
    repository = f"https://huggingface.co/{pin['repository']}"
    w.add_name(pin["repository"].split("/")[1])
    w.add_organization(pin["repository"].split("/")[0])
    w.add_basename(parts["basename"])
    w.add_size_label(label)
    if parts["finetune"]:
        w.add_finetune(parts["finetune"])
    if parts["version"]:
        w.add_version(parts["version"])
    w.add_license(LICENSES[model_name])
    w.add_source_url(f"{repository}/tree/{pin['revision']}")
    w.add_source_repo_url(repository)
    w.add_file_type(LlamaFileType.ALL_F32)
    w.add_languages(sorted(LANGUAGES[model_name]))
    w.kv_data[0] = dict(sorted(w.kv_data[0].items(), key=lambda item: not item[0].startswith("general.")))

    path = os.path.join(out_dir, naming_convention(None, parts["basename"], parts["finetune"], parts["version"], label, "f32") + ".gguf")
    w.write_header_to_file(path)
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path
