"""Converts a pinned NeMo FastConformer checkpoint with an RNN-T or TDT decoder to the one GGUF file the C++ port reads.

usage: uv run python convert.py <model> <out dir>

Writes parakeet-tdt_ctc-0.6B-ja-F32.gguf, parakeet-tdt-0.6B-v3-F32.gguf, reazonspeech-nemo-619M-v2-F32.gguf or
Haas-616M-ja-v1.0-F32.gguf, named under GGUF's naming convention, in layout 4: the frontend's window and mel filterbank,
the subsampling, the conformer layers, the prediction network, LSTM layers or Haas's Kimi Delta Attention layers, and
the joint, with the SentencePiece pieces the token ids name and which of them stand for bytes, the settings of the
decoding transcribe() runs (greedy TDT's durations and limit, or the beam and length of RNN-T's alignment-length
synchronous beam search) and, for RNN-T, the limit of the greedy decoding NeMo also runs on it, for Haas the settings of
Haas-ASR's custom vocabulary with what SentencePiece encodes a word with, every other constant the C++ reads and the
model's identity in the GGUF specification's general keys. A hybrid checkpoint's CTC head is left out,
since NeMo decodes with its transducer.

Tensor shapes follow ggml, whose ne[0] is the last numpy axis: a Linear weight [out, in] is stored as is
(ne = [in, out]). Every tensor is written in float32; `speech quantize` makes the other weight types of the file, each
tensor in the type src/families/fastconformer/layout.cpp gives it. The batch norm of each convolution
module is folded into its depthwise convolution, which it follows in evaluation, and each LSTM layer's two biases are
summed. A checkpoint whose conformer layers have no biases (ConformerEncoder's use_bias false) is written without them,
and fastconformer.encoder.use_bias says so. Haas runs in the environment of haas/ (pins.py).
"""

import argparse
import os

import torch
from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel, EncDecRNNTBPEModel

from gguf_writer import write_model
from pins import MODELS, restore


def decoder_settings(model_name, model):
    """Checks the restored official decoder and records the settings the GGUF layout stores."""
    cfg = model.cfg
    # The decoding transcribe() runs by default, which the C++ ports for each kind of transducer: TDT's greedy label
    # looping (greedy_batch, GreedyBatchedTDTInfer), or RNN-T's alignment-length synchronous beam search (alsd,
    # BeamRNNTInfer). Both feed the blank first as the prediction network's padding (its embedding zero, so the blank is
    # the start of the sequence) and use a ReLU joint. The joint's log-softmax, which NeMo applies on the CPU alone, is
    # left out; the beam search takes its own log-softmax over the tokens and the blank, which the C++ computes.
    # A model with a prompt (EncDecRNNTBPEModelWithPrompt and its hybrid) is told its language through an input the C++
    # does not have; a hybrid's CTC head is left aside while cur_decoder is "rnnt".
    dec, joint, decoding = model.decoder, model.joint, model.decoding
    assert type(model) in (EncDecRNNTBPEModel, EncDecHybridRNNTCTCBPEModel)
    assert type(model) is EncDecRNNTBPEModel or model.cur_decoder == "rnnt"
    decoder = {"GreedyBatchedTDTInfer": "tdt", "BeamRNNTInfer": "rnnt"}[type(decoding.decoding).__name__]
    if decoder == "tdt":
        assert decoding.cfg.strategy == "greedy_batch" and not decoding.cfg.get("big_blank_durations")
        assert type(decoding.decoding.decoding_computer).__name__ == "GreedyBatchedTDTLabelLoopingComputer"
        assert not decoding.decoding.decoding_computer.has_fusion_models()
        durations = [int(d) for d in decoding.cfg.durations]
    else:
        beam = decoding.decoding
        # A beam of 1 runs greedy_search() instead. The blank is the last class (index_incr 0), and a float target
        # length is a multiple of the frames.
        assert decoding.cfg.strategy == "alsd" and beam.search_algorithm == beam.align_length_sync_decoding
        assert beam.beam_size > 1 and beam.return_best_hypothesis and beam.softmax_temperature == 1.0
        assert beam.language_model is None and beam.ngram_lm is None and not beam.hat_subtract_ilm
        assert beam.blank == beam.vocab_size and isinstance(beam.alsd_max_target_length, float)
        # The greedy decoding a caller may choose instead, GreedyBatchedRNNTInfer or GreedyRNNTInfer, takes the most tokens
        # on one frame as RNNTDecoding reads it from the configuration's greedy settings; without a limit it would emit
        # tokens on one frame for as long as the joint gives them.
        greedy_max_symbols = decoding.cfg.greedy.get("max_symbols", None) or decoding.cfg.greedy.get("max_symbols_per_step", None)
        assert isinstance(greedy_max_symbols, int) and greedy_max_symbols > 0, greedy_max_symbols
        durations = []
    assert dec.blank_as_pad and dec.blank_idx == decoding.blank_id and not dec.random_state_sampling
    prediction = {"LSTMDropout": "lstm", "KDAPredictionNetwork": "kda"}[type(dec.prediction.dec_rnn).__name__]
    network = dec.prediction.dec_rnn
    if prediction == "lstm":
        # LSTMDropout is the plain LSTM rnn() makes without a normalization; its dropout is off in evaluation.
        assert network.lstm.proj_size == 0 and network.lstm.bias and not network.lstm.bidirectional
    else:
        # Haas's KDATransducerDecoder: pre-norm residual blocks of fla's KimiDeltaAttention with nn.LayerNorm, the blank as
        # the embedding's padding and no normalization of its own. Each layer as the C++ computes it: short convolutions
        # of q, k and v with a SiLU and no bias, a gate without a lower bound and beta through a sigmoid in the kernel,
        # as many value heads as heads, values as wide as keys, and an output norm gated by a sigmoid.
        assert type(dec).__name__ == "KDATransducerDecoder" and cfg.decoder.get("normalization_mode") is None
        assert len(network.layers) == len(network.norms) == dec.pred_rnn_layers
        first = network.layers[0]
        for layer, norm in zip(network.layers, network.norms):
            assert type(layer).__name__ == "KimiDeltaAttention" and layer.use_short_conv and not layer.conv_bias
            assert not layer.allow_neg_eigval and not layer.safe_gate and layer.lower_bound is None
            assert layer.num_v_heads == layer.num_heads and layer.head_v_dim == layer.head_k_dim == layer.head_dim
            assert layer.f_proj[0].out_features == layer.g_proj[0].out_features == layer.head_v_dim
            assert layer.f_proj[0].bias is None and layer.f_proj[1].bias is None and layer.g_proj[0].bias is None
            assert layer.g_proj[1].bias is not None and layer.b_proj.bias is None and layer.o_proj.bias is None
            assert all(c.activation == "silu" and c.bias is None for c in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d))
            assert layer.o_norm.activation == "sigmoid" and layer.o_norm.weight is not None and layer.o_norm.bias is None
            assert (layer.num_heads, layer.head_dim, layer.conv_size, layer.o_norm.eps) == (
                first.num_heads, first.head_dim, first.conv_size, first.o_norm.eps)
            assert type(norm) is torch.nn.LayerNorm and norm.elementwise_affine and norm.eps == network.norms[0].eps
        assert network.final_norm is None or (type(network.final_norm) is torch.nn.LayerNorm and network.final_norm.eps == network.norms[0].eps)
    assert not dec.is_adapter_available() and not joint.is_adapter_available()
    assert float(dec.prediction.embed.weight[dec.blank_idx].abs().max()) == 0
    assert joint.activation == "relu" and joint.temperature == 1.0
    assert joint.num_extra_outputs == len(durations) == len(decoding.durations or [])
    assert joint.joint_net[-1].out_features == decoding.blank_id + 1 + len(durations)

    settings = {
        "kind": decoder, "prediction": prediction, "blank": int(decoding.blank_id),
        "layers": int(dec.pred_rnn_layers), "durations": durations,
        "max_symbols": int(decoding.decoding.max_symbols) if decoder == "tdt" else greedy_max_symbols,
        "separators": list(decoding.segment_seperators), "punctuation": sorted(decoding.supported_punctuation or []),
    }
    if prediction == "kda":
        settings["kda"] = {
            "num_heads": int(first.num_heads), "norm_eps": float(network.norms[0].eps),
            "gate_norm_eps": float(first.o_norm.eps), "qk_norm_eps": 1e-6,
            "final_norm": network.final_norm is not None,
        }
    if decoder == "rnnt":
        settings.update(beam_size=int(beam.beam_size), score_norm=bool(beam.score_norm),
                        max_target_ratio=float(beam.alsd_max_target_length))
    if MODELS[model_name].get("environment") == "haas":
        import inspect

        from haas_asr import boosting
        from haas_asr.vocabulary import Vocabulary
        from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig

        defaults = BoostingTreeModelConfig()
        assert not defaults.uniform_weights
        settings["vocabulary"] = {
            "context_score": float(boosting.CONTEXT_SCORE), "depth_scaling": float(boosting.DEPTH_SCALING),
            "unk_score": float(defaults.unk_score),
            "alpha": float(inspect.signature(Vocabulary).parameters["alpha"].default),
            "speech_rms": float(boosting.SPEECH_RMS), "speech_frame": int(boosting.SPEECH_FRAME),
        }
    # NeMo before 3.0 has no language-tag stripping option and strips none.
    assert not getattr(decoding, "strip_lang_tags", False)
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("out_dir")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model = restore(MODELS[args.model])
    settings = decoder_settings(args.model, model)
    tokenizer = model.tokenizer
    assert not tokenizer.legacy and tokenizer.vocab_size == tokenizer.tokenizer.vocab_size()
    path = write_model(args.model, args.out_dir, model.cfg, model.preprocessor.featurizer, model.encoder,
                       model.state_dict(), tokenizer.tokenizer, settings)
    print("wrote", path)


if __name__ == "__main__":
    main()
