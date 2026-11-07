#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "boosting.h"
#include "transducer.h"

namespace fastconformer {

class KeywordPrefix;

/** The tokens a decoding emitted, in order, with the encoder frames it emitted them on. */
struct Decoding {
    std::vector<int32_t> ids;
    /** The encoder frame each token was emitted on. */
    std::vector<int64_t> frames;
    /** The duration in frames TDT predicted with each token; empty for a decoding that predicts none. */
    std::vector<int32_t> durations;
    /** The graphs the decoding computed. Each is a round trip to the backend, so that on a GPU its time grows with them. */
    size_t graphs = 0;
};

/**
 * Told the fraction of the encoder's frames a decoding has passed, as it goes; answering false stops the decoding,
 * which returns the tokens it has found.
 */
using DecodingProgress = std::function<bool(double done)>;

/**
 * The words a greedy decoding boosts, as NeMo's label-looping decoding fuses a boosting tree: wherever the decoding
 * emits a token rather than a blank, which it decides without the words, it emits instead the token whose logit plus
 * alpha times its bonus from the tree's state is largest, and each token it emits moves the tree's state.
 */
struct Boost {
    const BoostingTree & tree;
    float alpha;
};

/** A decoding of the transducer's outputs into tokens. */
class Decoder {
public:
    virtual ~Decoder() = default;

    /**
     * The tokens of `projected`, the encoder's output through Joint::project_encoder(), [T, hidden] row-major, telling
     * `progress`, when it is given, how far the decoding has come.
     */
    virtual Decoding decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const = 0;

    /** The same with `boost`'s words boosted, by a decoding that boosts words (TDT's); any other throws. */
    virtual Decoding decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress,
                            const Boost & boost) const;

    /** Records the bounded prediction prefix for a subsequent boosted decoding; only TDT supports this path. */
    virtual Decoding record_keyword_prefix(const std::vector<float> & projected, ggml_backend_t backend,
                                          const DecodingProgress & progress, KeywordPrefix & prefix) const;
    /** Reuses a recorded prediction prefix until the first different token ID; joint decisions are computed again. */
    virtual Decoding decode_keyword_prefix(const std::vector<float> & projected, ggml_backend_t backend,
                                          const DecodingProgress & progress, const Boost & boost, const KeywordPrefix & prefix) const;

    virtual int blank() const = 0;
};

/**
 * The decodings of a model with the file `m`, by the names of the request option decoding, the one NeMo's transcribe()
 * runs by default first: "greedy" for fastconformer.decoder.kind "tdt" (TdtDecoder), and "beam", the beam search its
 * checkpoint configures (AlsdDecoder), and "greedy" (RnntGreedyDecoder) for "rnnt".
 */
std::vector<std::string> decodings(const ModelFile & m);

/** The decoding of decodings(m) named `name`. */
std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint, const std::string & name);

/** The index of the largest of `n` values, the first of equal ones, as torch.max() and argmax() choose it. */
int first_argmax(const float * v, int n);

}  // namespace fastconformer
