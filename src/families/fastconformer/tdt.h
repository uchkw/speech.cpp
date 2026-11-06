#pragma once

#include <cstdint>
#include <vector>

#include "decoder.h"

namespace fastconformer {

/**
 * NeMo's greedy TDT decoding as transcribe() runs it, GreedyBatchedTDTLabelLoopingComputer.torch_impl()
 * (nemo/collections/asr/parts/submodules/transducer_decoding/tdt_label_looping.py) for one utterance.
 *
 * From frame 0 with the blank fed to the prediction network, each step evaluates the joint at the current frame:
 * the label is the argmax of the tokens and the blank, the duration the argmax of the durations' outputs, and
 * the frame advances by the duration, by 1 at least for a blank. Blanks are passed over with the same
 * prediction until a token is found or the frames run out. A token is emitted, even when its duration ends the
 * utterance, and fed to the prediction network for the next step; after max_symbols tokens on one frame whose
 * last duration is 0, the frame advances by 1. Each token keeps the frame it was found on and the duration predicted
 * with it, the time index and the token duration the decoding stores for timestamps.
 */
class TdtDecoder : public Decoder {
public:
    TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint);
    TdtDecoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint, bool reuse_joint);

    Decoding decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress) const override;
    Decoding decode(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress,
                    const Boost & boost) const override;

    int blank() const override { return blank_; }

private:
    Decoding run(const std::vector<float> & projected, ggml_backend_t backend, const DecodingProgress & progress, const Boost * boost) const;

    const PredictionNetwork & prediction_;
    const Joint & joint_;
    int blank_;
    std::vector<int32_t> durations_;
    uint32_t max_symbols_;
    bool reuse_joint_;
};

}  // namespace fastconformer
