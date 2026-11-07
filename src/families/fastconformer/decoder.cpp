#include "decoder.h"

#include <stdexcept>
#include <string>

#include "alsd.h"
#include "rnnt-greedy.h"
#include "tdt.h"

namespace fastconformer {

namespace {

template <typename D>
std::unique_ptr<Decoder> make(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint) {
    return std::make_unique<D>(m, prediction, joint);
}

/** A decoding of the transducers of one fastconformer.decoder.kind, under its name. */
struct Offered {
    const char * kind;
    const char * name;
    std::unique_ptr<Decoder> (*make)(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint);
};

/** The decodings of each kind, the one NeMo's transcribe() runs by default first. */
const Offered offered[] = {
    {"tdt", "greedy", make<TdtDecoder>},
    {"rnnt", "beam", make<AlsdDecoder>},
    {"rnnt", "greedy", make<RnntGreedyDecoder>},
};

std::string kind_of(const ModelFile & m) {
    return m.one_of("fastconformer.decoder.kind", {"tdt", "rnnt"});
}

}  // namespace

std::vector<std::string> decodings(const ModelFile & m) {
    const std::string kind = kind_of(m);
    std::vector<std::string> names;
    for (const Offered & o : offered) {
        if (kind == o.kind) names.push_back(o.name);
    }
    return names;
}

std::unique_ptr<Decoder> make_decoder(const ModelFile & m, const PredictionNetwork & prediction, const Joint & joint, const std::string & name) {
    const std::string kind = kind_of(m);
    for (const Offered & o : offered) {
        if (kind == o.kind && name == o.name) return o.make(m, prediction, joint);
    }
    throw std::logic_error("a FastConformer model with a " + kind + " decoder has no decoding named " + name);
}

Decoding Decoder::decode(const std::vector<float> &, ggml_backend_t, const DecodingProgress &, const Boost &) const {
    throw std::logic_error("this decoding boosts no words");
}

Decoding Decoder::decode(const std::vector<float> &, ggml_backend_t, const DecodingProgress &, TdtPrefixCache::Reading &) const {
    throw std::logic_error("this decoding caches no prediction prefix");
}

int first_argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

}  // namespace fastconformer
