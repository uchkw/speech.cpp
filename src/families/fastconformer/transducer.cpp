#include "transducer.h"

#include <string>

#include "kda.h"

namespace fastconformer {

namespace {

ggml_tensor * linear(ggml_context * ctx, const ModelFile & m, ggml_tensor * x, const std::string & name) {
    return ggml_add(ctx, mul_mat(ctx, m.tensor(name + ".weight"), x), m.tensor(name + ".bias"));
}

/**
 * Stacked torch.nn.LSTM layers, whose state is h of each layer, then c of each layer, [hidden] each. A layer's step:
 * gates = W_ih x + W_hh h + b, stacked as input, forget, cell and output; c' = sigmoid(f) c + sigmoid(i) tanh(g) and
 * h' = sigmoid(o) tanh(c'), which the next layer takes. The converter sums b_ih and b_hh.
 */
class LstmLayers : public PredictionLayers {
public:
    explicit LstmLayers(const ModelFile & m)
        : m_(m), layers_((int) m.u32("fastconformer.decoder.prediction_layers")), hidden_(m.width("pred.embed.weight", 0)) {}

    int64_t state_size() const override { return 2 * hidden_ * layers_; }

    ggml_tensor * build(ggml_context * ctx, ggml_tensor * x, ggml_tensor * state, ggml_tensor ** next) const override {
        const size_t width = (size_t) hidden_ * sizeof(float);
        ggml_tensor * h_out = nullptr;
        ggml_tensor * c_out = nullptr;
        for (int l = 0; l < layers_; l++) {
            const std::string p = "pred.lstm." + std::to_string(l);
            ggml_tensor * h = ggml_view_1d(ctx, state, hidden_, (size_t) l * width);
            ggml_tensor * c = ggml_view_1d(ctx, state, hidden_, (size_t) (layers_ + l) * width);
            ggml_tensor * gates = ggml_add(ctx, ggml_add(ctx, mul_mat(ctx, m_.tensor(p + ".ih.weight"), x), mul_mat(ctx, m_.tensor(p + ".hh.weight"), h)),
                                           m_.tensor(p + ".bias"));
            auto gate = [&](int i) { return ggml_view_1d(ctx, gates, hidden_, (size_t) i * width); };
            ggml_tensor * next_c =
                ggml_add(ctx, ggml_mul(ctx, ggml_sigmoid(ctx, gate(1)), c), ggml_mul(ctx, ggml_sigmoid(ctx, gate(0)), ggml_tanh(ctx, gate(2))));
            ggml_tensor * next_h = ggml_mul(ctx, ggml_sigmoid(ctx, gate(3)), ggml_tanh(ctx, next_c));
            h_out = h_out ? ggml_concat(ctx, h_out, next_h, 0) : next_h;
            c_out = c_out ? ggml_concat(ctx, c_out, next_c, 0) : next_c;
            x = next_h;
        }
        *next = ggml_concat(ctx, h_out, c_out, 0);
        return x;
    }

    std::vector<std::string> matrices() const override {
        std::vector<std::string> names;
        for (int l = 0; l < layers_; l++) {
            for (const char * part : {".ih.weight", ".hh.weight"}) names.push_back("pred.lstm." + std::to_string(l) + part);
        }
        return names;
    }

private:
    const ModelFile & m_;
    int layers_;
    int64_t hidden_;
};

}  // namespace

std::unique_ptr<PredictionLayers> make_prediction_layers(const ModelFile & m, ggml_backend_t backend, bool cache_constants) {
    if (m.one_of("fastconformer.decoder.prediction", {"lstm", "kda"}) == "kda") return std::make_unique<KdaLayers>(m, backend, cache_constants);
    return std::make_unique<LstmLayers>(m);
}

bool decoding_tensor(const std::string & name) {
    for (const char * prefix : {"pred.", "joint.pred.", "joint.out."}) {
        if (name.rfind(prefix, 0) == 0) return true;
    }
    return false;
}

bool quantized_decoding(const ModelFile & m) {
    std::vector<std::string> names = make_prediction_layers(m)->matrices();
    names.insert(names.end(), {"joint.pred.weight", "joint.out.weight"});
    for (const std::string & name : names) {
        if (!ggml_is_quantized(m.stored_type(name))) return false;
    }
    return true;
}

PredictionNetwork::PredictionNetwork(const ModelFile & m, ggml_backend_t backend, bool cache_constants)
    : m_(m), layers_(make_prediction_layers(m, backend, cache_constants)), hidden_((int) m.tensor("pred.embed.weight")->ne[0]) {}

PredictionNetwork::Step PredictionNetwork::build(Graph & g, int32_t label, const PredictionState & state) const {
    return build(g, g.input(std::vector<int32_t>{label}, 1), g.input(state, (int64_t) state.size()));
}

PredictionNetwork::Step PredictionNetwork::build(Graph & g, ggml_tensor * label, ggml_tensor * state) const {
    ggml_context * ctx = g.ctx();
    ggml_tensor * next = nullptr;
    ggml_tensor * output = layers_->build(ctx, ggml_get_rows(ctx, m_.tensor("pred.embed.weight"), label), state, &next);
    return {output, next};
}

Joint::Joint(const ModelFile & m)
    : m_(m), hidden_((int) m.tensor("joint.out.weight")->ne[0]), outputs_((int) m.tensor("joint.out.weight")->ne[1]) {}

ggml_tensor * Joint::project_encoder(ggml_context * ctx, ggml_tensor * encoded) const {
    return linear(ctx, m_, encoded, "joint.enc");
}

ggml_tensor * Joint::project_prediction(ggml_context * ctx, ggml_tensor * prediction) const {
    return linear(ctx, m_, prediction, "joint.pred");
}

ggml_tensor * Joint::build(ggml_context * ctx, ggml_tensor * f, ggml_tensor * g) const {
    return linear(ctx, m_, ggml_relu(ctx, ggml_add(ctx, f, g)), "joint.out");
}

}  // namespace fastconformer
