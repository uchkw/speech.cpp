// Checks Haas's query-dependent relative-score mask and F32 FA_EXT against an independent scalar formula.
// Usage: 007b-attention-check <gpu|cpu|device-name>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "graph.h"

namespace {

constexpr int D = 128;
constexpr int H = 8;

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct Inputs {
    int t;
    std::vector<float> q, k, v, p, u, b;

    Inputs(int length, int amplitude) : t(length), q(D*H*t), k(q.size()), v(q.size()), p(D*H*(2*t - 1)), u(D*H), b(D*H) {
        auto fill = [](std::vector<float> & values, int seed, int scale) {
            for (size_t i = 0; i < values.size(); ++i) {
                const int c = i % D, h = (i / D) % H, pos = i / (D*H);
                values[i] = (float) (((17*c + 31*pos + 13*h + seed) % 127) - 63)*scale/128.0f;
            }
        };
        fill(q, 5, amplitude);
        fill(k, 19, 1);
        fill(v, 43, 1);
        fill(p, 71, 1);
        for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d) {
            u[h*D + d] = (float) ((h + 1)*(d % 5 - 2))/64;
            b[h*D + d] = (float) ((H - h)*(d % 7 - 3))/64;
        }
    }

    size_t at(int pos, int head, int channel) const { return (pos*H + head)*D + channel; }

    float position(int query, int key, int head, int control = 0) const {
        const int ph = control == 1 ? 0 : head;
        const int pi = control == 2 ? t - 1 + query - key : t - 1 - query + key;
        double dot = 0;
        for (int d = 0; d < D; ++d) dot += (q[at(query, ph, d)] + b[ph*D + d])*p[at(pi, ph, d)];
        const float scaled = (float) dot*(1.0f/std::sqrt((float) D));
        return ggml_fp16_to_fp32(ggml_fp32_to_fp16(scaled));
    }

    std::vector<float> reference(int control = 0) const {
        std::vector<float> output(q.size());
        std::vector<double> scores(t);
        const float scale = 1.0f/std::sqrt((float) D);
        for (int h = 0; h < H; ++h) for (int i = 0; i < t; ++i) {
            for (int j = 0; j < t; ++j) {
                double ac = 0;
                for (int d = 0; d < D; ++d) ac += (q[at(i, h, d)] + u[h*D + d])*k[at(j, h, d)];
                scores[j] = (float) ac*scale + position(i, j, h, control);
            }
            const double maximum = *std::max_element(scores.begin(), scores.end());
            double sum = 0;
            for (double & value : scores) sum += value = std::exp(value - maximum);
            for (int d = 0; d < D; ++d) {
                double value = 0;
                for (int j = 0; j < t; ++j) value += scores[j]*v[at(j, h, d)];
                output[at(i, h, d)] = (float) (value/sum);
            }
        }
        return output;
    }
};

double difference(const char * label, const std::vector<float> & got, const std::vector<float> & wanted) {
    require(got.size() == wanted.size(), "attention output shapes differ");
    double signal = 0, noise = 0, maximum = 0;
    size_t changed = 0, argmax = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        require(std::isfinite(got[i]) && std::isfinite(wanted[i]), "attention contains a nonfinite value");
        const double e = (double) got[i] - wanted[i];
        signal += (double) wanted[i]*wanted[i];
        noise += e*e;
        maximum = std::max(maximum, std::fabs(e));
        changed += got[i] != wanted[i];
    }
    for (size_t i = 0; i < got.size(); i += D) {
        const auto a = std::max_element(got.begin() + i, got.begin() + i + D) - got.begin();
        const auto b = std::max_element(wanted.begin() + i, wanted.begin() + i + D) - wanted.begin();
        argmax += a != b;
    }
    const double nmse = signal == 0 ? noise : noise/signal;
    std::printf("%s: NMSE %.9g SNR %.3f dB max_abs %.9g changed %zu channel_argmax_changed %zu\n",
                label, nmse, noise == 0 ? INFINITY : 10*std::log10(signal/noise), maximum, changed, argmax);
    return nmse;
}

void check(ggml_backend_t backend, int t, int amplitude) {
    const Inputs input(t, amplitude);
    Graph graph;
    auto * ctx = graph.ctx();
    auto heads = [&](ggml_tensor * value) { return ggml_permute(ctx, value, 0, 2, 1, 3); };
    auto * q = graph.input(input.q, D, H, t);
    auto * qu = heads(ggml_add(ctx, q, graph.input(input.u, D, H)));
    auto * qv = heads(ggml_add(ctx, q, graph.input(input.b, D, H)));
    auto * k = ggml_cont(ctx, heads(graph.input(input.k, D, H, t)));
    auto * v = ggml_cont(ctx, heads(graph.input(input.v, D, H, t)));
    auto * p = heads(graph.input(input.p, D, H, 2*t - 1));
    auto * bd = mul_mat(ctx, p, qv);
    bd = ggml_view_3d(ctx, bd, t, t, H, (size_t) (2*t - 2)*sizeof(float), bd->nb[2], (size_t) (t - 1)*sizeof(float));
    auto * mask = ggml_cast(ctx, ggml_scale(ctx, ggml_cont(ctx, bd), 1.0f/std::sqrt((float) D)), GGML_TYPE_F16);
    auto * output = ggml_flash_attn_ext(ctx, qu, k, v, mask, 1.0f/std::sqrt((float) D), 0.0f, 0.0f);
    ggml_prec_set_acc(output, GGML_PREC_F32);
    require(ggml_backend_supports_op(backend, output), "backend does not support the F32 attention fixture");
    graph.output(output);
    auto * readable_mask = ggml_cast(ctx, mask, GGML_TYPE_F32);
    graph.output(readable_mask);
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(bool(allocator), "cannot allocate the attention fixture");
    graph.compute(backend, allocator.get());
    const auto observed_mask = Graph::read(readable_mask);
    float max_mask = 0;
    for (int h = 0; h < H; ++h) for (int i = 0; i < t; ++i) for (int j = 0; j < t; ++j) {
        const float value = observed_mask[(h*t + i)*t + j];
        require(std::isfinite(value), "the relative-score mask is not finite");
        require(value == input.position(i, j, h), "the head-specific relative shift or F16 mask value differs");
        max_mask = std::max(max_mask, std::fabs(value));
    }
    std::printf("T=%d amplitude=%d max_abs_mask=%g backend=%s\n", t, amplitude, max_mask, ggml_backend_name(backend));
    const auto reference = input.reference();
    // The pinned ggml FLASH_ATTN_EXT check uses this NMSE bound; the encoder's official bounds stay unchanged.
    require(difference("flash/scalar", Graph::read(output), reference) < 5e-4, "flash attention exceeds ggml's existing NMSE bound");
    if (t == 33 && amplitude == 1) {
        require(difference("shared-head control", input.reference(1), reference) > 5e-4,
                "the fixture does not distinguish a broadcast head mask");
        require(difference("reversed-position control", input.reference(2), reference) > 5e-4,
                "the fixture does not distinguish a reversed relative position");
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        require(args.size() == 2, "usage: 007b-attention-check <gpu|cpu|device-name>");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[1]), 4), ggml_backend_free);
        for (int t : {1, 16, 17, 19, 20, 31, 32, 33, 63, 64, 65, 171, 172, 375, 376, 384}) check(backend.get(), t, 1);
        for (int t : {19, 20, 33, 376}) check(backend.get(), t, 8);
        std::printf("passed: synthetic attention arithmetic; official encoder and recognition quality are not checked\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
