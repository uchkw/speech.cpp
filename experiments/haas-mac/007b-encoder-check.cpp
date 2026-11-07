// Saves ordinary and retained Haas encoder tensors and records the attention path without timing it.
// Usage: 007b-encoder-check model.gguf device expected-fa-count output-directory [features.npy...]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "fastconformer/encoder.h"
#include "fastconformer/layout.h"
#include "fastconformer/transducer.h"
#include "json.h"
#include "npy.h"

namespace {

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

using Snapshot = std::map<std::string, std::vector<float>>;

Snapshot evaluate(const fastconformer::Encoder & encoder, const fastconformer::Joint & joint,
                  ggml_backend_t backend, const std::vector<float> & features, int64_t frames,
                  bool retained, int expected) {
    Graph graph;
    fastconformer::EncoderStages stages;
    auto * encoded = encoder.build(graph, features, frames, retained ? &stages : nullptr);
    auto * projected = joint.project_encoder(graph.ctx(), encoded);
    graph.output(projected);
    std::map<std::string, ggml_tensor *> tensors{{"projected", projected}};
    if (retained) {
        tensors.emplace("subsampled", stages.subsampled);
        tensors.emplace("encoded", encoded);
        for (size_t i = 0; i < stages.layers.size(); ++i) tensors.emplace("layer-" + std::to_string(i), stages.layers[i]);
        for (const auto & item : tensors) graph.output(item.second);
    }
    std::vector<ggml_tensor *> masks;
    int attention = 0;
    for (int i = 0; i < ggml_graph_n_nodes(graph.cgraph()); ++i) {
        const auto * node = ggml_graph_node(graph.cgraph(), i);
        if (node->op != GGML_OP_FLASH_ATTN_EXT) continue;
        ++attention;
        const int64_t t = encoder.subsampled_frames(frames);
        const auto * mask = node->src[3];
        require(node->src[0]->ne[0] == 128 && node->src[0]->ne[1] == t && node->src[0]->ne[2] == 8 &&
                    node->src[0]->ne[3] == 1 && node->src[0]->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32 &&
                    node->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(node->src[1]) &&
                    ggml_is_contiguous(node->src[2]) && mask && mask->type == GGML_TYPE_F16 &&
                    mask->ne[0] == t && mask->ne[1] == t && mask->ne[2] == 8 && mask->ne[3] == 1 &&
                    ggml_is_contiguous(mask), "the actual flash-attention shape, type or head-dependent mask differs");
        if (retained) {
            auto * readable = ggml_cast(graph.ctx(), node->src[3], GGML_TYPE_F32);
            masks.push_back(readable);
            graph.output(readable);
        }
    }
    require(attention == expected, "the actual flash-attention count differs from the requested path");
    std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocator(
        ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)), ggml_gallocr_free);
    require(bool(allocator), "cannot allocate the encoder graph");
    graph.compute(backend, allocator.get());
    for (size_t i = 0; i < masks.size(); ++i) {
        const auto values = Graph::read(masks[i]);
        float maximum = 0;
        size_t zero = 0, subnormal = 0;
        for (float value : values) {
            require(std::isfinite(value), "a real encoder F16 relative-score mask is not finite");
            maximum = std::max(maximum, std::fabs(value));
            zero += value == 0;
            subnormal += value != 0 && std::fabs(value) < 0x1p-14f;
        }
        std::printf("{\"mask_layer\":%zu,\"max_abs\":%.9g,\"zero\":%zu,\"subnormal\":%zu}\n",
                    i, maximum, zero, subnormal);
    }
    Snapshot result;
    for (const auto & item : tensors) {
        auto values = Graph::read(item.second);
        for (float value : values) require(std::isfinite(value), "an encoder tensor is not finite");
        result.emplace(item.first, std::move(values));
    }
    std::printf("{\"frames\":%lld,\"retained\":%s,\"flash_attention\":%d,\"expected\":%d}\n",
                (long long) frames, retained ? "true" : "false", attention, expected);
    return result;
}

void save(const std::filesystem::path & directory, const Snapshot & snapshot, int64_t t) {
    std::filesystem::create_directories(directory);
    for (const auto & item : snapshot) {
        const auto path = directory / (item.first + ".f32");
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char *>(item.second.data()), item.second.size()*sizeof(float));
        file.close();
        require(bool(file), "cannot save the encoder comparison tensor");
        require(item.second.size() % (size_t) t == 0, "an encoder tensor has a partial frame");
        const size_t width = item.second.size() / (size_t) t;
        std::printf("{\"tensor\":%s,\"frames\":%lld,\"width\":%zu,\"channel_argmax\":[",
                    json_string(path.u8string()).c_str(), (long long) t, width);
        for (int64_t i = 0; i < t; ++i) {
            const auto first = item.second.begin() + i*width;
            const auto argmax = std::max_element(first, first + width) - first;
            std::printf("%s%zu", i ? "," : "", (size_t) argmax);
        }
        std::printf("]}\n");
    }
}

// Preserves the two failing projected snapshots and prints one compact stderr evidence record.
// Best effort only: any problem here is reported on stderr and never replaces the original error,
// and the shared save helper is unsuitable because its whole-frame guard and success records
// would relabel or hide this failure.
void preserve_projected_failure(const std::filesystem::path & directory, const std::string & name, int64_t frames,
                                const std::vector<float> & ordinary, const std::vector<float> & retained) {
    try {
        std::filesystem::create_directories(directory);
        const auto write = [&](const char * file_name, const std::vector<float> & values) {
            // Raw host F32 bytes with no shape assumption, so an irregular extent cannot block the dump.
            std::ofstream file(directory / file_name, std::ios::binary);
            file.write(reinterpret_cast<const char *>(values.data()), values.size()*sizeof(float));
            file.close();
            if (!file) std::fprintf(stderr, "{\"evidence_save_failed\":\"%s\"}\n", file_name);
        };
        write("normal-projected.f32", ordinary);
        write("retained-projected.f32", retained);

        // Only the shared span is compared, so neither vector is read past its end.
        const size_t shared = std::min(ordinary.size(), retained.size());
        size_t different = 0;
        size_t first = shared;
        double maximum = 0;
        for (size_t i = 0; i < shared; ++i) {
            // Byte equality also separates values with different raw representations, such as +0.0 and -0.0.
            if (std::memcmp(&ordinary[i], &retained[i], sizeof(float)) == 0) continue;
            ++different;
            if (first == shared) first = i;
            maximum = std::max(maximum, std::fabs((double) ordinary[i] - (double) retained[i]));
        }
        // A channel width only exists when the element count divides into whole output frames.
        const auto width = [frames](size_t count) {
            return frames > 0 && count % (size_t) frames == 0 ? std::to_string(count / (size_t) frames)
                                                              : std::string("null");
        };
        std::fprintf(stderr,
            "{\"evidence\":\"projected-mismatch\",\"input\":%s,\"frames\":%lld,"
            "\"normal_elements\":%zu,\"retained_elements\":%zu,\"normal_width\":%s,\"retained_width\":%s,"
            "\"extent_mismatch\":%s,\"normal_extra_elements\":%zu,\"retained_extra_elements\":%zu,"
            "\"different_raw_elements\":%zu,\"first_differing_element\":%lld,\"max_abs_difference\":%.17g,"
            "\"failure_directory\":%s}\n",
            json_string(name).c_str(), (long long) frames, ordinary.size(), retained.size(), width(ordinary.size()).c_str(),
            width(retained.size()).c_str(), ordinary.size() == retained.size() ? "false" : "true",
            ordinary.size() - shared, retained.size() - shared, different,
            first == shared ? -1LL : (long long) first, maximum, json_string(directory.u8string()).c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "{\"evidence_error\":%s}\n", json_string(error.what()).c_str());
    }
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const auto args = utf8_args(argc, argv);
        require(args.size() >= 5, "usage: 007b-encoder-check model.gguf device expected-fa-count output-directory [features.npy...]");
        const int expected = std::stoi(args[3]);
        require(expected == 0 || expected == 24, "expected FA count must be 0 or 24");
        require(!std::getenv("GGML_METAL_FUSION_DEBUG") && !std::getenv("GGML_METAL_FUSION_DISABLE"),
                "this check forbids fusion debug or disabling generic fusion");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(start_device(find_device(args[2]), 4), ggml_backend_free);
        const ModelFile model(args[1], backend.get(), fastconformer::layout);
        const fastconformer::Encoder encoder(model, backend.get());
        const fastconformer::Joint joint(model);
        const int mels = model.size("fastconformer.frontend.n_mels");
        const auto destination = std::filesystem::u8path(args[4]);
        auto check = [&](const std::vector<float> & features, int64_t frames, const std::string & name) {
            const auto normal = evaluate(encoder, joint, backend.get(), features, frames, false, expected);
            const auto retained = evaluate(encoder, joint, backend.get(), features, frames, true, expected);
            const auto & a = normal.at("projected");
            const auto & b = retained.at("projected");
            if (a.size() != b.size() || std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) != 0) {
                preserve_projected_failure(destination / name / "failure", name,
                                           encoder.subsampled_frames(frames), a, b);
                throw std::runtime_error("retaining layer outputs changes the ordinary projected encoder output");
            }
            const auto t = encoder.subsampled_frames(frames);
            save(destination / name / "normal", normal, t);
            save(destination / name / "retained", retained, t);
        };
        if (args.size() == 5) {
            for (int64_t t : {1, 16, 17, 19, 20, 31, 32, 33, 63, 64, 65, 171, 172, 375, 376, 384}) {
                const int64_t frames = t*encoder.subsampling_factor() - 1;
                std::vector<float> features((size_t) frames*mels);
                for (size_t i = 0; i < features.size(); ++i) {
                    features[i] = std::sin((float) i*0.013f) + 0.1f*std::cos((float) i*0.037f);
                }
                check(features, frames, "synthetic-" + std::to_string(t));
            }
        } else {
            for (size_t i = 5; i < args.size(); ++i) {
                const auto features = read_npy(args[i]);
                require(features.shape.size() == 2 && features.shape[1] == mels && features.shape[0] > 0,
                        "features must have shape [positive frames, model mels]");
                check(features.f32, features.shape[0], "input-" + std::to_string(i - 5));
            }
        }
        std::printf("{\"passed\":true,\"official_reference_checked\":false,\"recognition_checked\":false,\"timing\":false}\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
