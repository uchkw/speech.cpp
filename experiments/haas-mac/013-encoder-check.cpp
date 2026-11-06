// The normal encoder check of trial 013: the ordinary recognition encoder graph on one allocator, for the
// byte-for-byte comparison of the original-kernel, candidate-off and candidate-on binaries of a future build.
// It builds the graph exactly as fastconformer::Recognizer::encode does, Encoder::build followed by
// Joint::project_encoder with the final projected tensor as the only output; any other output or retained
// intermediate would change what the allocator places where, which is what this check observes. One ordinary
// graph allocator serves the whole fixed sequence of subsampled lengths 1, 31, 32, 33, 171, 172, 375, 33, 1, so
// it grows, shrinks and repeats, and the repeated lengths must come out byte-identical from the same bytes in.
// It initializes no fusion statistics, sets no command-buffer property and leaves the command buffer's default
// partitioning alone; a process that sets GGML_METAL_FUSION_DEBUG or GGML_METAL_FUSION_DISABLE is refused,
// because either would change what the backend fuses or logs. Synthetic features say nothing about recognition
// quality or any official reference; they check allocation and arithmetic stability alone.
//
// The output file holds, in native byte order: the 8 bytes "sp013enc", a u32 format version, a u32 subsampling
// factor, mels, d_model, joint hidden, conformer layers and case count, then per case a u32 index, feature
// frames, intended subsampled length and actual length, a u64 nonzero count and a u64 element count, followed
// by the projected output's elements as float32 in ggml order, hidden fastest.
//
// usage: 013-encoder-check <model.gguf> <output.bin>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "args.h"
#include "backend.h"
#include "error.h"
#include "fastconformer/encoder.h"
#include "fastconformer/layout.h"
#include "fastconformer/transducer.h"
#include "graph.h"
#include "model-file.h"
#include "weight-types.h"

using namespace fastconformer;

namespace {

/** The subsampled lengths the check runs, in order: growth, a boundary around 32 and 172, one long input, then
    two lengths again, which must reproduce their first bytes exactly. */
const int64_t LENGTHS[] = {1, 31, 32, 33, 171, 172, 375, 33, 1};

/** Finite synthetic features of [frames, mels]: one fixed-seed xorshift stream over exactly that many values,
    each in [-2, 2), so that the frame count alone decides the bytes the encoder reads. */
std::vector<float> synthetic_features(int64_t frames, int64_t mels) {
    std::vector<float> features((size_t) (frames * mels));
    uint32_t state = 0x01309ecd;
    for (float & v : features) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        v = (float) (int32_t) state / 1073741824.0f;
    }
    return features;
}

/** The least feature frames that sub-sample to `t`: each halving of Encoder::subsampled_frames() inverted once,
    checked against the encoder itself so that the intended length is the one the graph really produces. */
int64_t frames_for(const Encoder & encoder, int64_t t) {
    int64_t frames = t;
    for (int64_t factor = encoder.subsampling_factor(); factor > 1; factor /= 2) frames = frames * 2 - 1;
    if (encoder.subsampled_frames(frames) != t) {
        throw Error(Fault::File, "the model's subsampling does not give the lengths this check takes");
    }
    return frames;
}

void write_u32(FILE * f, uint32_t v) {
    if (std::fwrite(&v, sizeof v, 1, f) != 1) throw Error(Fault::Io, "cannot write the check's output file");
}

void write_u64(FILE * f, uint64_t v) {
    if (std::fwrite(&v, sizeof v, 1, f) != 1) throw Error(Fault::Io, "cannot write the check's output file");
}

}  // namespace

/** The check, shared with the graph-debug launcher: the ordinary binary reaches it through main below, and
    013-graph-debug-main.cpp reaches it with this file compiled with SPEECH_013_ENCODER_CHECK_NO_MAIN defined,
    which renames this file's main away. */
int speech_013_encoder_check_main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() != 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <output.bin>\n",
                     args.empty() ? "013-encoder-check" : args[0].c_str());
        return 2;
    }
    for (const char * setting : {"GGML_METAL_FUSION_DEBUG", "GGML_METAL_FUSION_DISABLE"}) {
        if (std::getenv(setting) != nullptr) {
            std::fprintf(stderr, "error: %s is set; it changes what the Metal backend fuses or logs, so this "
                                 "comparison would say nothing. Unset it and run the check again.\n",
                         setting);
            return 1;
        }
    }
    const char * dw_silu = std::getenv("GGML_METAL_DW_SILU");
    std::printf("dw-silu: %s\n", dw_silu != nullptr ? dw_silu : "unset (off)");
    if (std::getenv("GGML_METAL_GRAPH_DEBUG") != nullptr) {
        std::printf("graph-debug: set; the default sink drops Debug messages, so the node trace needs the "
                    "013-graph-debug launcher\n");
    }
    try {
        if (!takes_weights(args[1], layout, {"Q4_K", "Q8_0", "F16"},
                           "the trial fuses the F32 depthwise tensors that these files hold alike")) {
            return 2;
        }
        {
            const ModelFile metadata(args[1], layout);
            const bool haas = metadata.u32("fastconformer.encoder.num_layers") == 24 &&
                              metadata.one_of("fastconformer.decoder.prediction", {"lstm", "kda"}) == "kda";
            if (!haas) {
                std::fprintf(stderr, "error: %s is not Haas's FastConformer of 24 conformer layers with the KDA "
                                     "prediction network\n",
                             args[1].c_str());
                return 2;
            }
        }
        // This check observes what Metal fuses alone, so it takes the device whose name is MTL0 (ggml-metal
        // names the Metal device MTL0 via its "MTL%d" naming, not "Metal") and requires the MTL registry
        // behind it; when MTL0 is absent it fails loudly instead of falling back to the CPU or another GPU,
        // which would make the comparison say nothing.
        const ggml_backend_dev_t device = find_device("MTL0");
        const char * registry = ggml_backend_reg_name(ggml_backend_dev_backend_reg(device));
        if (std::strcmp(registry, "MTL") != 0) {
            throw Error(Fault::Device, std::string("the device ") + ggml_backend_dev_name(device) +
                                           " comes from the \"" + registry + "\" registry, not Metal's \"MTL\"",
                        "device");
        }
        ggml_backend_t backend = start_device(device, 4);
        std::printf("backend: %s, registry: %s, cpu threads: 4\n", ggml_backend_name(backend), registry);
        bool ok = true;
        {
            const ModelFile model(args[1], backend, layout);
            const Encoder encoder(model, backend);
            const Joint joint(model);
            const int64_t mels = model.u32("fastconformer.frontend.n_mels");
            const std::string output = args[2];
            std::unique_ptr<FILE, decltype(&std::fclose)> file(ggml_fopen(output.c_str(), "wb"), &std::fclose);
            if (!file) throw Error(Fault::Io, "cannot open the output file " + output);
            if (std::fwrite("sp013enc", 1, 8, file.get()) != 8) {
                throw Error(Fault::Io, "cannot write the check's output file");
            }
            write_u32(file.get(), 1);
            write_u32(file.get(), (uint32_t) encoder.subsampling_factor());
            write_u32(file.get(), (uint32_t) mels);
            write_u32(file.get(), (uint32_t) encoder.d_model());
            write_u32(file.get(), (uint32_t) joint.hidden());
            write_u32(file.get(), 24);
            write_u32(file.get(), (uint32_t) (sizeof LENGTHS / sizeof LENGTHS[0]));

            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            if (allocr == nullptr) throw Error(Fault::OutOfMemory, "cannot create a graph allocator");
            std::unordered_map<int64_t, std::vector<float>> first_of;
            uint64_t nonzero_total = 0, element_total = 0;
            bool repeats_identical = true, all_finite = true;
            for (size_t i = 0; i < sizeof LENGTHS / sizeof LENGTHS[0]; i++) {
                const int64_t t = LENGTHS[i];
                const int64_t frames = frames_for(encoder, t);
                Graph g;
                ggml_tensor * projected = joint.project_encoder(g.ctx(), encoder.build(g, synthetic_features(frames, mels), frames));
                g.output(projected);
                g.compute(backend, allocr);
                const std::vector<float> values = Graph::read(projected);

                if (projected->ne[0] != joint.hidden() || projected->ne[1] != t) {
                    std::printf("FAIL case %zu: the projection is %lld x %lld, expected %d x %lld\n", i,
                                (long long) projected->ne[0], (long long) projected->ne[1], joint.hidden(),
                                (long long) t);
                    ok = false;
                }
                uint64_t nonzero = 0;
                size_t nonfinite = 0;
                for (const float v : values) {
                    if (std::isfinite(v)) {
                        if (v != 0.0f) nonzero++;
                    } else {
                        nonfinite++;
                    }
                }
                if (nonfinite != 0) {
                    std::printf("FAIL case %zu: %zu of %zu values are not finite\n", i, nonfinite, values.size());
                    ok = false;
                    all_finite = false;
                }
                if (values.empty() || nonzero == 0) {
                    std::printf("FAIL case %zu: every value is zero\n", i);
                    ok = false;
                }
                std::string repeat;
                const auto seen = first_of.find(t);
                if (seen == first_of.end()) {
                    first_of.emplace(t, values);
                } else if (seen->second.size() != values.size() ||
                           std::memcmp(seen->second.data(), values.data(), values.size() * sizeof(float)) != 0) {
                    std::printf("FAIL case %zu: length %lld differs from its first occurrence\n", i, (long long) t);
                    ok = false;
                    repeats_identical = false;
                    repeat = ", repeat differs";
                } else {
                    repeat = ", repeat identical";
                }

                write_u32(file.get(), (uint32_t) i);
                write_u32(file.get(), (uint32_t) frames);
                write_u32(file.get(), (uint32_t) t);
                write_u32(file.get(), (uint32_t) projected->ne[1]);
                write_u64(file.get(), nonzero);
                write_u64(file.get(), (uint64_t) values.size());
                if (std::fwrite(values.data(), sizeof(float), values.size(), file.get()) != values.size()) {
                    throw Error(Fault::Io, "cannot write the projected output of case " + std::to_string(i));
                }
                nonzero_total += nonzero;
                element_total += values.size();
                std::printf("case %zu: %lld frames -> T %lld, %lld x %lld, nonzero %llu of %llu%s\n", i,
                            (long long) frames, (long long) t, (long long) projected->ne[0],
                            (long long) projected->ne[1], (unsigned long long) nonzero,
                            (unsigned long long) values.size(), repeat.c_str());
            }
            ggml_gallocr_free(allocr);
            std::printf("summary: %zu cases, %llu of %llu values nonzero, repeats %s, finite %s\n",
                        sizeof LENGTHS / sizeof LENGTHS[0], (unsigned long long) nonzero_total,
                        (unsigned long long) element_total, repeats_identical ? "byte-identical" : "different",
                        all_finite ? "yes" : "no");
            const long bytes = std::ftell(file.get());
            if (std::fclose(file.release()) != 0 || bytes < 0) {
                throw Error(Fault::Io, "cannot finish the output file " + output);
            }
            std::printf("wrote %ld bytes to %s\n", bytes, output.c_str());
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

#ifndef SPEECH_013_ENCODER_CHECK_NO_MAIN
int main(int argc, char ** argv) {
    return speech_013_encoder_check_main(argc, argv);
}
#endif
