// Checks the encoder projection against NeMo's RNNTJoint.project_encoder(), from the official encoder output
// in each dump of reference/fastconformer/dump-encoder.py. This isolates the joint's projection from encoder error.
//
// usage: fastconformer-joint-encoder-check <model.gguf> <reference out dir> [gpu|cpu|device name]

#include <cstdio>
#include <stdexcept>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "reference-dumps.h"
#include "fastconformer/layout.h"
#include "fastconformer/transducer.h"
#include "ggml-cpu.h"
#include "npy.h"
#include "weight-types.h"

using namespace fastconformer;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir> [gpu|cpu|device name]\n", args[0].c_str());
        return 2;
    }
    try {
        if (!takes_weights(args[1], layout, {"F32", "F16"}, "its bounds are those of float32 and half precision")) return 2;
        ggml_backend_t backend = init_backend(args.size() > 3 ? args[3] : "");
        std::printf("backend: %s\n", ggml_backend_name(backend));
        bool ok = true;
        {
            const ModelFile model(args[1], backend, layout);
            const Joint joint(model);
            // Measured on an M2 Ultra CPU with Haas, eleven 2 to 28.2 s inputs, 2026-11-07: 125.4 to 125.9 dB
            // with F32 weights and 63.9 to 64.9 dB with F16. Half-precision products use the joint check's 40 dB bound.
            const bool float32 = ggml_backend_is_cpu(backend) && model.tensor("joint.enc.weight")->type == GGML_TYPE_F32;
            const double threshold_db = float32 ? 90 : 40;
            std::printf("encoder projection threshold: %.0f dB\n", threshold_db);
            ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
            for (const auto & d : reference_dumps(args[2], model)) {
                const Npy encoded = read_npy((d / "encoded.npy").u8string());
                const Npy want = read_npy((d / "joint_encoded.npy").u8string());
                if (encoded.shape.size() != 2 || want.shape.size() != 2 || encoded.shape[0] != want.shape[0] ||
                    encoded.shape[1] != model.size("fastconformer.encoder.d_model") || want.shape[1] != joint.hidden()) {
                    throw std::runtime_error(d.u8string() + ": encoded.npy and joint_encoded.npy have incompatible shapes");
                }
                Graph graph;
                ggml_tensor * out = joint.project_encoder(graph.ctx(), graph.input(encoded.f32, encoded.shape[1], encoded.shape[0]));
                graph.output(out);
                graph.compute(backend, allocr);
                const Diff difference = compare(Graph::read(out), want.f32);
                std::printf("%s (%lld frames)\n", d.filename().u8string().c_str(), (long long) encoded.shape[0]);
                print_diff("  encoder projection", difference);
                ok = ok && difference.snr_db > threshold_db;
            }
            ggml_gallocr_free(allocr);
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
