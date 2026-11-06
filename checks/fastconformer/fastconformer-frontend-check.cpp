// Checks the FastConformer frontend against NeMo's preprocessor: the normalized log-mel features of each dump
// of reference/fastconformer/dump.py, computed from the dump's own audio. The dumps are those of the model, in
// <reference out dir>/<its general.name>/.
//
// usage: fastconformer-frontend-check <model.gguf> <reference out dir>

#include <cstdio>

#include "args.h"
#include "backend.h"
#include "compare.h"
#include "reference-dumps.h"
#include "fastconformer/layout.h"
#include "fastconformer/frontend.h"
#include "ggml-cpu.h"
#include "nemo-frontend.h"
#include "npy.h"

using namespace fastconformer;

int main(int argc, char ** argv) {
    const std::vector<std::string> args = utf8_args(argc, argv);
    if (args.size() < 3) {
        std::fprintf(stderr, "usage: %s <model.gguf> <reference out dir>\n", args[0].c_str());
        return 2;
    }
    try {
        start_ggml();
        // The frontend runs on the host; the backend only holds its window and filterbank.
        ggml_backend_t backend = ggml_backend_cpu_init();
        bool ok = true;
        {
            const ModelFile model(args[1], backend, layout);
            const NemoFrontend frontend = make_frontend(model);
            for (const auto & d : reference_dumps(args[2], model)) {
                const Npy audio = read_npy((d / "audio.npy").u8string());
                const Npy want = read_npy((d / "features.npy").u8string());
                const std::vector<float> got = frontend.features(audio.f32);
                const Diff diff = compare(got, want.f32);
                std::printf("%s (%lld frames)\n", d.filename().u8string().c_str(), (long long) want.shape[0]);
                print_diff("  features", diff);
                // Measured on an Apple M5: 117 to 127 dB with parakeet-tdt_ctc-0.6b-ja on 2026-10-05 and 94 to 125 dB
                // with parakeet-tdt-0.6b-v3 on 2026-10-06, the gap between this double precision and torch.stft()'s
                // float32. The same steps in float64 in PyTorch give the dump's fr_fr-10043298898524273336 the same
                // 94.0 dB, with the largest difference in the second mel bin. A wrong window, padding, filterbank or
                // normalization falls far below.
                ok = ok && got.size() == want.f32.size() && want.shape[1] == frontend.mels() && diff.snr_db > 90;
            }
        }
        ggml_backend_free(backend);
        return ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
