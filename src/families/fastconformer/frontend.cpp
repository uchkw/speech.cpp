#include "frontend.h"

#include <cstdlib>
#include <cstring>

#include "error.h"

namespace fastconformer {

NemoFrontend make_frontend(const ModelFile & model) {
    bool vdsp = false;
    if (model.str("general.name") == "Haas-ja-v1.0" && model.str("fastconformer.decoder.prediction") == "kda") {
        const char * value = std::getenv("SPEECH_HAAS_FRONTEND_VDSP");
        if (value && std::strcmp(value, "0") != 0) {
            if (std::strcmp(value, "1") != 0) {
                throw Error(Fault::InvalidArgument, "SPEECH_HAAS_FRONTEND_VDSP must be 0 or 1");
            }
            vdsp = true;
        }
    }
    return NemoFrontend(model, "fastconformer.", true, vdsp);
}

}  // namespace fastconformer
