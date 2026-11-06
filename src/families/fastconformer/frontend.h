#pragma once

#include "nemo-frontend.h"

namespace fastconformer {

/** Constructs the family's frontend, with the explicitly selected Haas vDSP experiment where requested. */
NemoFrontend make_frontend(const ModelFile & model);

}  // namespace fastconformer
