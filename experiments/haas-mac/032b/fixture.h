#pragma once

#include <cstdint>
#include <string>

#include "model-file.h"

/**
 * The synthetic FastConformer of the 032b check: small enough to build and quantize in moments, with a head width of
 * 128 (d_model 256 over 2 heads) and every matrix axis a whole block of 256 values, so a file of Q4_K holds its
 * matrices in Q4_K. The subsampling has three stride-2 layers and the conformer convolution a kernel of 9, as the
 * family's models have them.
 */
namespace batch2 {

constexpr int kMels = 80, kChannels = 256, kDModel = 256, kHeads = 2, kLayers = 2, kFfn = 512;
constexpr int kConvKernel = 9, kSubLayers = 3, kJointHidden = 256, kOutputs = 8;

/** The frontend frames that give `t` frames after the stride-2 subsampling: calc_length() of the three layers. */
constexpr int64_t frames_of(int64_t t) { return ((int64_t) 1 << kSubLayers) * (t - 1) + 1; }

/**
 * Writes the check's F32 model file at `path`, every tensor filled with finite nonzero values of its own. With
 * `local_attention` the file names rel_pos_local_attn with its context and global tokens, which the batched pair
 * encoder must refuse; the tensors are the same either way.
 */
void write_fixture(const std::string & path, bool local_attention);

}  // namespace batch2

/** The layout of the 032b check's own file, which names exactly the tensors write_fixture() writes. */
extern const Layout kBatch2Layout;
