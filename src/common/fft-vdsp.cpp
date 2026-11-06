#include "fft-vdsp.h"

#include <string>

#include "error.h"

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

struct VdspRealFft::Setup {
#ifdef __APPLE__
    explicit Setup(int size) : value(vDSP_DFT_zrop_CreateSetupD(nullptr, (vDSP_Length) size, vDSP_DFT_FORWARD)) {
        if (!value) throw Error(Fault::OutOfMemory, "cannot allocate the vDSP setup for " + std::to_string(size) + " values");
    }
    ~Setup() { vDSP_DFT_DestroySetupD(value); }
    vDSP_DFT_SetupD value;
#endif
};

VdspRealFft::VdspRealFft(int size) : size_(size) {
    int factor = size, twos = 0;
    while (factor > 0 && factor % 2 == 0) {
        factor /= 2;
        ++twos;
    }
    if (twos < 4 || (factor != 1 && factor != 3 && factor != 5 && factor != 15)) {
        throw Error(Fault::InvalidArgument, "vDSP does not support a real DFT of " + std::to_string(size) +
                    " values; use SPEECH_HAAS_FRONTEND_VDSP=0");
    }
#ifdef __APPLE__
    setup_ = std::make_unique<Setup>(size);
#else
    throw Error(Fault::InvalidArgument, "the vDSP frontend requires macOS; use SPEECH_HAAS_FRONTEND_VDSP=0");
#endif
}

VdspRealFft::~VdspRealFft() = default;

void VdspRealFft::transform(const double * in, std::complex<double> * out, double * scratch) const {
#ifdef __APPLE__
    const int half = size_ / 2;
    double * real = scratch, * imag = scratch + half;
    for (int i = 0; i < half; ++i) {
        real[i] = in[2 * i];
        imag[i] = in[2 * i + 1];
    }
    vDSP_DFT_ExecuteD(setup_->value, real, imag, real, imag);
    // A real forward DFT doubles every bin and packs the real Nyquist value into the imaginary DC slot.
    out[0] = {0.5 * real[0], 0.0};
    out[half] = {0.5 * imag[0], 0.0};
    for (int i = 1; i < half; ++i) out[i] = {0.5 * real[i], 0.5 * imag[i]};
#else
    (void) in;
    (void) out;
    (void) scratch;
    throw Error(Fault::InvalidArgument, "the vDSP frontend requires macOS; use SPEECH_HAAS_FRONTEND_VDSP=0");
#endif
}
