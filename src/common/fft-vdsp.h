#pragma once

#include <complex>
#include <memory>

/** A double-precision real DFT with an owned Accelerate setup and caller-owned working storage. */
class VdspRealFft {
public:
    /** Requires macOS and size = f * 2^n, with f in {1, 3, 5, 15} and n >= 4. */
    explicit VdspRealFft(int size);
    ~VdspRealFft();

    /** Transforms size inputs into size / 2 + 1 unscaled bins, using size scratch doubles; the arrays do not overlap. */
    void transform(const double * in, std::complex<double> * out, double * scratch) const;

private:
    struct Setup;
    int size_;
    std::unique_ptr<Setup> setup_;
};
