// Checks the double vDSP real DFT against a compensated direct DFT and RealFft, including DC, Nyquist, all impulse
// positions, quiet signals, poisoned working storage and unsupported lengths. This check does not measure speed.
// Usage: fft-vdsp-check

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <limits>
#include <vector>

#include "error.h"
#include "fft.h"
#include "fft-vdsp.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

/** Neumaier's compensated sum keeps the direct reference's error from growing with its term count. */
class Sum {
public:
    void add(double value) {
        const double next = sum_ + value;
        carry_ += std::fabs(sum_) >= std::fabs(value) ? (sum_ - next) + value : (value - next) + sum_;
        sum_ = next;
    }
    double value() const { return sum_ + carry_; }
private:
    double sum_ = 0, carry_ = 0;
};

std::vector<std::complex<double>> direct(const std::vector<double> & input) {
    const size_t n = input.size();
    std::vector<std::complex<double>> output(n / 2 + 1);
    for (size_t k = 0; k < output.size(); ++k) {
        Sum re, im;
        for (size_t i = 0; i < n; ++i) {
            const double angle = -2 * kPi * (double) (i * k % n) / (double) n;
            re.add(input[i] * std::cos(angle));
            im.add(input[i] * std::sin(angle));
        }
        output[k] = {re.value(), im.value()};
    }
    return output;
}

bool close(const std::complex<double> * got, const std::complex<double> * want, size_t bins, int n, double & worst) {
    double error = 0, norm = 0;
    for (size_t k = 0; k < bins; ++k) {
        if (!std::isfinite(got[k].real()) || !std::isfinite(got[k].imag())) return false;
        error += std::norm(got[k] - want[k]);
        norm += std::norm(want[k]);
    }
    if (norm == 0) return error == 0;
    const double relative = std::sqrt(error / norm) / std::numeric_limits<double>::epsilon();
    worst = std::max(worst, relative);
    int log2_size = 0;
    for (int size = n; size > 1; size /= 2) ++log2_size;
    // This is the existing fft-check bound; changing the implementation does not relax it.
    return relative < 6 * log2_size;
}

}  // namespace

int main() {
    try {
        bool ok = true;
        for (int n : {0, 1, 2, 6, 14, 42, 50, 400}) {
            bool refused = false;
            try { const VdspRealFft unsupported(n); }
            catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
            ok = ok && refused;
        }
#ifndef __APPLE__
        bool refused = false;
        try { const VdspRealFft unavailable(512); }
        catch (const Error & error) { refused = error.fault() == Fault::InvalidArgument; }
        std::printf("vDSP unavailable: explicit selection %s; numerical cases are not run\n", refused ? "refused" : "FAIL");
        return ok && refused ? 0 : 1;
#else
        size_t cases = 0;
        for (int n : {16, 32, 48, 80, 240, 512}) {
            const RealFft ordinary(n);
            const VdspRealFft vdsp(n);
            const size_t bins = (size_t) n / 2 + 1;
            const std::complex<double> guard{12345.0, -9876.0};
            const double poison = std::numeric_limits<double>::quiet_NaN();
            std::vector<double> input(n), scratch((size_t) n + 2, 54321.0);
            std::vector<std::complex<double>> output(bins + 2, guard), baseline(bins);
            double worst_direct = 0, worst_pair = 0;
            auto check = [&](const std::vector<std::complex<double>> & expected) {
                std::fill(scratch.begin() + 1, scratch.end() - 1, poison);
                std::fill(output.begin() + 1, output.end() - 1, std::complex<double>{poison, poison});
                const auto saved = input;
                vdsp.transform(input.data(), output.data() + 1, scratch.data() + 1);
                ordinary.transform(input.data(), baseline.data());
                const bool passed = input == saved && output.front() == guard && output.back() == guard &&
                    scratch.front() == 54321.0 && scratch.back() == 54321.0 &&
                    output[1].imag() == 0 && output[bins].imag() == 0 &&
                    close(output.data() + 1, expected.data(), bins, n, worst_direct) &&
                    close(output.data() + 1, baseline.data(), bins, n, worst_pair);
                if (!passed) std::printf("FAIL: size %d, case %zu\n", n, cases);
                ok = ok && passed;
                ++cases;
            };
            for (int pattern = 0; pattern < 6; ++pattern) {
                for (int i = 0; i < n; ++i) {
                    if (pattern == 0) input[i] = std::sin(i * 1.37) + 0.5 * std::cos(i * i * 0.79);
                    if (pattern == 1) input[i] = 0.75;
                    if (pattern == 2) input[i] = i % 2 ? -0.5 : 0.5;
                    if (pattern == 3) input[i] = std::sin(2 * kPi * i / n) + 0.25 * std::cos(6 * kPi * i / n);
                    if (pattern == 4) input[i] = 1e-30 * (std::sin(i * 1.37) + 0.5 * std::cos(i * i * 0.79));
                    if (pattern == 5) input[i] = 0.0;
                }
                check(direct(input));
            }
            for (int impulse = 0; impulse < n; ++impulse) {
                std::fill(input.begin(), input.end(), 0.0);
                input[(size_t) impulse] = 1.0;
                std::vector<std::complex<double>> expected(bins);
                for (size_t k = 0; k < bins; ++k) {
                    const double angle = -2 * kPi * (double) ((size_t) impulse * k % (size_t) n) / n;
                    expected[k] = {std::cos(angle), std::sin(angle)};
                }
                check(expected);
            }
            std::printf("size %d: worst direct %.3f eps, ordinary %.3f eps\n", n, worst_direct, worst_pair);
        }
        std::printf("vDSP real DFT: %zu cases, %s; no performance measurement\n", cases, ok ? "ok" : "FAIL");
        return ok ? 0 : 1;
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
