#pragma once

#include <cmath>

namespace rocket_physics {
namespace smooth {

/**
 * @brief Smooth approximation of max(a, b)
 * Uses softplus: max(a,b) ≈ log(exp(k*a) + exp(k*b)) / k
 * @param a First value
 * @param b Second value
 * @param k Smoothness parameter (larger = sharper, default 10)
 * @return Smooth maximum
 */
inline double smooth_max(double a, double b, double k = 10.0) {
    if (std::abs(a - b) < 1e-10) return a;
    double m = std::max(a, b);
    double diff = a - b;
    return m + std::log(1.0 + std::exp(-k * std::abs(diff))) / k;
}

/**
 * @brief Smooth approximation of min(a, b)
 * @param a First value
 * @param b Second value
 * @param k Smoothness parameter
 * @return Smooth minimum
 */
inline double smooth_min(double a, double b, double k = 10.0) {
    return -smooth_max(-a, -b, k);
}

/**
 * @brief Smooth clamp: clamp(x, lo, hi) using smooth min/max
 * @param x Value to clamp
 * @param lo Lower bound
 * @param hi Upper bound
 * @param k Smoothness parameter
 * @return Clamped value
 */
inline double smooth_clamp(double x, double lo, double hi, double k = 10.0) {
    return smooth_min(smooth_max(x, lo, k), hi, k);
}

/**
 * @brief Smooth approximation of atan2 that avoids NaN at origin
 * @param y Y component
 * @param x X component
 * @param eps Small epsilon to avoid division by zero
 * @return Smooth angle in radians
 */
inline double smooth_atan2(double y, double x, double eps = 1e-8) {
    double r = std::sqrt(x*x + y*y);
    if (r < eps) {
        // Near origin: use smooth approximation
        return y / (eps + x);
    }
    return std::atan2(y, x);
}

/**
 * @brief Smooth sign function
 * @param x Input value
 * @param eps Smoothness parameter
 * @return Smooth sign (-1 to 1)
 */
inline double smooth_sign(double x, double eps = 1e-6) {
    return x / std::sqrt(x*x + eps*eps);
}

} // namespace smooth
} // namespace rocket_physics
