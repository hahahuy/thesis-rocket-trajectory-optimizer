#include "physics/core/event_detection.hpp"

namespace physics {
namespace core {

std::pair<bool,double> detect_event_bisection(
    const std::function<double(double)> &g,
    double t0,
    double t1,
    int max_iter,
    double tol
) {
    double g0 = g(t0);
    double g1 = g(t1);
    if (g0 == 0.0) return {true, t0};
    if (g1 == 0.0) return {true, t1};
    if (g0 * g1 > 0.0) return {false, t1};

    double a = t0;
    double b = t1;
    for (int i = 0; i < max_iter; ++i) {
        double m = 0.5 * (a + b);
        double gm = g(m);
        if (std::abs(gm) < tol || 0.5 * (b - a) < tol) {
            return {true, m};
        }
        if (g0 * gm <= 0.0) {
            b = m;
            g1 = gm;
        } else {
            a = m;
            g0 = gm;
        }
    }
    return {true, 0.5 * (a + b)};
}

} // namespace core
} // namespace physics


