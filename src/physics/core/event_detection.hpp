#pragma once

#include <functional>
#include <utility>

namespace physics {
namespace core {

// Simple scalar event: find t in [t0, t1] where g(t) crosses zero.
// Returns pair(found, t_event)
std::pair<bool,double> detect_event_bisection(
    const std::function<double(double)> &g,
    double t0,
    double t1,
    int max_iter = 30,
    double tol = 1e-6
);

} // namespace core
} // namespace physics


