#include "physics/guidance/q_guidance.hpp"
#include <cmath>

namespace physics {
namespace guidance {

std::pair<double,double> compute_thrust_direction(
    double x_err,
    double y_err,
    double vx_err,
    double vy_err,
    const QGuidanceParams &p
) {
    // Linear combination of position/velocity errors as a crude proxy
    double ax_c = -p.gain_position * x_err - p.gain_velocity * vx_err;
    double ay_c = -p.gain_position * y_err - p.gain_velocity * vy_err;
    double norm = std::hypot(ax_c, ay_c);
    if (norm < 1e-9) {
        return {1.0, 0.0};
    }
    double cx = ax_c / norm;
    double cy = ay_c / norm;
    return {cx, cy};
}

} // namespace guidance
} // namespace physics


