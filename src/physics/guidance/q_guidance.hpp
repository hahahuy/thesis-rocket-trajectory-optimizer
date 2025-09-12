#pragma once

#include <utility>

namespace physics {
namespace guidance {

struct QGuidanceParams {
    double gain_position = 0.0;   // m^-1
    double gain_velocity = 0.0;   // s/m
    double max_gimbal_rad = 0.35; // rad
};

// Minimal Q-guidance placeholder: maps (position, velocity) error to commanded thrust vector angle
// Returns thrust direction unit vector components (cos(theta), sin(theta))
std::pair<double,double> compute_thrust_direction(
    double x_err,
    double y_err,
    double vx_err,
    double vy_err,
    const QGuidanceParams &p
);

} // namespace guidance
} // namespace physics


