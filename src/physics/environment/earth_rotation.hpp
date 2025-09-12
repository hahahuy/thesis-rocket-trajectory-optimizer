#pragma once

#include <utility>

namespace physics {
namespace environment {

struct EarthRotationParams {
    double omega;      // rad/s
    double latitude;   // rad
    bool enable_coriolis;
    bool enable_centrifugal;

    EarthRotationParams()
        : omega(7.2921159e-5), latitude(0.0), enable_coriolis(false), enable_centrifugal(false) {}
};

// 2D planar approximations of Coriolis and centrifugal accelerations in local-level frame (x East, y Up)
// velocity: (vx, vy) in m/s, position: (x, y) with y~altitude; returns (ax, ay)
std::pair<double,double> coriolis_accel_2d(double vx, double vy, const EarthRotationParams &p);
std::pair<double,double> centrifugal_accel_2d(double x, double y, const EarthRotationParams &p);

} // namespace environment
} // namespace physics


