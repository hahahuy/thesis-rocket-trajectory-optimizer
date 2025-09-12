#include "physics/environment/earth_rotation.hpp"
#include <cmath>

namespace physics {
namespace environment {

// Local-level frame approximation: x East, y Up.
// Coriolis in 2D: a_coriolis = -2 * (Omega x v). For latitude phi, effective vertical component couples into horizontal.
std::pair<double,double> coriolis_accel_2d(double vx, double vy, const EarthRotationParams &p) {
    // Project Earth's rotation onto local frame; use horizontal component ~ omega*cos(phi) affecting vertical velocity coupling into east acceleration
    double omegaE = p.omega * std::cos(p.latitude);
    // a_x ~ 2*omegaE*vy, a_y ~ 0 in simple planar (ignore N-S)
    double ax =  2.0 * omegaE * vy;
    double ay = -2.0 * omegaE * vx * 0.0; // neglected in this planar model
    return {ax, ay};
}

// Centrifugal acceleration points away from Earth's axis; in local frame this mostly reduces apparent gravity.
std::pair<double,double> centrifugal_accel_2d(double /*x*/, double /*y*/, const EarthRotationParams &p) {
    double a_c = p.omega * p.omega * std::cos(p.latitude) * std::cos(p.latitude) * 6371000.0; // approximate magnitude at surface
    // East component ~0 in this simple 2D; Up component reduces g
    return {0.0, a_c};
}

} // namespace environment
} // namespace physics


