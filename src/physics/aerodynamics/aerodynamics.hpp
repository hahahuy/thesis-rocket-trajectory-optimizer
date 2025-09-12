#pragma once

namespace physics {
namespace aerodynamics {

struct AeroParams {
    double Cd_subsonic = 0.3;
    double Cd_transonic_peak = 1.2;
    double Cd_supersonic = 0.8;
    double mach_transonic_start = 0.8;
    double mach_transonic_end = 1.2;
    double Cd_alpha_slope = 0.0; // per rad, 0 for planar model without AoA
};

// Piecewise-linear interpolation for Cd vs Mach with a transonic bump
double drag_coefficient(double mach, double alpha_rad, const AeroParams &p);

} // namespace aerodynamics
} // namespace physics


