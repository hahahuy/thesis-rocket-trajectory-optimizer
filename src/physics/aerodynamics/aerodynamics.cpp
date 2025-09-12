#include "physics/aerodynamics/aerodynamics.hpp"
#include <algorithm>
#include <cmath>

namespace physics {
namespace aerodynamics {

static inline double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

double drag_coefficient(double mach, double alpha_rad, const AeroParams &p) {
    double Cd_mach;
    if (mach <= p.mach_transonic_start) {
        // Subsonic
        Cd_mach = p.Cd_subsonic;
    } else if (mach >= p.mach_transonic_end) {
        // Supersonic, decay towards Cd_supersonic
        // Add a smooth decay after the peak
        double t = clamp((mach - p.mach_transonic_end) / 1.0, 0.0, 1.0);
        Cd_mach = p.Cd_supersonic + (p.Cd_transonic_peak - p.Cd_supersonic) * std::exp(-2.0 * t);
    } else {
        // Transonic ramp up to peak
        double t = (mach - p.mach_transonic_start) / (p.mach_transonic_end - p.mach_transonic_start);
        Cd_mach = p.Cd_subsonic + t * (p.Cd_transonic_peak - p.Cd_subsonic);
    }

    double Cd_alpha = p.Cd_alpha_slope * std::abs(alpha_rad);
    return Cd_mach + Cd_alpha;
}

} // namespace aerodynamics
} // namespace physics


