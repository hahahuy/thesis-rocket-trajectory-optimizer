#include "physics/environment/isa_atmosphere.hpp"
#include <algorithm>
#include <cmath>

namespace physics {
namespace environment {

// Physical constants
static constexpr double kR = 287.05287;     // J/(kg*K)
static constexpr double kGamma = 1.4;       // specific heat ratio
static constexpr double kG0 = 9.80665;      // m/s^2

const std::vector<ISA_Layer> ISA_Atmosphere::kLayers = {
    // base_alt [m], T_base [K], P_base [Pa], lapse [K/m]
    {     0.0, 288.15, 101325.0,   -0.0065},  // 0-11 km (troposphere)
    { 11000.0, 216.65,  22632.1,    0.0   },  // 11-20 km (tropopause)
    { 20000.0, 216.65,   5474.89,    0.001 },  // 20-32 km (stratosphere)
    { 32000.0, 228.65,    868.019,   0.0028},  // 32-47 km
    { 47000.0, 270.65,    110.906,   0.0   },  // 47-51 km
    { 51000.0, 270.65,     66.9389, -0.0028},  // 51-71 km
    { 71000.0, 214.65,      3.9564, -0.002  }  // 71-86 km
};

static inline double sutherland_viscosity(double T) {
    // Sutherland's law for air
    const double C1 = 1.458e-6; // kg/(m*s*sqrt(K))
    const double S = 110.4;     // K
    return C1 * std::sqrt(T) * T / (T + S);
}

AtmosphereState ISA_Atmosphere::compute_properties(double altitude_m) {
    AtmosphereState out{};
    double h = std::max(0.0, altitude_m);

    // Find current layer (last with base_altitude <= h)
    size_t idx = 0;
    for (size_t i = 0; i + 1 < kLayers.size(); ++i) {
        if (h >= kLayers[i+1].base_altitude) idx = i + 1; else break;
    }
    const ISA_Layer &layer = kLayers[idx];

    const double L = layer.lapse_rate;
    const double T0 = layer.base_temperature;
    const double P0 = layer.base_pressure;
    const double h0 = layer.base_altitude;

    double T;
    double P;
    if (std::abs(L) > 1e-12) {
        T = T0 + L * (h - h0);
        P = P0 * std::pow(T0 / T, kG0 / (kR * L));
    } else {
        T = T0;
        P = P0 * std::exp(-kG0 * (h - h0) / (kR * T0));
    }

    double rho = P / (kR * T);
    double a = std::sqrt(kGamma * kR * T);

    out.temperature = T;
    out.pressure = P;
    out.density = rho;
    out.speed_of_sound = a;
    out.viscosity = sutherland_viscosity(T);
    return out;
}

} // namespace environment
} // namespace physics


