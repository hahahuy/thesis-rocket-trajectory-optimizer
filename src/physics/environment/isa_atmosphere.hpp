#pragma once

#include <vector>

namespace physics {
namespace environment {

struct AtmosphereState {
    double temperature;    // K
    double pressure;       // Pa
    double density;        // kg/m^3
    double speed_of_sound; // m/s
    double viscosity;      // Pa*s (optional, Sutherland)
};

struct ISA_Layer {
    double base_altitude;     // m
    double base_temperature;  // K
    double base_pressure;     // Pa
    double lapse_rate;        // K/m
};

class ISA_Atmosphere {
public:
    // Computes standard atmosphere properties up to ~86 km (1976 ISA)
    // altitude in meters above mean sea level
    static AtmosphereState compute_properties(double altitude_m);

private:
    static const std::vector<ISA_Layer> kLayers;
};

} // namespace environment
} // namespace physics


