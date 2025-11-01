#include "atmosphere.hpp"
#include <cmath>
#include <algorithm>

namespace rocket_physics {

// ISA Atmosphere implementation
ISAAtmosphere::ISAAtmosphere() {
}

std::pair<double, double> ISAAtmosphere::computeProperties(double altitude) const {
    double density = computeDensity(altitude);
    double pressure = computePressure(altitude);
    return {density, pressure};
}

double ISAAtmosphere::computeDensity(double altitude) const {
    if (altitude < 0.0) {
        return RHO0;
    }
    
    int layer = getLayerIndex(altitude);
    return computeLayerProperties(altitude, layer);
}

double ISAAtmosphere::computePressure(double altitude) const {
    if (altitude < 0.0) {
        return P0;
    }
    
    int layer = getLayerIndex(altitude);
    double temperature = computeTemperature(altitude);
    double density = computeDensity(altitude);
    
    return density * R * temperature;
}

double ISAAtmosphere::computeTemperature(double altitude) const {
    if (altitude < 0.0) {
        return T0;
    }
    
    int layer = getLayerIndex(altitude);
    double T = T0;
    
    // Compute temperature based on layer
    if (layer == 0) { // Troposphere
        T = T0 + L0 * altitude;
    } else if (layer == 1) { // Tropopause
        T = T0 + L0 * H1;
    } else if (layer == 2) { // Lower stratosphere
        T = T0 + L0 * H1 + L2 * (altitude - H1);
    } else if (layer == 3) { // Upper stratosphere
        T = T0 + L0 * H1 + L2 * (H2 - H1) + L3 * (altitude - H2);
    } else if (layer == 4) { // Stratopause
        T = T0 + L0 * H1 + L2 * (H2 - H1) + L3 * (H3 - H2);
    } else if (layer == 5) { // Mesosphere
        T = T0 + L0 * H1 + L2 * (H2 - H1) + L3 * (H3 - H2) + L5 * (altitude - H3);
    } else { // Mesopause and above
        T = T0 + L0 * H1 + L2 * (H2 - H1) + L3 * (H3 - H2) + L5 * (H5 - H3) + L6 * (altitude - H5);
    }
    
    return T;
}

double ISAAtmosphere::computeSpeedOfSound(double altitude) const {
    double temperature = computeTemperature(altitude);
    return std::sqrt(gamma * R * temperature);
}

double ISAAtmosphere::computeMachNumber(double velocity, double altitude) const {
    double speed_of_sound = computeSpeedOfSound(altitude);
    return velocity / speed_of_sound;
}

int ISAAtmosphere::getLayerIndex(double altitude) const {
    if (altitude <= H1) return 0;
    if (altitude <= H2) return 1;
    if (altitude <= H3) return 2;
    if (altitude <= H4) return 3;
    if (altitude <= H5) return 4;
    if (altitude <= H6) return 5;
    return 6;
}

double ISAAtmosphere::computeLayerProperties(double altitude, int layer) const {
    // Simplified implementation - in practice, this would use the full ISA equations
    double temperature = computeTemperature(altitude);
    double pressure = P0 * std::pow(temperature / T0, -g0 / (R * L0));
    return pressure / (R * temperature);
}

// Exponential Atmosphere implementation
ExponentialAtmosphere::ExponentialAtmosphere(double rho0, double h_scale, double T0)
    : rho0_(rho0), h_scale_(h_scale), T0_(T0), R_(287.0), gamma_(1.4) {
}

std::pair<double, double> ExponentialAtmosphere::computeProperties(double altitude) const {
    double density = computeDensity(altitude);
    double pressure = computePressure(altitude);
    return {density, pressure};
}

double ExponentialAtmosphere::computeDensity(double altitude) const {
    if (altitude < 0.0) {
        return rho0_;
    }
    return rho0_ * std::exp(-altitude / h_scale_);
}

double ExponentialAtmosphere::computePressure(double altitude) const {
    if (altitude < 0.0) {
        return 101325.0; // Sea level pressure
    }
    double density = computeDensity(altitude);
    double temperature = computeTemperature(altitude);
    return density * R_ * temperature;
}

double ExponentialAtmosphere::computeTemperature(double altitude) const {
    if (altitude < 0.0) {
        return T0_;
    }
    // Linear temperature model
    return T0_ - 0.0065 * altitude;
}

double ExponentialAtmosphere::computeSpeedOfSound(double altitude) const {
    double temperature = computeTemperature(altitude);
    return std::sqrt(gamma_ * R_ * temperature);
}

double ExponentialAtmosphere::computeMachNumber(double velocity, double altitude) const {
    double speed_of_sound = computeSpeedOfSound(altitude);
    return velocity / speed_of_sound;
}

// Wind Model implementations
ConstantWindModel::ConstantWindModel(const Vec3& wind_velocity) : wind_velocity_(wind_velocity) {
}

Vec3 ConstantWindModel::computeWind(const Vec3& position, double t) const {
    return wind_velocity_;
}

AltitudeWindModel::AltitudeWindModel(std::function<Vec3(double)> wind_profile) : wind_profile_(wind_profile) {
}

Vec3 AltitudeWindModel::computeWind(const Vec3& position, double t) const {
    double altitude = position.norm() - 6371000.0; // Earth radius
    return wind_profile_(altitude);
}

// Factory functions
std::shared_ptr<ISAAtmosphere> createISAAtmosphere() {
    return std::make_shared<ISAAtmosphere>();
}

std::shared_ptr<ExponentialAtmosphere> createExponentialAtmosphere(double rho0, double h_scale, double T0) {
    return std::make_shared<ExponentialAtmosphere>(rho0, h_scale, T0);
}

std::shared_ptr<NoWindModel> createNoWindModel() {
    return std::make_shared<NoWindModel>();
}

std::shared_ptr<ConstantWindModel> createConstantWindModel(const Vec3& wind_velocity) {
    return std::make_shared<ConstantWindModel>(wind_velocity);
}

std::shared_ptr<AltitudeWindModel> createAltitudeWindModel(std::function<Vec3(double)> wind_profile) {
    return std::make_shared<AltitudeWindModel>(wind_profile);
}

} // namespace rocket_physics
