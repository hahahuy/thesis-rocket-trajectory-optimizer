#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <functional>
#include <memory>

namespace rocket_physics {

/**
 * @brief Atmospheric model interface
 */
class Atmosphere {
public:
    /**
     * @brief Destructor
     */
    virtual ~Atmosphere() = default;
    
    /**
     * @brief Compute atmospheric properties at given altitude
     * @param altitude Altitude [m]
     * @return Pair of (density, pressure) [kg/m³, Pa]
     */
    virtual std::pair<double, double> computeProperties(double altitude) const = 0;
    
    /**
     * @brief Compute density at given altitude
     * @param altitude Altitude [m]
     * @return Density [kg/m³]
     */
    virtual double computeDensity(double altitude) const = 0;
    
    /**
     * @brief Compute pressure at given altitude
     * @param altitude Altitude [m]
     * @return Pressure [Pa]
     */
    virtual double computePressure(double altitude) const = 0;
    
    /**
     * @brief Compute temperature at given altitude
     * @param altitude Altitude [m]
     * @return Temperature [K]
     */
    virtual double computeTemperature(double altitude) const = 0;
    
    /**
     * @brief Compute speed of sound at given altitude
     * @param altitude Altitude [m]
     * @return Speed of sound [m/s]
     */
    virtual double computeSpeedOfSound(double altitude) const = 0;
    
    /**
     * @brief Compute Mach number
     * @param velocity Velocity magnitude [m/s]
     * @param altitude Altitude [m]
     * @return Mach number
     */
    virtual double computeMachNumber(double velocity, double altitude) const = 0;
};

/**
 * @brief International Standard Atmosphere (ISA) model
 */
class ISAAtmosphere : public Atmosphere {
public:
    /**
     * @brief Constructor
     */
    ISAAtmosphere();
    
    /**
     * @brief Compute atmospheric properties
     * @param altitude Altitude [m]
     * @return Pair of (density, pressure) [kg/m³, Pa]
     */
    std::pair<double, double> computeProperties(double altitude) const override;
    
    /**
     * @brief Compute density
     * @param altitude Altitude [m]
     * @return Density [kg/m³]
     */
    double computeDensity(double altitude) const override;
    
    /**
     * @brief Compute pressure
     * @param altitude Altitude [m]
     * @return Pressure [Pa]
     */
    double computePressure(double altitude) const override;
    
    /**
     * @brief Compute temperature
     * @param altitude Altitude [m]
     * @return Temperature [K]
     */
    double computeTemperature(double altitude) const override;
    
    /**
     * @brief Compute speed of sound
     * @param altitude Altitude [m]
     * @return Speed of sound [m/s]
     */
    double computeSpeedOfSound(double altitude) const override;
    
    /**
     * @brief Compute Mach number
     * @param velocity Velocity magnitude [m/s]
     * @param altitude Altitude [m]
     * @return Mach number
     */
    double computeMachNumber(double velocity, double altitude) const override;

private:
    // ISA constants
    static constexpr double R = 287.0;           // Gas constant for air [J/(kg·K)]
    static constexpr double g0 = 9.80665;        // Standard gravity [m/s²]
    static constexpr double gamma = 1.4;         // Ratio of specific heats
    
    // Sea level properties
    static constexpr double T0 = 288.15;          // Sea level temperature [K]
    static constexpr double P0 = 101325.0;        // Sea level pressure [Pa]
    static constexpr double RHO0 = 1.225;         // Sea level density [kg/m³]
    
    // ISA layer boundaries and lapse rates
    static constexpr double H0 = 0.0;             // Sea level [m]
    static constexpr double H1 = 11000.0;         // Tropopause [m]
    static constexpr double H2 = 20000.0;         // Lower stratosphere [m]
    static constexpr double H3 = 32000.0;         // Upper stratosphere [m]
    static constexpr double H4 = 47000.0;         // Stratopause [m]
    static constexpr double H5 = 51000.0;         // Mesosphere [m]
    static constexpr double H6 = 71000.0;         // Mesopause [m]
    
    static constexpr double L0 = -0.0065;        // Troposphere lapse rate [K/m]
    static constexpr double L1 = 0.0;             // Tropopause lapse rate [K/m]
    static constexpr double L2 = 0.001;           // Lower stratosphere lapse rate [K/m]
    static constexpr double L3 = 0.0028;          // Upper stratosphere lapse rate [K/m]
    static constexpr double L4 = 0.0;             // Stratopause lapse rate [K/m]
    static constexpr double L5 = -0.0028;         // Mesosphere lapse rate [K/m]
    static constexpr double L6 = -0.002;          // Mesopause lapse rate [K/m]
    
    // Helper methods
    int getLayerIndex(double altitude) const;
    double computeLayerProperties(double altitude, int layer) const;
};

/**
 * @brief Exponential atmosphere model (simplified)
 */
class ExponentialAtmosphere : public Atmosphere {
public:
    /**
     * @brief Constructor
     * @param rho0 Sea level density [kg/m³]
     * @param h_scale Scale height [m]
     * @param T0 Sea level temperature [K]
     */
    ExponentialAtmosphere(double rho0 = 1.225, double h_scale = 8400.0, double T0 = 288.15);
    
    /**
     * @brief Compute atmospheric properties
     * @param altitude Altitude [m]
     * @return Pair of (density, pressure) [kg/m³, Pa]
     */
    std::pair<double, double> computeProperties(double altitude) const override;
    
    /**
     * @brief Compute density
     * @param altitude Altitude [m]
     * @return Density [kg/m³]
     */
    double computeDensity(double altitude) const override;
    
    /**
     * @brief Compute pressure
     * @param altitude Altitude [m]
     * @return Pressure [Pa]
     */
    double computePressure(double altitude) const override;
    
    /**
     * @brief Compute temperature
     * @param altitude Altitude [m]
     * @return Temperature [K]
     */
    double computeTemperature(double altitude) const override;
    
    /**
     * @brief Compute speed of sound
     * @param altitude Altitude [m]
     * @return Speed of sound [m/s]
     */
    double computeSpeedOfSound(double altitude) const override;
    
    /**
     * @brief Compute Mach number
     * @param velocity Velocity magnitude [m/s]
     * @param altitude Altitude [m]
     * @return Mach number
     */
    double computeMachNumber(double velocity, double altitude) const override;

private:
    double rho0_;        // Sea level density [kg/m³]
    double h_scale_;      // Scale height [m]
    double T0_;          // Sea level temperature [K]
    double R_;           // Gas constant for air [J/(kg·K)]
    double gamma_;       // Ratio of specific heats
};

/**
 * @brief Wind model interface
 */
class WindModel {
public:
    /**
     * @brief Destructor
     */
    virtual ~WindModel() = default;
    
    /**
     * @brief Compute wind velocity at given position and time
     * @param position Position vector [m]
     * @param t Time [s]
     * @return Wind velocity vector [m/s]
     */
    virtual Vec3 computeWind(const Vec3& position, double t) const = 0;
};

/**
 * @brief No wind model
 */
class NoWindModel : public WindModel {
public:
    /**
     * @brief Compute wind velocity (always zero)
     * @param position Position vector [m]
     * @param t Time [s]
     * @return Zero wind velocity
     */
    Vec3 computeWind(const Vec3& position, double t) const override {
        return Vec3::Zero();
    }
};

/**
 * @brief Constant wind model
 */
class ConstantWindModel : public WindModel {
public:
    /**
     * @brief Constructor
     * @param wind_velocity Constant wind velocity [m/s]
     */
    explicit ConstantWindModel(const Vec3& wind_velocity);
    
    /**
     * @brief Compute wind velocity
     * @param position Position vector [m]
     * @param t Time [s]
     * @return Constant wind velocity
     */
    Vec3 computeWind(const Vec3& position, double t) const override;

private:
    Vec3 wind_velocity_;
};

/**
 * @brief Altitude-dependent wind model
 */
class AltitudeWindModel : public WindModel {
public:
    /**
     * @brief Constructor
     * @param wind_profile Wind velocity as function of altitude
     */
    explicit AltitudeWindModel(std::function<Vec3(double)> wind_profile);
    
    /**
     * @brief Compute wind velocity
     * @param position Position vector [m]
     * @param t Time [s]
     * @return Wind velocity at given altitude
     */
    Vec3 computeWind(const Vec3& position, double t) const override;

private:
    std::function<Vec3(double)> wind_profile_;
};

/**
 * @brief Factory functions
 */

/**
 * @brief Create ISA atmosphere model
 * @return Shared pointer to ISA atmosphere
 */
std::shared_ptr<ISAAtmosphere> createISAAtmosphere();

/**
 * @brief Create exponential atmosphere model
 * @param rho0 Sea level density [kg/m³]
 * @param h_scale Scale height [m]
 * @param T0 Sea level temperature [K]
 * @return Shared pointer to exponential atmosphere
 */
std::shared_ptr<ExponentialAtmosphere> createExponentialAtmosphere(double rho0 = 1.225,
                                                                   double h_scale = 8400.0,
                                                                   double T0 = 288.15);

/**
 * @brief Create no wind model
 * @return Shared pointer to no wind model
 */
std::shared_ptr<NoWindModel> createNoWindModel();

/**
 * @brief Create constant wind model
 * @param wind_velocity Constant wind velocity [m/s]
 * @return Shared pointer to constant wind model
 */
std::shared_ptr<ConstantWindModel> createConstantWindModel(const Vec3& wind_velocity);

/**
 * @brief Create altitude-dependent wind model
 * @param wind_profile Wind velocity as function of altitude
 * @return Shared pointer to altitude wind model
 */
std::shared_ptr<AltitudeWindModel> createAltitudeWindModel(std::function<Vec3(double)> wind_profile);

} // namespace rocket_physics
