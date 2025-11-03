#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <functional>
#include <memory>

namespace rocket_physics {

/**
 * @brief 6-DOF rocket dynamics class
 * 
 * Implements the complete 6-degree-of-freedom rocket dynamics including:
 * - Translational motion (position, velocity)
 * - Rotational motion (quaternion, angular velocity)
 * - Mass dynamics
 * - Atmospheric effects
 * - Wind effects
 * - Dynamic pressure monitoring
 */
class Dynamics {
public:
    /**
     * @brief Constructor
     * @param phys Physical parameters
     * @param limits Operational limits
     * @param gravity_func Gravity function callback
     * @param wind_func Wind function callback
     */
    Dynamics(const Phys& phys, const Limits& limits,
             GravityFunc gravity_func = nullptr,
             WindFunc wind_func = nullptr);
    
    /**
     * @brief Destructor
     */
    ~Dynamics() = default;
    
    /**
     * @brief Compute state derivative
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return State derivative
     */
    State computeDerivative(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Compute forces and moments
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Pair of (forces, moments) in body frame
     */
    std::pair<Vec3, Vec3> computeForcesAndMoments(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Compute atmospheric properties
     * @param altitude Altitude [m]
     * @return Pair of (density, pressure) [kg/m³, Pa]
     */
    std::pair<double, double> computeAtmosphericProperties(double altitude) const;
    
    /**
     * @brief Compute dynamic pressure
     * @param state Current state
     * @return Dynamic pressure [Pa]
     */
    double computeDynamicPressure(const State& state) const;
    
    /**
     * @brief Compute angle of attack
     * @param state Current state
     * @return Angle of attack [rad]
     */
    double computeAngleOfAttack(const State& state) const;
    
    /**
     * @brief Compute load factor
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Load factor [g]
     */
    double computeLoadFactor(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Check constraint violations
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Diagnostic information
     */
    Diag checkConstraints(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Set gravity function
     * @param gravity_func Gravity function callback
     */
    void setGravityFunction(GravityFunc gravity_func);
    
    /**
     * @brief Set wind function
     * @param wind_func Wind function callback
     */
    void setWindFunction(WindFunc wind_func);
    
    /**
     * @brief Get physical parameters
     * @return Reference to physical parameters
     */
    const Phys& getPhys() const { return phys_; }
    
    /**
     * @brief Get operational limits
     * @return Reference to operational limits
     */
    const Limits& getLimits() const { return limits_; }

private:
    // Physical parameters
    Phys phys_;
    Limits limits_;
    
    // Function callbacks
    GravityFunc gravity_func_;
    WindFunc wind_func_;
    
    // Internal computation methods
    Vec3 computeGravity(const Vec3& position) const;
    Vec3 computeWind(const Vec3& position, double t) const;
    Vec3 computeAerodynamicForces(const State& state, const Vec3& wind) const;
    
    /**
     * @brief Compute aerodynamic lift force
     * @param state Current state
     * @param wind Wind velocity
     * @return Lift force in body frame [N]
     */
    Vec3 computeLiftForce(const State& state, const Vec3& wind) const;
    
    /**
     * @brief Compute aerodynamic drag force
     * @param state Current state
     * @param wind Wind velocity
     * @return Drag force in body frame [N]
     */
    Vec3 computeDragForce(const State& state, const Vec3& wind) const;
    Vec3 computeThrustForces(const Control& control) const;
    Vec3 computeAerodynamicMoments(const State& state, const Vec3& wind) const;
    Vec3 computeThrustMoments(const Control& control) const;
    double computeMassFlowRate(const Control& control) const;
    
    // Atmospheric model
    double computeDensity(double altitude) const;
    double computePressure(double altitude) const;
    double computeTemperature(double altitude) const;
    
    // Coordinate transformations
    Vec3 transformToBodyFrame(const Vec3& inertial_vector, const Quaterniond& q_bi) const;
    Vec3 transformToInertialFrame(const Vec3& body_vector, const Quaterniond& q_bi) const;
    
    // Utility functions
    Vec3 computeRelativeVelocity(const State& state, const Vec3& wind) const;
    double computeMachNumber(const State& state) const;
    double computeReynoldsNumber(const State& state) const;
};

/**
 * @brief Factory function to create dynamics object
 * @param phys Physical parameters
 * @param limits Operational limits
 * @param gravity_func Gravity function callback
 * @param wind_func Wind function callback
 * @return Shared pointer to dynamics object
 */
std::shared_ptr<Dynamics> createDynamics(const Phys& phys, const Limits& limits,
                                        GravityFunc gravity_func = nullptr,
                                        WindFunc wind_func = nullptr);

} // namespace rocket_physics
