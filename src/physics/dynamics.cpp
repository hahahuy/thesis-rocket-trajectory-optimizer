#include "dynamics.hpp"
#include <cmath>
#include <algorithm>

namespace rocket_physics {

Dynamics::Dynamics(const Phys& phys, const Limits& limits,
                   GravityFunc gravity_func, WindFunc wind_func)
    : phys_(phys), limits_(limits), gravity_func_(gravity_func), wind_func_(wind_func) {
}

State Dynamics::computeDerivative(const State& state, const Control& control, double t) const {
    // Compute forces and moments
    auto [forces, moments] = computeForcesAndMoments(state, control, t);
    
    // State derivative
    State state_dot;
    
    // Position derivative: r_dot = v
    state_dot.r_i = state.v_i;
    
    // Velocity derivative: v_dot = F/m (in inertial frame)
    Vec3 forces_inertial = transformToInertialFrame(forces, state.q_bi);
    state_dot.v_i = forces_inertial / state.m;
    
    // Quaternion derivative: q_dot = 0.5 * q * [0, ω]
    Quaterniond omega_quat(0.0, state.w_b.x(), state.w_b.y(), state.w_b.z());
    Quaterniond q_dot = state.q_bi * omega_quat;
    state_dot.q_bi = Quaterniond(0.5 * q_dot.w(), 0.5 * q_dot.x(), 0.5 * q_dot.y(), 0.5 * q_dot.z());
    
    // Angular velocity derivative: ω_dot = I^(-1) * (M - ω × Iω)
    Vec3 angular_momentum = phys_.I_b * state.w_b;
    Vec3 coriolis_moment = state.w_b.cross(angular_momentum);
    state_dot.w_b = phys_.I_b.inverse() * (moments - coriolis_moment);
    
    // Mass derivative: m_dot = -ṁ (mass flow rate)
    state_dot.m = -computeMassFlowRate(control);
    
    return state_dot;
}

std::pair<Vec3, Vec3> Dynamics::computeForcesAndMoments(const State& state, const Control& control, double t) const {
    // Compute wind
    Vec3 wind = computeWind(state.r_i, t);
    
    // Compute forces
    Vec3 gravity_force = computeGravity(state.r_i) * state.m;
    Vec3 aero_force = computeAerodynamicForces(state, wind);
    Vec3 thrust_force = computeThrustForces(control);
    
    // Total forces in body frame
    Vec3 total_forces = transformToBodyFrame(gravity_force, state.q_bi) + aero_force + thrust_force;
    
    // Compute moments
    Vec3 aero_moment = computeAerodynamicMoments(state, wind);
    Vec3 thrust_moment = computeThrustMoments(control);
    
    // Add control surface moment if delta is non-zero
    if (std::abs(control.delta) > 1e-6) {
        double altitude = std::max(0.0, state.r_i.norm() - 6371000.0);
        double density = computeDensity(altitude);
        Vec3 relative_velocity = computeRelativeVelocity(state, wind);
        double speed = relative_velocity.norm();
        double q = 0.5 * density * speed * speed;
        
        // Control surface moment: M_delta = q * S_ref * l_ref * C_delta * delta
        double M_delta = q * phys_.S_ref * phys_.l_ref * phys_.C_delta * control.delta;
        aero_moment.y() += M_delta; // Add to pitch moment
    }
    
    // Total moments in body frame
    Vec3 total_moments = aero_moment + thrust_moment;
    
    return {total_forces, total_moments};
}

std::pair<double, double> Dynamics::computeAtmosphericProperties(double altitude) const {
    double density = computeDensity(altitude);
    double pressure = computePressure(altitude);
    return {density, pressure};
}

double Dynamics::computeDynamicPressure(const State& state) const {
    Vec3 wind = computeWind(state.r_i, 0.0); // Assume steady wind for now
    Vec3 relative_velocity = computeRelativeVelocity(state, wind);
    double speed = relative_velocity.norm();
    
    double altitude = state.r_i.norm() - 6371000.0; // Earth radius
    double density = computeDensity(altitude);
    
    return 0.5 * density * speed * speed;
}

double Dynamics::computeAngleOfAttack(const State& state) const {
    Vec3 wind = computeWind(state.r_i, 0.0);
    Vec3 relative_velocity = computeRelativeVelocity(state, wind);
    
    if (relative_velocity.norm() < 1e-6) {
        return 0.0;
    }
    
    // Transform relative velocity to body frame
    Vec3 v_rel_body = transformToBodyFrame(relative_velocity, state.q_bi);
    
    // Compute angle of attack (angle between velocity and body x-axis)
    double vx = v_rel_body.x();
    double vy = v_rel_body.y();
    double vz = v_rel_body.z();
    
    return std::atan2(std::sqrt(vy*vy + vz*vz), vx);
}

double Dynamics::computeLoadFactor(const State& state, const Control& control, double t) const {
    // Compute normal load factor: n = L / (m * g0)
    // Where L is the aerodynamic lift force
    
    // Get wind and relative velocity
    Vec3 wind = computeWind(state.r_i, t);
    Vec3 relative_velocity = computeRelativeVelocity(state, wind);
    double speed = relative_velocity.norm();
    
    if (speed < 1e-6) {
        return 1.0; // No aerodynamic load at zero velocity
    }
    
    // Compute altitude and atmospheric density
    double altitude = std::max(0.0, state.r_i.norm() - 6371000.0); // Earth radius
    double density = computeDensity(altitude);
    
    // Dynamic pressure
    double q = 0.5 * density * speed * speed;
    
    // Compute angle of attack
    Vec3 v_rel_body = transformToBodyFrame(relative_velocity, state.q_bi);
    double vx = v_rel_body.x();
    double vy = v_rel_body.y();
    double vz = v_rel_body.z();
    double alpha = std::atan2(std::sqrt(vy*vy + vz*vz), vx);
    
    // Lift force: L = q * S_ref * CL_alpha * alpha
    double L = q * phys_.S_ref * phys_.CL_alpha * alpha;
    
    // Normal load factor in g's
    double n = std::abs(L / (state.m * phys_.g0));
    
    // Add baseline 1g for gravity
    return n + 1.0;
}

Diag Dynamics::checkConstraints(const State& state, const Control& control, double t) const {
    Diag diag;
    
    // Compute atmospheric properties
    double altitude = std::max(0.0, state.r_i.norm() - 6371000.0); // Earth radius
    auto [density, pressure] = computeAtmosphericProperties(altitude);
    diag.rho = density;
    
    // Compute dynamic pressure
    double q = computeDynamicPressure(state);
    diag.q = q;
    diag.q_violation = (q > limits_.q_max);
    
    // Check mass underflow
    diag.m_underflow = (state.m < limits_.m_dry);
    
    // Compute angle of attack
    diag.alpha = computeAngleOfAttack(state);
    
    // Compute load factor
    diag.n = computeLoadFactor(state, control, t);
    diag.n_violation = (diag.n > limits_.n_max);
    
    return diag;
}

void Dynamics::setGravityFunction(GravityFunc gravity_func) {
    gravity_func_ = gravity_func;
}

void Dynamics::setWindFunction(WindFunc wind_func) {
    wind_func_ = wind_func;
}

Vec3 Dynamics::computeGravity(const Vec3& position) const {
    if (gravity_func_) {
        return gravity_func_(position);
    }
    
    // Default: simple inverse square law
    double r = position.norm();
    double g_magnitude = 9.81 * (6371000.0 / r) * (6371000.0 / r); // Earth radius = 6371000 m
    return -g_magnitude * position.normalized();
}

Vec3 Dynamics::computeWind(const Vec3& position, double t) const {
    if (wind_func_) {
        return wind_func_(position, t);
    }
    
    // Default: no wind
    return Vec3::Zero();
}

Vec3 Dynamics::computeAerodynamicForces(const State& state, const Vec3& wind) const {
    Vec3 relative_velocity = computeRelativeVelocity(state, wind);
    double speed = relative_velocity.norm();
    
    if (speed < 1e-6) {
        return Vec3::Zero();
    }
    
    // Compute atmospheric properties
    double altitude = state.r_i.norm() - 6371000.0;
    double density = computeDensity(altitude);
    
    // Dynamic pressure
    double q = 0.5 * density * speed * speed;
    
    // Transform relative velocity to body frame
    Vec3 v_rel_body = transformToBodyFrame(relative_velocity, state.q_bi);
    Vec3 v_rel_unit = v_rel_body.normalized();
    
    // Drag force (opposite to velocity direction)
    Vec3 drag_force = -q * phys_.S_ref * phys_.Cd * v_rel_unit;
    
    // Lift force (perpendicular to velocity direction)
    Vec3 lift_force = q * phys_.S_ref * phys_.Cl * Vec3(0, -v_rel_unit.z(), v_rel_unit.y());
    
    return drag_force + lift_force;
}

Vec3 Dynamics::computeThrustForces(const Control& control) const {
    return control.T * control.uT_b;
}

Vec3 Dynamics::computeAerodynamicMoments(const State& state, const Vec3& wind) const {
    // Compute aerodynamic moments including tail/wing effects
    
    Vec3 relative_velocity = computeRelativeVelocity(state, wind);
    double speed = relative_velocity.norm();
    
    if (speed < 1e-6) {
        return Vec3::Zero();
    }
    
    // Compute atmospheric properties
    double altitude = std::max(0.0, state.r_i.norm() - 6371000.0);
    double density = computeDensity(altitude);
    double q = 0.5 * density * speed * speed;
    
    // Transform relative velocity to body frame
    Vec3 v_rel_body = transformToBodyFrame(relative_velocity, state.q_bi);
    double vx = v_rel_body.x();
    double vy = v_rel_body.y();
    double vz = v_rel_body.z();
    
    // Angle of attack
    double alpha = std::atan2(std::sqrt(vy*vy + vz*vz), vx);
    
    // Pitch moment: M_pitch = q * S_ref * l_ref * (Cm_alpha * alpha + C_delta * delta)
    // Note: delta is obtained from control, but we need to pass it somehow
    // For now, use only Cm_alpha term (control surface deflection handled separately if needed)
    double M_pitch = q * phys_.S_ref * phys_.l_ref * phys_.Cm_alpha * alpha;
    
    Vec3 moments;
    moments.x() = 0.0; // Roll moment (simplified)
    moments.y() = M_pitch; // Pitch moment
    moments.z() = 0.0; // Yaw moment (simplified)
    
    return moments;
}

Vec3 Dynamics::computeThrustMoments(const Control& control) const {
    // Thrust gimbal creates moments about the center of gravity
    // Moment = r_cg × F_thrust
    return phys_.r_cg.cross(control.T * control.uT_b);
}

double Dynamics::computeMassFlowRate(const Control& control) const {
    if (control.T <= 0.0) {
        return 0.0;
    }
    
    // Mass flow rate = T / (Isp * g0)
    return control.T / (phys_.Isp * phys_.g0);
}

double Dynamics::computeDensity(double altitude) const {
    if (altitude < 0.0) {
        return phys_.rho0;
    }
    
    // Exponential atmosphere model
    return phys_.rho0 * std::exp(-altitude / phys_.h_scale);
}

double Dynamics::computePressure(double altitude) const {
    if (altitude < 0.0) {
        return 101325.0; // Sea level pressure
    }
    
    // Simplified pressure model
    double density = computeDensity(altitude);
    double temperature = 288.15 - 0.0065 * altitude; // Linear temperature model
    double R = 287.0; // Gas constant for air
    
    return density * R * temperature;
}

double Dynamics::computeTemperature(double altitude) const {
    if (altitude < 0.0) {
        return 288.15; // Sea level temperature
    }
    
    // Linear temperature model
    return 288.15 - 0.0065 * altitude;
}

Vec3 Dynamics::transformToBodyFrame(const Vec3& inertial_vector, const Quaterniond& q_bi) const {
    return q_bi.inverse() * inertial_vector;
}

Vec3 Dynamics::transformToInertialFrame(const Vec3& body_vector, const Quaterniond& q_bi) const {
    return q_bi * body_vector;
}

Vec3 Dynamics::computeRelativeVelocity(const State& state, const Vec3& wind) const {
    return state.v_i - wind;
}

double Dynamics::computeMachNumber(const State& state) const {
    double altitude = state.r_i.norm() - 6371000.0;
    double temperature = computeTemperature(altitude);
    double speed_of_sound = std::sqrt(1.4 * 287.0 * temperature); // γ = 1.4, R = 287
    return state.v_i.norm() / speed_of_sound;
}

double Dynamics::computeReynoldsNumber(const State& state) const {
    double altitude = state.r_i.norm() - 6371000.0;
    double density = computeDensity(altitude);
    double temperature = computeTemperature(altitude);
    double viscosity = 1.81e-5 * std::pow(temperature / 288.15, 0.7); // Sutherland's law
    double characteristic_length = 1.0; // Assume 1m characteristic length
    
    return density * state.v_i.norm() * characteristic_length / viscosity;
}

std::shared_ptr<Dynamics> createDynamics(const Phys& phys, const Limits& limits,
                                        GravityFunc gravity_func, WindFunc wind_func) {
    return std::make_shared<Dynamics>(phys, limits, gravity_func, wind_func);
}

} // namespace rocket_physics
