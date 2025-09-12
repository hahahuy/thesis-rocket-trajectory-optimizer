#include "physics/dynamics.hpp"
#include <algorithm>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <stdexcept>
#include "physics/environment/isa_atmosphere.hpp"
#include "physics/aerodynamics/aerodynamics.hpp"
#include "physics/environment/earth_rotation.hpp"

namespace physics {

AscentDynamics::State AscentDynamics::rhs(const State& s, const Control& u, const Params& p) {
    State ds_dt;
    
    // Position derivatives
    ds_dt.x = s.vx;
    ds_dt.y = s.vy;
    
    // Velocity magnitude for drag calculation
    double v_mag = std::sqrt(s.vx * s.vx + s.vy * s.vy);
    
    // Atmospheric density: if user sets rho0==0 treat as vacuum and do not override with ISA
    double rho = atmospheric_density(s.y, p);
    physics::environment::AtmosphereState isa{};
    if (s.y >= 0.0 && p.rho0 > 0.0) {
        isa = physics::environment::ISA_Atmosphere::compute_properties(s.y);
        rho = isa.density;
    }
    
    // Wind effects (if enabled)
    double wind_x = 0.0, wind_y = 0.0;
    if (p.enable_wind) {
        auto wind = wind_velocity(s.y, 0.0, p); // Time not implemented yet
        wind_x = wind.first;
        wind_y = wind.second;
    }
    
    // Relative velocity (velocity relative to air)
    double v_rel_x = s.vx - wind_x;
    double v_rel_y = s.vy - wind_y;
    double v_rel_mag = std::sqrt(v_rel_x * v_rel_x + v_rel_y * v_rel_y);
    
    // Drag force magnitude using Mach-dependent Cd when possible
    double drag_mag = 0.0;
    if (v_rel_mag > 1e-6) {
        double a = (isa.speed_of_sound > 1e-9) ? isa.speed_of_sound : std::sqrt(1.4 * 287.05287 * 288.15);
        double mach = v_rel_mag / a;
        physics::aerodynamics::AeroParams aeroParams; // default params
        double Cd_eff = physics::aerodynamics::drag_coefficient(mach, 0.0, aeroParams);
        drag_mag = 0.5 * rho * Cd_eff * p.A * v_rel_mag * v_rel_mag;
    }
    
    // Thrust components
    double thrust_x = u.T * std::cos(u.theta);
    double thrust_y = u.T * std::sin(u.theta);
    
    // Drag components (opposing relative velocity)
    double drag_x = 0.0, drag_y = 0.0;
    if (v_rel_mag > 1e-6) {
        drag_x = -drag_mag * v_rel_x / v_rel_mag;
        drag_y = -drag_mag * v_rel_y / v_rel_mag;
    }
    
    // Gravity and Earth rotation corrections
    double g = gravity(s.y, p);
    std::pair<double,double> cent{0.0, 0.0};
    std::pair<double,double> cor{0.0, 0.0};
    if (p.earth.enable_centrifugal) {
        cent = physics::environment::centrifugal_accel_2d(s.x, s.y, p.earth);
    }
    if (p.earth.enable_coriolis) {
        cor = physics::environment::coriolis_accel_2d(s.vx, s.vy, p.earth);
    }
    double weight_y = -s.m * g + s.m * cent.second; // reduce effective gravity if enabled
    
    // Acceleration (Newton's second law)
    if (s.m > 1e-3) { // Avoid division by near-zero mass
        ds_dt.vx = (thrust_x + drag_x) / s.m + cor.first;
        ds_dt.vy = (thrust_y + drag_y + weight_y) / s.m + cor.second;
    } else {
        ds_dt.vx = 0.0;
        ds_dt.vy = -g; // Free fall if no mass
    }
    
    // Mass flow rate (propellant consumption)
    // dm/dt = -T / (Isp * g0)
    if (u.T > 1e-6 && s.m > p.m_dry) { // Only consume if thrusting and fuel available
        ds_dt.m = -u.T / (p.Isp * p.g0);
    } else {
        ds_dt.m = 0.0; // No consumption if no thrust or out of fuel
    }
    
    // Ensure mass doesn't go below dry mass
    if (s.m + ds_dt.m < p.m_dry) {
        ds_dt.m = std::max(0.0, p.m_dry - s.m);
    }
    
    return ds_dt;
}

double AscentDynamics::atmospheric_density(double altitude, const Params& p) {
    // Exponential atmosphere model: rho(h) = rho0 * exp(-h/H)
    return p.rho0 * std::exp(-altitude / p.H);
}

std::pair<double, double> AscentDynamics::wind_velocity(double altitude, double time, const Params& p) {
    // Simple wind model - can be extended for more complex profiles
    if (!p.enable_wind) {
        return {0.0, 0.0};
    }
    
    // Example: Linear wind profile with altitude + simple gust
    double wind_x = 0.1 * altitude / 1000.0; // 0.1 m/s per km altitude
    
    // Simple gust model (sinusoidal)
    double gust_freq = 0.1; // Hz
    double gust_amplitude = 5.0; // m/s
    wind_x += gust_amplitude * std::sin(2.0 * M_PI * gust_freq * time);
    
    double wind_y = 0.0; // No vertical wind for now
    
    return {wind_x, wind_y};
}

double AscentDynamics::gravity(double altitude, const Params& p) {
    // Constant gravity for now - can be extended for 1/r² variation
    // For more accuracy: g(h) = g0 * (R/(R+h))²
    if (altitude < 100000.0) { // Below 100 km, use constant g
        return p.g0;
    } else {
        // Use 1/r² law for high altitudes
        double R = p.R_earth;
        double g_var = p.g0 * (R * R) / ((R + altitude) * (R + altitude));
        return g_var;
    }
}

// ForwardIntegrator implementation

std::vector<AscentDynamics::State> ForwardIntegrator::integrate_rk4(
    const RHSFunction& rhs,
    const AscentDynamics::State& s0,
    const std::vector<AscentDynamics::Control>& controls,
    const AscentDynamics::Params& params,
    double dt
) {
    std::vector<AscentDynamics::State> trajectory;
    trajectory.reserve(controls.size() + 1);
    trajectory.push_back(s0);
    
    AscentDynamics::State current_state = s0;
    
    for (size_t i = 0; i < controls.size(); ++i) {
        current_state = rk4_step(rhs, current_state, controls[i], params, dt);
        trajectory.push_back(current_state);
    }
    
    return trajectory;
}

std::vector<std::pair<double, AscentDynamics::State>> ForwardIntegrator::integrate_rk45(
    const RHSFunction& rhs,
    const AscentDynamics::State& s0,
    const std::function<AscentDynamics::Control(double)>& control_func,
    const AscentDynamics::Params& params,
    double t0,
    double tf,
    double rtol,
    double atol,
    double max_step
) {
    std::vector<std::pair<double, AscentDynamics::State>> trajectory;
    trajectory.reserve(static_cast<size_t>((tf - t0) / max_step * 2)); // Rough estimate
    
    double t = t0;
    AscentDynamics::State s = s0;
    double dt = std::min(max_step, (tf - t0) / 100.0); // Initial step size
    
    trajectory.emplace_back(t, s);
    
    const double min_step = 1e-8;
    const double max_step_growth = 5.0;
    const double safety_factor = 0.9;
    
    while (t < tf) {
        if (t + dt > tf) {
            dt = tf - t;
        }
        
        AscentDynamics::Control u = control_func(t + dt/2.0); // Midpoint control
        
        // Take RK45 step
        auto [s_new, s_error] = rk45_step(rhs, s, u, params, dt);
        
        // Compute error norm
        double error = error_norm(s_error, s_new, rtol, atol);
        
        if (error <= 1.0) {
            // Accept step
            t += dt;
            s = s_new;
            trajectory.emplace_back(t, s);
            
            // Adapt step size for next iteration
            if (error > 0.0) {
                double factor = safety_factor * std::pow(1.0 / error, 0.2);
                dt = std::min(max_step, std::min(max_step_growth * dt, factor * dt));
            } else {
                dt = std::min(max_step, max_step_growth * dt);
            }
        } else {
            // Reject step and reduce step size
            double factor = safety_factor * std::pow(1.0 / error, 0.25);
            dt = std::max(min_step, factor * dt);
            
            if (dt < min_step) {
                throw std::runtime_error("RK45 integrator: Step size became too small");
            }
        }
    }
    
    return trajectory;
}

AscentDynamics::State ForwardIntegrator::rk4_step(
    const RHSFunction& rhs,
    const AscentDynamics::State& s,
    const AscentDynamics::Control& u,
    const AscentDynamics::Params& p,
    double dt
) {
    // Classic RK4: k1, k2, k3, k4
    auto k1 = rhs(s, u, p);
    auto k2 = rhs(s + k1 * (dt/2.0), u, p);
    auto k3 = rhs(s + k2 * (dt/2.0), u, p);
    auto k4 = rhs(s + k3 * dt, u, p);
    
    return s + (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt/6.0);
}

std::pair<AscentDynamics::State, AscentDynamics::State> ForwardIntegrator::rk45_step(
    const RHSFunction& rhs,
    const AscentDynamics::State& s,
    const AscentDynamics::Control& u,
    const AscentDynamics::Params& p,
    double dt
) {
    // Dormand-Prince 5(4) coefficients
    static const double a21 = 1.0/5.0;
    static const double a31 = 3.0/40.0, a32 = 9.0/40.0;
    static const double a41 = 44.0/45.0, a42 = -56.0/15.0, a43 = 32.0/9.0;
    static const double a51 = 19372.0/6561.0, a52 = -25360.0/2187.0, a53 = 64448.0/6561.0, a54 = -212.0/729.0;
    static const double a61 = 9017.0/3168.0, a62 = -355.0/33.0, a63 = 46732.0/5247.0, a64 = 49.0/176.0, a65 = -5103.0/18656.0;
    
    // 5th order solution coefficients
    static const double b1 = 35.0/384.0, b3 = 500.0/1113.0, b4 = 125.0/192.0, b5 = -2187.0/6784.0, b6 = 11.0/84.0;
    
    // 4th order solution coefficients (for error estimation)
    static const double bh1 = 5179.0/57600.0, bh3 = 7571.0/16695.0, bh4 = 393.0/640.0, bh5 = -92097.0/339200.0, bh6 = 187.0/2100.0, bh7 = 1.0/40.0;
    
    // RK stages
    auto k1 = rhs(s, u, p);
    auto k2 = rhs(s + k1 * (a21 * dt), u, p);
    auto k3 = rhs(s + (k1 * a31 + k2 * a32) * dt, u, p);
    auto k4 = rhs(s + (k1 * a41 + k2 * a42 + k3 * a43) * dt, u, p);
    auto k5 = rhs(s + (k1 * a51 + k2 * a52 + k3 * a53 + k4 * a54) * dt, u, p);
    auto k6 = rhs(s + (k1 * a61 + k2 * a62 + k3 * a63 + k4 * a64 + k5 * a65) * dt, u, p);
    
    // 5th order solution
    auto s_new = s + (k1 * b1 + k3 * b3 + k4 * b4 + k5 * b5 + k6 * b6) * dt;
    
    // 4th order solution for error estimation
    auto k7 = rhs(s_new, u, p);
    auto s_4th = s + (k1 * bh1 + k3 * bh3 + k4 * bh4 + k5 * bh5 + k6 * bh6 + k7 * bh7) * dt;
    
    // Error estimate
    auto error = s_new + s_4th * (-1.0);
    
    return {s_new, error};
}

double ForwardIntegrator::error_norm(const AscentDynamics::State& error, 
                                    const AscentDynamics::State& s,
                                    double rtol, double atol) {
    // Compute mixed absolute/relative error norm
    double norm = 0.0;
    
    // Position errors (scale by 1000 to convert m to km for better numerics)
    double scale_x = atol + rtol * std::abs(s.x) / 1000.0;
    double scale_y = atol + rtol * std::abs(s.y) / 1000.0;
    norm += (error.x / (1000.0 * scale_x)) * (error.x / (1000.0 * scale_x));
    norm += (error.y / (1000.0 * scale_y)) * (error.y / (1000.0 * scale_y));
    
    // Velocity errors
    double scale_vx = atol + rtol * std::abs(s.vx);
    double scale_vy = atol + rtol * std::abs(s.vy);
    norm += (error.vx / scale_vx) * (error.vx / scale_vx);
    norm += (error.vy / scale_vy) * (error.vy / scale_vy);
    
    // Mass error
    double scale_m = atol + rtol * std::abs(s.m);
    norm += (error.m / scale_m) * (error.m / scale_m);
    
    return std::sqrt(norm / 5.0); // Divide by number of states
}

std::vector<std::pair<double, AscentDynamics::State>> ForwardIntegrator::integrate_rk45_with_staging(
    const RHSFunction& rhs,
    const AscentDynamics::State& s0,
    const std::function<AscentDynamics::Control(double, int)> &control_func,
    AscentDynamics::Params& params,
    double t0,
    double tf,
    double rtol,
    double atol,
    double max_step
) {
    std::vector<std::pair<double, AscentDynamics::State>> trajectory;
    trajectory.reserve(static_cast<size_t>((tf - t0) / max_step * 2));

    double t = t0;
    AscentDynamics::State s = s0;
    double dt = std::min(max_step, (tf - t0) / 100.0);
    trajectory.emplace_back(t, s);

    const double min_step = 1e-8;
    const double max_step_growth = 5.0;
    const double safety_factor = 0.9;

    int stage_index = params.vehicle ? params.vehicle->current_stage : 0;

    while (t < tf) {
        if (t + dt > tf) dt = tf - t;

        AscentDynamics::Control u = control_func(t + dt/2.0, stage_index);

        auto [s_new, s_error] = rk45_step(rhs, s, u, params, dt);
        double error = error_norm(s_error, s_new, rtol, atol);

        if (error <= 1.0) {
            t += dt;
            s = s_new;

            // Staging check
            if (params.vehicle) {
                const auto &veh = *params.vehicle;
                if (stage_index >= 0 && stage_index < static_cast<int>(veh.stages.size())) {
                    const auto &st = veh.stages[stage_index];
                    double stage_dry_mass = st.dry_mass;
                    if (s.m <= stage_dry_mass + 1e-6) {
                        // Perform separation: drop dry mass of current stage and advance stage
                        double mass_before = s.m;
                        physics::propulsion::StagingLogic::perform_separation(s.m, *params.vehicle);
                        stage_index = params.vehicle->current_stage;
                        // Ensure mass is not negative
                        if (s.m < 0.0) s.m = 0.0;
                    }
                }
            }

            trajectory.emplace_back(t, s);

            if (error > 0.0) {
                double factor = safety_factor * std::pow(1.0 / error, 0.2);
                dt = std::min(max_step, std::min(max_step_growth * dt, factor * dt));
            } else {
                dt = std::min(max_step, max_step_growth * dt);
            }
        } else {
            double factor = safety_factor * std::pow(1.0 / error, 0.25);
            dt = std::max(min_step, factor * dt);
            if (dt < min_step) {
                throw std::runtime_error("RK45 integrator (staging): Step size became too small");
            }
        }
    }

    return trajectory;
}

} // namespace physics
