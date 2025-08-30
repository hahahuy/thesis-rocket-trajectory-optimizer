#pragma once

#include <functional>
#include <vector>
#include <cmath>
#include <cassert>

namespace physics {

/**
 * @brief 2D rocket dynamics for ascent trajectory optimization
 * 
 * This module implements planar 3-DOF dynamics for rocket ascent:
 * - Position: (x, y) in inertial frame
 * - Velocity: (vx, vy) in inertial frame  
 * - Mass: m (decreasing due to propellant consumption)
 * 
 * Control inputs:
 * - Thrust magnitude: T [N]
 * - Thrust angle: theta [rad] (measured from horizontal)
 */
class AscentDynamics {
public:
    /**
     * @brief State vector for 2D rocket dynamics
     */
    struct State {
        double x;   ///< Horizontal position [m]
        double y;   ///< Vertical position [m] (altitude)
        double vx;  ///< Horizontal velocity [m/s]
        double vy;  ///< Vertical velocity [m/s]
        double m;   ///< Mass [kg]
        
        State() : x(0), y(0), vx(0), vy(0), m(0) {}
        State(double x_, double y_, double vx_, double vy_, double m_) 
            : x(x_), y(y_), vx(vx_), vy(vy_), m(m_) {}
            
        // Vector operations for integration
        State operator+(const State& other) const {
            return State(x + other.x, y + other.y, vx + other.vx, vy + other.vy, m + other.m);
        }
        
        State operator*(double scalar) const {
            return State(x * scalar, y * scalar, vx * scalar, vy * scalar, m * scalar);
        }
        
        State& operator+=(const State& other) {
            x += other.x; y += other.y; vx += other.vx; vy += other.vy; m += other.m;
            return *this;
        }
    };

    /**
     * @brief Control inputs for rocket
     */
    struct Control {
        double T;      ///< Thrust magnitude [N]
        double theta;  ///< Thrust angle [rad] from horizontal (positive = upward)
        
        Control() : T(0), theta(0) {}
        Control(double T_, double theta_) : T(T_), theta(theta_) {}
    };

    /**
     * @brief Physical parameters for rocket and environment
     */
    struct Params {
        // Rocket parameters
        double Cd;     ///< Drag coefficient [-]
        double A;      ///< Reference area [m²]
        double Isp;    ///< Specific impulse [s]
        double Tmax;   ///< Maximum thrust [N]
        double m_dry;  ///< Dry mass (structure) [kg]
        double m_prop; ///< Propellant mass [kg]
        
        // Environment parameters
        double rho0;   ///< Sea level air density [kg/m³]
        double H;      ///< Scale height [m]
        double g0;     ///< Standard gravity [m/s²]
        double R_earth; ///< Earth radius [m] (for gravity variation if needed)
        
        // Wind parameters (optional)
        bool enable_wind;  ///< Enable wind effects
        
        Params() : Cd(0.3), A(1.0), Isp(300.0), Tmax(100000.0), 
                   m_dry(1000.0), m_prop(4000.0),
                   rho0(1.225), H(8400.0), g0(9.81), R_earth(6.371e6),
                   enable_wind(false) {}
    };

    /**
     * @brief Right-hand side of the ODE system: ds/dt = rhs(s, u, p)
     * 
     * Implements the equations of motion:
     * dx/dt = vx
     * dy/dt = vy
     * dvx/dt = (T*cos(theta) - D*vx/|v|) / m
     * dvy/dt = (T*sin(theta) - D*vy/|v| - m*g) / m
     * dm/dt = -T / (Isp * g0)
     * 
     * Where D = 0.5 * rho(y) * Cd * A * |v|²
     * 
     * @param s Current state
     * @param u Control inputs
     * @param p Physical parameters
     * @return State derivative ds/dt
     */
    static State rhs(const State& s, const Control& u, const Params& p);

    /**
     * @brief Atmospheric density model
     * @param altitude Altitude above sea level [m]
     * @param p Parameters containing rho0 and H
     * @return Air density [kg/m³]
     */
    static double atmospheric_density(double altitude, const Params& p);

    /**
     * @brief Wind velocity model
     * @param altitude Altitude [m]
     * @param time Time [s]
     * @param p Parameters
     * @return Wind velocity components (wx, wy) [m/s]
     */
    static std::pair<double, double> wind_velocity(double altitude, double time, const Params& p);

    /**
     * @brief Gravity model (constant for now, can be extended)
     * @param altitude Altitude [m]
     * @param p Parameters
     * @return Gravitational acceleration [m/s²]
     */
    static double gravity(double altitude, const Params& p);
};

/**
 * @brief Forward integrator class for trajectory simulation
 */
class ForwardIntegrator {
public:
    using RHSFunction = std::function<AscentDynamics::State(const AscentDynamics::State&, 
                                                           const AscentDynamics::Control&, 
                                                           const AscentDynamics::Params&)>;

    /**
     * @brief Fixed-step RK4 integrator
     * @param rhs Right-hand side function
     * @param s0 Initial state
     * @param controls Control sequence (one per time step)
     * @param params Physical parameters
     * @param dt Time step [s]
     * @return Trajectory as vector of states
     */
    static std::vector<AscentDynamics::State> integrate_rk4(
        const RHSFunction& rhs,
        const AscentDynamics::State& s0,
        const std::vector<AscentDynamics::Control>& controls,
        const AscentDynamics::Params& params,
        double dt
    );

    /**
     * @brief Adaptive RK45 integrator (Dormand-Prince method)
     * @param rhs Right-hand side function
     * @param s0 Initial state
     * @param controls Control interpolation function
     * @param params Physical parameters
     * @param t0 Initial time [s]
     * @param tf Final time [s]
     * @param rtol Relative tolerance
     * @param atol Absolute tolerance
     * @param max_step Maximum step size [s]
     * @return Trajectory as (time, state) pairs
     */
    static std::vector<std::pair<double, AscentDynamics::State>> integrate_rk45(
        const RHSFunction& rhs,
        const AscentDynamics::State& s0,
        const std::function<AscentDynamics::Control(double)>& control_func,
        const AscentDynamics::Params& params,
        double t0,
        double tf,
        double rtol = 1e-6,
        double atol = 1e-8,
        double max_step = 1.0
    );

private:
    /**
     * @brief Single RK4 step
     */
    static AscentDynamics::State rk4_step(
        const RHSFunction& rhs,
        const AscentDynamics::State& s,
        const AscentDynamics::Control& u,
        const AscentDynamics::Params& p,
        double dt
    );

    /**
     * @brief Single RK45 step with error estimation
     */
    static std::pair<AscentDynamics::State, AscentDynamics::State> rk45_step(
        const RHSFunction& rhs,
        const AscentDynamics::State& s,
        const AscentDynamics::Control& u,
        const AscentDynamics::Params& p,
        double dt
    );
    
    /**
     * @brief Compute error norm for adaptive stepping
     */
    static double error_norm(const AscentDynamics::State& error, 
                           const AscentDynamics::State& s,
                           double rtol, double atol);
};

} // namespace physics
