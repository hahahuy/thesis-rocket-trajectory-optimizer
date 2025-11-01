#pragma once

#include "types.hpp"
#include "dynamics.hpp"
#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <vector>

namespace rocket_physics {

/**
 * @brief Base class for numerical integrators
 */
class Integrator {
public:
    /**
     * @brief Constructor
     * @param dynamics Dynamics object
     */
    explicit Integrator(std::shared_ptr<Dynamics> dynamics);
    
    /**
     * @brief Destructor
     */
    virtual ~Integrator() = default;
    
    /**
     * @brief Integrate one step
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @param dt Time step
     * @return New state after integration
     */
    virtual State integrate(const State& state, const Control& control, double t, double dt) = 0;
    
    /**
     * @brief Integrate over time interval
     * @param initial_state Initial state
     * @param control_func Control function (time -> control)
     * @param t_start Start time
     * @param t_end End time
     * @param dt Time step
     * @return Vector of states at each time step
     */
    virtual std::vector<State> integrateInterval(const State& initial_state,
                                                std::function<Control(double)> control_func,
                                                double t_start, double t_end, double dt) = 0;
    
    /**
     * @brief Get dynamics object
     * @return Shared pointer to dynamics
     */
    std::shared_ptr<Dynamics> getDynamics() const { return dynamics_; }

protected:
    std::shared_ptr<Dynamics> dynamics_;
};

/**
 * @brief Runge-Kutta 4th order integrator
 */
class RK4Integrator : public Integrator {
public:
    /**
     * @brief Constructor
     * @param dynamics Dynamics object
     */
    explicit RK4Integrator(std::shared_ptr<Dynamics> dynamics);
    
    /**
     * @brief Integrate one step using RK4
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @param dt Time step
     * @return New state after integration
     */
    State integrate(const State& state, const Control& control, double t, double dt) override;
    
    /**
     * @brief Integrate over time interval using RK4
     * @param initial_state Initial state
     * @param control_func Control function
     * @param t_start Start time
     * @param t_end End time
     * @param dt Time step
     * @return Vector of states
     */
    std::vector<State> integrateInterval(const State& initial_state,
                                       std::function<Control(double)> control_func,
                                       double t_start, double t_end, double dt) override;

private:
    // RK4 coefficients
    static constexpr double k1_coeff = 1.0/6.0;
    static constexpr double k2_coeff = 1.0/3.0;
    static constexpr double k3_coeff = 1.0/3.0;
    static constexpr double k4_coeff = 1.0/6.0;
};

/**
 * @brief Runge-Kutta 4th/5th order adaptive integrator
 */
class RK45Integrator : public Integrator {
public:
    /**
     * @brief Constructor
     * @param dynamics Dynamics object
     * @param tolerance Error tolerance
     * @param min_step Minimum step size
     * @param max_step Maximum step size
     */
    RK45Integrator(std::shared_ptr<Dynamics> dynamics, double tolerance = 1e-6,
                   double min_step = 1e-9, double max_step = 1.0);
    
    /**
     * @brief Integrate one step using RK45
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @param dt Time step
     * @return New state after integration
     */
    State integrate(const State& state, const Control& control, double t, double dt) override;
    
    /**
     * @brief Integrate over time interval using RK45
     * @param initial_state Initial state
     * @param control_func Control function
     * @param t_start Start time
     * @param t_end End time
     * @param dt Initial time step
     * @return Vector of states
     */
    std::vector<State> integrateInterval(const State& initial_state,
                                       std::function<Control(double)> control_func,
                                       double t_start, double t_end, double dt) override;
    
    /**
     * @brief Get last step size used
     * @return Last step size
     */
    double getLastStepSize() const { return last_step_size_; }
    
    /**
     * @brief Get total number of function evaluations
     * @return Number of function evaluations
     */
    int getFunctionEvaluations() const { return function_evaluations_; }

private:
    double tolerance_;
    double min_step_;
    double max_step_;
    double last_step_size_;
    int function_evaluations_;
    
    // RK45 coefficients
    static constexpr double a21 = 1.0/4.0;
    static constexpr double a31 = 3.0/32.0, a32 = 9.0/32.0;
    static constexpr double a41 = 1932.0/2197.0, a42 = -7200.0/2197.0, a43 = 7296.0/2197.0;
    static constexpr double a51 = 439.0/216.0, a52 = -8.0, a53 = 3680.0/513.0, a54 = -845.0/4104.0;
    static constexpr double a61 = -8.0/27.0, a62 = 2.0, a63 = -3544.0/2565.0, a64 = 1859.0/4104.0, a65 = -11.0/40.0;
    
    static constexpr double b1 = 16.0/135.0, b2 = 0.0, b3 = 6656.0/12825.0, b4 = 28561.0/56430.0, b5 = -9.0/50.0, b6 = 2.0/55.0;
    static constexpr double c1 = 25.0/216.0, c2 = 0.0, c3 = 1408.0/2565.0, c4 = 2197.0/4104.0, c5 = -1.0/5.0, c6 = 0.0;
    
    // Helper methods
    State computeRK45Step(const State& state, const Control& control, double t, double dt);
    double computeError(const State& state, const State& state_4th, const State& state_5th);
    double computeOptimalStepSize(double current_dt, double error, double tolerance);
};

/**
 * @brief Factory function to create RK4 integrator
 * @param dynamics Dynamics object
 * @return Shared pointer to RK4 integrator
 */
std::shared_ptr<RK4Integrator> createRK4Integrator(std::shared_ptr<Dynamics> dynamics);

/**
 * @brief Factory function to create RK45 integrator
 * @param dynamics Dynamics object
 * @param tolerance Error tolerance
 * @param min_step Minimum step size
 * @param max_step Maximum step size
 * @return Shared pointer to RK45 integrator
 */
std::shared_ptr<RK45Integrator> createRK45Integrator(std::shared_ptr<Dynamics> dynamics,
                                                    double tolerance = 1e-6,
                                                    double min_step = 1e-9,
                                                    double max_step = 1.0);

} // namespace rocket_physics
