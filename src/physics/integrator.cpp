#include "integrator.hpp"
#include <algorithm>
#include <cmath>

namespace rocket_physics {

// Base Integrator class
Integrator::Integrator(std::shared_ptr<Dynamics> dynamics) : dynamics_(dynamics) {
}

// RK4Integrator implementation
RK4Integrator::RK4Integrator(std::shared_ptr<Dynamics> dynamics) : Integrator(dynamics) {
}

State RK4Integrator::integrate(const State& state, const Control& control, double t, double dt) {
    // RK4 method: k1, k2, k3, k4
    State k1 = dynamics_->computeDerivative(state, control, t);
    State k2 = dynamics_->computeDerivative(
        State(state.r_i + 0.5*dt*k1.r_i, state.v_i + 0.5*dt*k1.v_i, 
              Quaterniond(state.q_bi.w() + 0.5*dt*k1.q_bi.w(), state.q_bi.x() + 0.5*dt*k1.q_bi.x(),
                         state.q_bi.y() + 0.5*dt*k1.q_bi.y(), state.q_bi.z() + 0.5*dt*k1.q_bi.z()),
              state.w_b + 0.5*dt*k1.w_b, state.m + 0.5*dt*k1.m), 
        control, t + 0.5*dt);
    State k3 = dynamics_->computeDerivative(
        State(state.r_i + 0.5*dt*k2.r_i, state.v_i + 0.5*dt*k2.v_i,
              Quaterniond(state.q_bi.w() + 0.5*dt*k2.q_bi.w(), state.q_bi.x() + 0.5*dt*k2.q_bi.x(),
                         state.q_bi.y() + 0.5*dt*k2.q_bi.y(), state.q_bi.z() + 0.5*dt*k2.q_bi.z()),
              state.w_b + 0.5*dt*k2.w_b, state.m + 0.5*dt*k2.m),
        control, t + 0.5*dt);
    State k4 = dynamics_->computeDerivative(
        State(state.r_i + dt*k3.r_i, state.v_i + dt*k3.v_i,
              Quaterniond(state.q_bi.w() + dt*k3.q_bi.w(), state.q_bi.x() + dt*k3.q_bi.x(),
                         state.q_bi.y() + dt*k3.q_bi.y(), state.q_bi.z() + dt*k3.q_bi.z()),
              state.w_b + dt*k3.w_b, state.m + dt*k3.m),
        control, t + dt);
    
    // Combine using RK4 coefficients
    State new_state;
    new_state.r_i = state.r_i + dt * (k1_coeff*k1.r_i + k2_coeff*k2.r_i + k3_coeff*k3.r_i + k4_coeff*k4.r_i);
    new_state.v_i = state.v_i + dt * (k1_coeff*k1.v_i + k2_coeff*k2.v_i + k3_coeff*k3.v_i + k4_coeff*k4.v_i);
    new_state.q_bi = Quaterniond(
        state.q_bi.w() + dt * (k1_coeff*k1.q_bi.w() + k2_coeff*k2.q_bi.w() + k3_coeff*k3.q_bi.w() + k4_coeff*k4.q_bi.w()),
        state.q_bi.x() + dt * (k1_coeff*k1.q_bi.x() + k2_coeff*k2.q_bi.x() + k3_coeff*k3.q_bi.x() + k4_coeff*k4.q_bi.x()),
        state.q_bi.y() + dt * (k1_coeff*k1.q_bi.y() + k2_coeff*k2.q_bi.y() + k3_coeff*k3.q_bi.y() + k4_coeff*k4.q_bi.y()),
        state.q_bi.z() + dt * (k1_coeff*k1.q_bi.z() + k2_coeff*k2.q_bi.z() + k3_coeff*k3.q_bi.z() + k4_coeff*k4.q_bi.z())
    );
    new_state.w_b = state.w_b + dt * (k1_coeff*k1.w_b + k2_coeff*k2.w_b + k3_coeff*k3.w_b + k4_coeff*k4.w_b);
    new_state.m = state.m + dt * (k1_coeff*k1.m + k2_coeff*k2.m + k3_coeff*k3.m + k4_coeff*k4.m);
    
    // Normalize quaternion
    utils::normalizeQuaternion(new_state);
    
    return new_state;
}

std::vector<State> RK4Integrator::integrateInterval(const State& initial_state,
                                                   std::function<Control(double)> control_func,
                                                   double t_start, double t_end, double dt) {
    std::vector<State> states;
    State current_state = initial_state;
    double current_time = t_start;
    
    states.push_back(current_state);
    
    while (current_time < t_end) {
        double step_dt = std::min(dt, t_end - current_time);
        Control current_control = control_func(current_time);
        
        current_state = integrate(current_state, current_control, current_time, step_dt);
        current_time += step_dt;
        
        states.push_back(current_state);
    }
    
    return states;
}

// RK45Integrator implementation
RK45Integrator::RK45Integrator(std::shared_ptr<Dynamics> dynamics, double tolerance,
                               double min_step, double max_step)
    : Integrator(dynamics), tolerance_(tolerance), min_step_(min_step), max_step_(max_step),
      last_step_size_(0.0), function_evaluations_(0) {
}

State RK45Integrator::integrate(const State& state, const Control& control, double t, double dt) {
    function_evaluations_ = 0;
    last_step_size_ = dt;
    
    return computeRK45Step(state, control, t, dt);
}

std::vector<State> RK45Integrator::integrateInterval(const State& initial_state,
                                                    std::function<Control(double)> control_func,
                                                    double t_start, double t_end, double dt) {
    std::vector<State> states;
    State current_state = initial_state;
    double current_time = t_start;
    double current_dt = dt;
    
    states.push_back(current_state);
    
    while (current_time < t_end) {
        double step_dt = std::min(current_dt, t_end - current_time);
        Control current_control = control_func(current_time);
        
        // Try the step
        State new_state = computeRK45Step(current_state, current_control, current_time, step_dt);
        
        // Check if we need to adjust step size
        if (last_step_size_ < step_dt) {
            // Step was reduced, try again with smaller step
            continue;
        }
        
        current_state = new_state;
        current_time += last_step_size_;
        current_dt = last_step_size_; // Use the actual step size that was used
        
        states.push_back(current_state);
    }
    
    return states;
}

State RK45Integrator::computeRK45Step(const State& state, const Control& control, double t, double dt) {
    // RK45 method with embedded error estimation
    State k1 = dynamics_->computeDerivative(state, control, t);
    function_evaluations_++;
    
    State k2 = dynamics_->computeDerivative(
        State(state.r_i + dt*a21*k1.r_i, state.v_i + dt*a21*k1.v_i,
              Quaterniond(state.q_bi.w() + dt*a21*k1.q_bi.w(), state.q_bi.x() + dt*a21*k1.q_bi.x(),
                         state.q_bi.y() + dt*a21*k1.q_bi.y(), state.q_bi.z() + dt*a21*k1.q_bi.z()),
              state.w_b + dt*a21*k1.w_b, state.m + dt*a21*k1.m),
        control, t + dt*a21);
    function_evaluations_++;
    
    State k3 = dynamics_->computeDerivative(
        State(state.r_i + dt*(a31*k1.r_i + a32*k2.r_i), state.v_i + dt*(a31*k1.v_i + a32*k2.v_i),
              Quaterniond(state.q_bi.w() + dt*(a31*k1.q_bi.w() + a32*k2.q_bi.w()),
                         state.q_bi.x() + dt*(a31*k1.q_bi.x() + a32*k2.q_bi.x()),
                         state.q_bi.y() + dt*(a31*k1.q_bi.y() + a32*k2.q_bi.y()),
                         state.q_bi.z() + dt*(a31*k1.q_bi.z() + a32*k2.q_bi.z())),
              state.w_b + dt*(a31*k1.w_b + a32*k2.w_b), state.m + dt*(a31*k1.m + a32*k2.m)),
        control, t + dt*(a31 + a32));
    function_evaluations_++;
    
    State k4 = dynamics_->computeDerivative(
        State(state.r_i + dt*(a41*k1.r_i + a42*k2.r_i + a43*k3.r_i),
              state.v_i + dt*(a41*k1.v_i + a42*k2.v_i + a43*k3.v_i),
              Quaterniond(state.q_bi.w() + dt*(a41*k1.q_bi.w() + a42*k2.q_bi.w() + a43*k3.q_bi.w()),
                         state.q_bi.x() + dt*(a41*k1.q_bi.x() + a42*k2.q_bi.x() + a43*k3.q_bi.x()),
                         state.q_bi.y() + dt*(a41*k1.q_bi.y() + a42*k2.q_bi.y() + a43*k3.q_bi.y()),
                         state.q_bi.z() + dt*(a41*k1.q_bi.z() + a42*k2.q_bi.z() + a43*k3.q_bi.z())),
              state.w_b + dt*(a41*k1.w_b + a42*k2.w_b + a43*k3.w_b),
              state.m + dt*(a41*k1.m + a42*k2.m + a43*k3.m)),
        control, t + dt*(a41 + a42 + a43));
    function_evaluations_++;
    
    State k5 = dynamics_->computeDerivative(
        State(state.r_i + dt*(a51*k1.r_i + a52*k2.r_i + a53*k3.r_i + a54*k4.r_i),
              state.v_i + dt*(a51*k1.v_i + a52*k2.v_i + a53*k3.v_i + a54*k4.v_i),
              Quaterniond(state.q_bi.w() + dt*(a51*k1.q_bi.w() + a52*k2.q_bi.w() + a53*k3.q_bi.w() + a54*k4.q_bi.w()),
                         state.q_bi.x() + dt*(a51*k1.q_bi.x() + a52*k2.q_bi.x() + a53*k3.q_bi.x() + a54*k4.q_bi.x()),
                         state.q_bi.y() + dt*(a51*k1.q_bi.y() + a52*k2.q_bi.y() + a53*k3.q_bi.y() + a54*k4.q_bi.y()),
                         state.q_bi.z() + dt*(a51*k1.q_bi.z() + a52*k2.q_bi.z() + a53*k3.q_bi.z() + a54*k4.q_bi.z())),
              state.w_b + dt*(a51*k1.w_b + a52*k2.w_b + a53*k3.w_b + a54*k4.w_b),
              state.m + dt*(a51*k1.m + a52*k2.m + a53*k3.m + a54*k4.m)),
        control, t + dt*(a51 + a52 + a53 + a54));
    function_evaluations_++;
    
    State k6 = dynamics_->computeDerivative(
        State(state.r_i + dt*(a61*k1.r_i + a62*k2.r_i + a63*k3.r_i + a64*k4.r_i + a65*k5.r_i),
              state.v_i + dt*(a61*k1.v_i + a62*k2.v_i + a63*k3.v_i + a64*k4.v_i + a65*k5.v_i),
              Quaterniond(state.q_bi.w() + dt*(a61*k1.q_bi.w() + a62*k2.q_bi.w() + a63*k3.q_bi.w() + a64*k4.q_bi.w() + a65*k5.q_bi.w()),
                         state.q_bi.x() + dt*(a61*k1.q_bi.x() + a62*k2.q_bi.x() + a63*k3.q_bi.x() + a64*k4.q_bi.x() + a65*k5.q_bi.x()),
                         state.q_bi.y() + dt*(a61*k1.q_bi.y() + a62*k2.q_bi.y() + a63*k3.q_bi.y() + a64*k4.q_bi.y() + a65*k5.q_bi.y()),
                         state.q_bi.z() + dt*(a61*k1.q_bi.z() + a62*k2.q_bi.z() + a63*k3.q_bi.z() + a64*k4.q_bi.z() + a65*k5.q_bi.z())),
              state.w_b + dt*(a61*k1.w_b + a62*k2.w_b + a63*k3.w_b + a64*k4.w_b + a65*k5.w_b),
              state.m + dt*(a61*k1.m + a62*k2.m + a63*k3.m + a64*k4.m + a65*k5.m)),
        control, t + dt*(a61 + a62 + a63 + a64 + a65));
    function_evaluations_++;
    
    // Compute 4th and 5th order solutions
    State state_4th;
    state_4th.r_i = state.r_i + dt * (c1*k1.r_i + c2*k2.r_i + c3*k3.r_i + c4*k4.r_i + c5*k5.r_i + c6*k6.r_i);
    state_4th.v_i = state.v_i + dt * (c1*k1.v_i + c2*k2.v_i + c3*k3.v_i + c4*k4.v_i + c5*k5.v_i + c6*k6.v_i);
    state_4th.q_bi = Quaterniond(
        state.q_bi.w() + dt * (c1*k1.q_bi.w() + c2*k2.q_bi.w() + c3*k3.q_bi.w() + c4*k4.q_bi.w() + c5*k5.q_bi.w() + c6*k6.q_bi.w()),
        state.q_bi.x() + dt * (c1*k1.q_bi.x() + c2*k2.q_bi.x() + c3*k3.q_bi.x() + c4*k4.q_bi.x() + c5*k5.q_bi.x() + c6*k6.q_bi.x()),
        state.q_bi.y() + dt * (c1*k1.q_bi.y() + c2*k2.q_bi.y() + c3*k3.q_bi.y() + c4*k4.q_bi.y() + c5*k5.q_bi.y() + c6*k6.q_bi.y()),
        state.q_bi.z() + dt * (c1*k1.q_bi.z() + c2*k2.q_bi.z() + c3*k3.q_bi.z() + c4*k4.q_bi.z() + c5*k5.q_bi.z() + c6*k6.q_bi.z())
    );
    state_4th.w_b = state.w_b + dt * (c1*k1.w_b + c2*k2.w_b + c3*k3.w_b + c4*k4.w_b + c5*k5.w_b + c6*k6.w_b);
    state_4th.m = state.m + dt * (c1*k1.m + c2*k2.m + c3*k3.m + c4*k4.m + c5*k5.m + c6*k6.m);
    
    State state_5th;
    state_5th.r_i = state.r_i + dt * (b1*k1.r_i + b2*k2.r_i + b3*k3.r_i + b4*k4.r_i + b5*k5.r_i + b6*k6.r_i);
    state_5th.v_i = state.v_i + dt * (b1*k1.v_i + b2*k2.v_i + b3*k3.v_i + b4*k4.v_i + b5*k5.v_i + b6*k6.v_i);
    state_5th.q_bi = Quaterniond(
        state.q_bi.w() + dt * (b1*k1.q_bi.w() + b2*k2.q_bi.w() + b3*k3.q_bi.w() + b4*k4.q_bi.w() + b5*k5.q_bi.w() + b6*k6.q_bi.w()),
        state.q_bi.x() + dt * (b1*k1.q_bi.x() + b2*k2.q_bi.x() + b3*k3.q_bi.x() + b4*k4.q_bi.x() + b5*k5.q_bi.x() + b6*k6.q_bi.x()),
        state.q_bi.y() + dt * (b1*k1.q_bi.y() + b2*k2.q_bi.y() + b3*k3.q_bi.y() + b4*k4.q_bi.y() + b5*k5.q_bi.y() + b6*k6.q_bi.y()),
        state.q_bi.z() + dt * (b1*k1.q_bi.z() + b2*k2.q_bi.z() + b3*k3.q_bi.z() + b4*k4.q_bi.z() + b5*k5.q_bi.z() + b6*k6.q_bi.z())
    );
    state_5th.w_b = state.w_b + dt * (b1*k1.w_b + b2*k2.w_b + b3*k3.w_b + b4*k4.w_b + b5*k5.w_b + b6*k6.w_b);
    state_5th.m = state.m + dt * (b1*k1.m + b2*k2.m + b3*k3.m + b4*k4.m + b5*k5.m + b6*k6.m);
    
    // Normalize quaternions
    utils::normalizeQuaternion(state_4th);
    utils::normalizeQuaternion(state_5th);
    
    // Compute error estimate
    double error = computeError(state, state_4th, state_5th);
    
    // Adjust step size based on error
    double optimal_dt = computeOptimalStepSize(dt, error, tolerance_);
    last_step_size_ = std::clamp(optimal_dt, min_step_, max_step_);
    
    // If error is too large, reduce step size and try again
    if (error > tolerance_ && last_step_size_ < dt) {
        return computeRK45Step(state, control, t, last_step_size_);
    }
    
    return state_5th; // Use 5th order solution
}

double RK45Integrator::computeError(const State& state, const State& state_4th, const State& state_5th) {
    // Compute relative error between 4th and 5th order solutions
    double error_r = (state_5th.r_i - state_4th.r_i).norm() / (state.r_i.norm() + 1e-10);
    double error_v = (state_5th.v_i - state_4th.v_i).norm() / (state.v_i.norm() + 1e-10);
    double error_w = (state_5th.w_b - state_4th.w_b).norm() / (state.w_b.norm() + 1e-10);
    double error_m = std::abs(state_5th.m - state_4th.m) / (std::abs(state.m) + 1e-10);
    
    return std::max({error_r, error_v, error_w, error_m});
}

double RK45Integrator::computeOptimalStepSize(double current_dt, double error, double tolerance) {
    if (error < 1e-10) {
        return current_dt * 1.5; // Increase step size
    }
    
    double safety_factor = 0.9;
    double scale_factor = safety_factor * std::pow(tolerance / error, 0.2);
    return current_dt * scale_factor;
}

// Factory functions
std::shared_ptr<RK4Integrator> createRK4Integrator(std::shared_ptr<Dynamics> dynamics) {
    return std::make_shared<RK4Integrator>(dynamics);
}

std::shared_ptr<RK45Integrator> createRK45Integrator(std::shared_ptr<Dynamics> dynamics,
                                                    double tolerance, double min_step, double max_step) {
    return std::make_shared<RK45Integrator>(dynamics, tolerance, min_step, max_step);
}

} // namespace rocket_physics
