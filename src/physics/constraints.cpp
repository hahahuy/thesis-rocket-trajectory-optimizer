#include "constraints.hpp"
#include <algorithm>
#include <cmath>
#include <memory>

namespace rocket_physics {

// ConstraintChecker implementation
ConstraintChecker::ConstraintChecker(const Limits& limits) : limits_(limits) {
}

std::vector<ConstraintViolation> ConstraintChecker::checkConstraints(const State& state, const Control& control, double t) const {
    std::vector<ConstraintViolation> violations;
    
    // Check all constraints
    violations.push_back(checkDynamicPressure(state));
    violations.push_back(checkLoadFactor(state, control, t));
    violations.push_back(checkAngleOfAttack(state));
    violations.push_back(checkMassUnderflow(state));
    violations.push_back(checkThrustLimit(control));
    violations.push_back(checkAltitude(state));
    violations.push_back(checkVelocity(state));
    
    return violations;
}

ConstraintViolation ConstraintChecker::checkDynamicPressure(const State& state) const {
    double q = computeDynamicPressure(state);
    double violation_magnitude = std::max(0.0, q - limits_.q_max);
    bool is_violated = q > limits_.q_max;
    
    return ConstraintViolation(ConstraintType::DYNAMIC_PRESSURE, q, limits_.q_max, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkLoadFactor(const State& state, const Control& control, double t) const {
    double n = computeLoadFactor(state, control, t);
    double violation_magnitude = std::max(0.0, n - limits_.n_max);
    bool is_violated = n > limits_.n_max;
    
    return ConstraintViolation(ConstraintType::LOAD_FACTOR, n, limits_.n_max, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkAngleOfAttack(const State& state) const {
    double alpha = computeAngleOfAttack(state);
    double violation_magnitude = std::max(0.0, alpha - limits_.alpha_max);
    bool is_violated = alpha > limits_.alpha_max;
    
    return ConstraintViolation(ConstraintType::ANGLE_OF_ATTACK, alpha, limits_.alpha_max, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkMassUnderflow(const State& state) const {
    double violation_magnitude = std::max(0.0, limits_.m_dry - state.m);
    bool is_violated = state.m < limits_.m_dry;
    
    return ConstraintViolation(ConstraintType::MASS_UNDERFLOW, state.m, limits_.m_dry, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkThrustLimit(const Control& control) const {
    double violation_magnitude = std::max(0.0, control.T - limits_.T_max);
    bool is_violated = control.T > limits_.T_max;
    
    return ConstraintViolation(ConstraintType::THRUST_LIMIT, control.T, limits_.T_max, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkGimbalRate(const Control& control, const Control& previous_control, double dt) const {
    double gimbal_rate = computeGimbalRate(control, previous_control, dt);
    double violation_magnitude = std::max(0.0, gimbal_rate - limits_.w_gimbal_max);
    bool is_violated = gimbal_rate > limits_.w_gimbal_max;
    
    return ConstraintViolation(ConstraintType::GIMBAL_RATE, gimbal_rate, limits_.w_gimbal_max, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkAltitude(const State& state) const {
    double altitude = computeAltitude(state);
    double violation_magnitude = std::max(0.0, 0.0 - altitude); // Assuming minimum altitude is 0
    bool is_violated = altitude < 0.0;
    
    return ConstraintViolation(ConstraintType::ALTITUDE_MIN, altitude, 0.0, violation_magnitude, is_violated);
}

ConstraintViolation ConstraintChecker::checkVelocity(const State& state) const {
    double velocity = computeVelocity(state);
    double violation_magnitude = std::max(0.0, velocity - 1000.0); // Assuming maximum velocity is 1000 m/s
    bool is_violated = velocity > 1000.0;
    
    return ConstraintViolation(ConstraintType::VELOCITY_MAX, velocity, 1000.0, violation_magnitude, is_violated);
}

bool ConstraintChecker::hasViolations(const State& state, const Control& control, double t) const {
    auto violations = checkConstraints(state, control, t);
    return std::any_of(violations.begin(), violations.end(), 
                       [](const ConstraintViolation& v) { return v.is_violated; });
}

int ConstraintChecker::getViolationCount(const State& state, const Control& control, double t) const {
    auto violations = checkConstraints(state, control, t);
    return std::count_if(violations.begin(), violations.end(),
                        [](const ConstraintViolation& v) { return v.is_violated; });
}

double ConstraintChecker::getMaxViolationMagnitude(const State& state, const Control& control, double t) const {
    auto violations = checkConstraints(state, control, t);
    double max_violation = 0.0;
    
    for (const auto& violation : violations) {
        if (violation.is_violated) {
            max_violation = std::max(max_violation, violation.violation_magnitude);
        }
    }
    
    return max_violation;
}

// Helper methods
double ConstraintChecker::computeDynamicPressure(const State& state) const {
    double velocity = state.v_i.norm();
    double altitude = state.r_i.norm() - 6371000.0; // Earth radius
    double density = 1.225 * std::exp(-altitude / 8400.0); // Simplified atmosphere
    return 0.5 * density * velocity * velocity;
}

double ConstraintChecker::computeLoadFactor(const State& state, const Control& control, double t) const {
    // Compute normal load factor: n = L / (m * g0)
    // Where L is the aerodynamic lift force
    
    // Get relative velocity (simplified - would need access to dynamics for wind)
    double speed = state.v_i.norm();
    if (speed < 1e-6) {
        return 1.0; // No aerodynamic load at zero velocity
    }
    
    // Compute altitude and atmospheric density
    double altitude = std::max(0.0, state.r_i.norm() - 6371000.0); // Earth radius
    double density = 1.225 * std::exp(-altitude / 8400.0); // Simplified atmosphere
    
    // Dynamic pressure
    double q = 0.5 * density * speed * speed;
    
    // Simplified angle of attack (would need body frame velocity)
    double alpha = 0.05; // Simplified - assume small angle
    
    // Lift force: L = q * S_ref * CL_alpha * alpha
    // Using default CL_alpha = 3.5, S_ref = 0.05
    double L = q * 0.05 * 3.5 * alpha;
    
    // Normal load factor in g's
    double n = std::abs(L / (state.m * 9.81));
    
    // Add baseline 1g for gravity
    return n + 1.0;
}

double ConstraintChecker::computeAngleOfAttack(const State& state) const {
    if (state.v_i.norm() < 1e-6) {
        return 0.0;
    }
    
    // Transform velocity to body frame (simplified)
    Vec3 v_body = state.v_i; // Assume no rotation for simplicity
    double vx = v_body.x();
    double vy = v_body.y();
    double vz = v_body.z();
    
    return std::atan2(std::sqrt(vy*vy + vz*vz), vx);
}

double ConstraintChecker::computeAltitude(const State& state) const {
    return state.r_i.norm() - 6371000.0; // Earth radius
}

double ConstraintChecker::computeVelocity(const State& state) const {
    return state.v_i.norm();
}

double ConstraintChecker::computeGimbalRate(const Control& control, const Control& previous_control, double dt) const {
    if (dt < 1e-10) {
        return 0.0;
    }
    
    Vec3 delta_direction = control.uT_b - previous_control.uT_b;
    return delta_direction.norm() / dt;
}

// ConstraintPenalty implementation
ConstraintPenalty::ConstraintPenalty(const std::vector<double>& penalty_weights) : penalty_weights_(penalty_weights) {
    // Initialize with default weights if not provided
    if (penalty_weights_.empty()) {
        penalty_weights_.resize(8, 1.0); // 8 constraint types
    }
}

double ConstraintPenalty::computePenalty(const std::vector<ConstraintViolation>& violations) const {
    double total_penalty = 0.0;
    
    for (const auto& violation : violations) {
        if (violation.is_violated) {
            total_penalty += computePenalty(violation);
        }
    }
    
    return total_penalty;
}

double ConstraintPenalty::computePenalty(const ConstraintViolation& violation) const {
    int index = getConstraintIndex(violation.type);
    if (index >= 0 && index < static_cast<int>(penalty_weights_.size())) {
        return penalty_weights_[index] * violation.violation_magnitude;
    }
    return 0.0;
}

void ConstraintPenalty::setPenaltyWeight(ConstraintType type, double weight) {
    int index = getConstraintIndex(type);
    if (index >= 0 && index < static_cast<int>(penalty_weights_.size())) {
        penalty_weights_[index] = weight;
    }
}

double ConstraintPenalty::getPenaltyWeight(ConstraintType type) const {
    int index = getConstraintIndex(type);
    if (index >= 0 && index < static_cast<int>(penalty_weights_.size())) {
        return penalty_weights_[index];
    }
    return 0.0;
}

int ConstraintPenalty::getConstraintIndex(ConstraintType type) const {
    switch (type) {
        case ConstraintType::DYNAMIC_PRESSURE: return 0;
        case ConstraintType::LOAD_FACTOR: return 1;
        case ConstraintType::ANGLE_OF_ATTACK: return 2;
        case ConstraintType::MASS_UNDERFLOW: return 3;
        case ConstraintType::THRUST_LIMIT: return 4;
        case ConstraintType::GIMBAL_RATE: return 5;
        case ConstraintType::ALTITUDE_MIN: return 6;
        case ConstraintType::VELOCITY_MAX: return 7;
        default: return -1;
    }
}

// ConstraintHandler implementation
ConstraintHandler::ConstraintHandler(std::shared_ptr<ConstraintChecker> checker,
                                   std::shared_ptr<ConstraintPenalty> penalty)
    : checker_(checker), penalty_(penalty) {
}

Control ConstraintHandler::handleViolations(const State& state, const Control& control, double t) const {
    auto violations = checker_->checkConstraints(state, control, t);
    
    if (!canHandleViolations(violations)) {
        return control; // Return original control if violations cannot be handled
    }
    
    Control modified_control = control;
    
    // Handle different types of violations
    for (const auto& violation : violations) {
        if (!violation.is_violated) continue;
        
        switch (violation.type) {
            case ConstraintType::THRUST_LIMIT:
                modified_control = handleThrustViolation(state, modified_control);
                break;
            case ConstraintType::GIMBAL_RATE:
                // Would need previous control for this
                break;
            case ConstraintType::DYNAMIC_PRESSURE:
            case ConstraintType::LOAD_FACTOR:
            case ConstraintType::ANGLE_OF_ATTACK:
                modified_control = handleAerodynamicViolation(state, modified_control);
                break;
            default:
                break;
        }
    }
    
    return modified_control;
}

bool ConstraintHandler::canHandleViolations(const std::vector<ConstraintViolation>& violations) const {
    // Check if violations can be handled by modifying control
    for (const auto& violation : violations) {
        if (violation.is_violated) {
            switch (violation.type) {
                case ConstraintType::MASS_UNDERFLOW:
                case ConstraintType::ALTITUDE_MIN:
                    return false; // Cannot be handled by control modification
                default:
                    break;
            }
        }
    }
    return true;
}

Control ConstraintHandler::handleThrustViolation(const State& state, const Control& control) const {
    Control modified_control = control;
    
    if (control.T > checker_->getLimits().T_max) {
        modified_control.T = checker_->getLimits().T_max;
    }
    
    return modified_control;
}

Control ConstraintHandler::handleGimbalViolation(const State& state, const Control& control, const Control& previous_control) const {
    // Simplified gimbal rate limiting
    Control modified_control = control;
    
    // Would implement gimbal rate limiting here
    // This is a placeholder implementation
    
    return modified_control;
}

Control ConstraintHandler::handleAerodynamicViolation(const State& state, const Control& control) const {
    Control modified_control = control;
    
    // Simplified aerodynamic constraint handling
    // Would implement angle of attack limiting, etc.
    
    return modified_control;
}

// Factory functions
std::shared_ptr<ConstraintChecker> createConstraintChecker(const Limits& limits) {
    return std::make_shared<ConstraintChecker>(limits);
}

std::shared_ptr<ConstraintPenalty> createConstraintPenalty(const std::vector<double>& penalty_weights) {
    return std::make_shared<ConstraintPenalty>(penalty_weights);
}

std::shared_ptr<ConstraintHandler> createConstraintHandler(std::shared_ptr<ConstraintChecker> checker,
                                                         std::shared_ptr<ConstraintPenalty> penalty) {
    return std::make_shared<ConstraintHandler>(checker, penalty);
}

/**
 * @brief Enforce operational limits on state and control
 * @param state State to enforce limits on
 * @param control Control to enforce limits on
 * @param limits Operational limits
 * @param diag Optional diagnostic pointer for logging violations
 */
void enforceLimits(State& state, Control& control, const Limits& limits, Diag* diag) {
    // Clamp thrust to limits
    control.T = std::clamp(control.T, 0.0, limits.T_max);
    
    // Ensure mass doesn't go below dry mass
    state.m = std::max(state.m, limits.m_dry);
    
    // Clamp control surface deflection if delta_limit exists
    if (limits.w_gimbal_max > 0) { // Using gimbal rate limit as proxy for delta limit check
        // Would need delta_limit in Limits struct - simplified for now
        // control.delta = std::clamp(control.delta, -limits.delta_limit, limits.delta_limit);
    }
    
    // Log violations if diagnostic is provided
    if (diag) {
        if (diag->q_violation) {
            // Would use logging system here - for now just set flag
            // spdlog::warn("q-limit exceeded: {:.1f} Pa", diag->q);
        }
        if (diag->n_violation) {
            // spdlog::warn("n-limit exceeded: {:.2f} g", diag->n);
        }
    }
}

/**
 * @brief Check q-limit with diagnostic update
 * @param q Dynamic pressure [Pa]
 * @param limits Operational limits
 * @param diag Optional diagnostic pointer
 * @return True if limit is exceeded
 */
bool checkQLimit(double q, const Limits& limits, Diag* diag) {
    bool exceed = (q > limits.q_max);
    if (diag) {
        diag->q_violation = exceed;
        diag->q = q;
    }
    return exceed;
}

/**
 * @brief Check n-limit with diagnostic update
 * @param n Load factor [g]
 * @param limits Operational limits
 * @param diag Optional diagnostic pointer
 * @return True if limit is exceeded
 */
bool checkNLimit(double n, const Limits& limits, Diag* diag) {
    bool exceed = (n > limits.n_max);
    if (diag) {
        diag->n_violation = exceed;
        diag->n = n;
    }
    return exceed;
}

} // namespace rocket_physics
