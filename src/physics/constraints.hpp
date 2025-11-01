#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <functional>
#include <vector>
#include <memory>

namespace rocket_physics {

/**
 * @brief Constraint violation type
 */
enum class ConstraintType {
    DYNAMIC_PRESSURE,    // q > q_max
    LOAD_FACTOR,         // n > n_max
    ANGLE_OF_ATTACK,     // α > α_max
    MASS_UNDERFLOW,      // m < m_dry
    THRUST_LIMIT,        // T > T_max
    GIMBAL_RATE,         // ω_gimbal > ω_max
    ALTITUDE_MIN,       // h < h_min
    VELOCITY_MAX         // v > v_max
};

/**
 * @brief Constraint violation information
 */
struct ConstraintViolation {
    ConstraintType type;
    double value;
    double limit;
    double violation_magnitude;
    bool is_violated;
    
    ConstraintViolation(ConstraintType t, double v, double l, double vm, bool iv)
        : type(t), value(v), limit(l), violation_magnitude(vm), is_violated(iv) {}
};

/**
 * @brief Constraint checker class
 */
class ConstraintChecker {
public:
    /**
     * @brief Constructor
     * @param limits Operational limits
     */
    explicit ConstraintChecker(const Limits& limits);
    
    /**
     * @brief Check all constraints
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Vector of constraint violations
     */
    std::vector<ConstraintViolation> checkConstraints(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Check dynamic pressure constraint
     * @param state Current state
     * @return Constraint violation information
     */
    ConstraintViolation checkDynamicPressure(const State& state) const;
    
    /**
     * @brief Check load factor constraint
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Constraint violation information
     */
    ConstraintViolation checkLoadFactor(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Check angle of attack constraint
     * @param state Current state
     * @return Constraint violation information
     */
    ConstraintViolation checkAngleOfAttack(const State& state) const;
    
    /**
     * @brief Check mass underflow constraint
     * @param state Current state
     * @return Constraint violation information
     */
    ConstraintViolation checkMassUnderflow(const State& state) const;
    
    /**
     * @brief Check thrust limit constraint
     * @param control Current control
     * @return Constraint violation information
     */
    ConstraintViolation checkThrustLimit(const Control& control) const;
    
    /**
     * @brief Check gimbal rate constraint
     * @param control Current control
     * @param previous_control Previous control
     * @param dt Time step
     * @return Constraint violation information
     */
    ConstraintViolation checkGimbalRate(const Control& control, const Control& previous_control, double dt) const;
    
    /**
     * @brief Check altitude constraint
     * @param state Current state
     * @return Constraint violation information
     */
    ConstraintViolation checkAltitude(const State& state) const;
    
    /**
     * @brief Check velocity constraint
     * @param state Current state
     * @return Constraint violation information
     */
    ConstraintViolation checkVelocity(const State& state) const;
    
    /**
     * @brief Check if any constraints are violated
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return True if any constraints are violated
     */
    bool hasViolations(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Get violation count
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Number of violated constraints
     */
    int getViolationCount(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Get maximum violation magnitude
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Maximum violation magnitude
     */
    double getMaxViolationMagnitude(const State& state, const Control& control, double t) const;

private:
    Limits limits_;
    
    // Helper methods
    double computeDynamicPressure(const State& state) const;
    double computeLoadFactor(const State& state, const Control& control, double t) const;
    double computeAngleOfAttack(const State& state) const;
    double computeAltitude(const State& state) const;
    double computeVelocity(const State& state) const;
    double computeGimbalRate(const Control& control, const Control& previous_control, double dt) const;
};

/**
 * @brief Constraint penalty function
 */
class ConstraintPenalty {
public:
    /**
     * @brief Constructor
     * @param penalty_weights Penalty weights for each constraint type
     */
    explicit ConstraintPenalty(const std::vector<double>& penalty_weights);
    
    /**
     * @brief Compute penalty value
     * @param violations Constraint violations
     * @return Total penalty value
     */
    double computePenalty(const std::vector<ConstraintViolation>& violations) const;
    
    /**
     * @brief Compute penalty for single violation
     * @param violation Constraint violation
     * @return Penalty value
     */
    double computePenalty(const ConstraintViolation& violation) const;
    
    /**
     * @brief Set penalty weight for constraint type
     * @param type Constraint type
     * @param weight Penalty weight
     */
    void setPenaltyWeight(ConstraintType type, double weight);
    
    /**
     * @brief Get penalty weight for constraint type
     * @param type Constraint type
     * @return Penalty weight
     */
    double getPenaltyWeight(ConstraintType type) const;

private:
    std::vector<double> penalty_weights_;
    
    int getConstraintIndex(ConstraintType type) const;
};

/**
 * @brief Constraint violation handler
 */
class ConstraintHandler {
public:
    /**
     * @brief Constructor
     * @param checker Constraint checker
     * @param penalty Constraint penalty function
     */
    ConstraintHandler(std::shared_ptr<ConstraintChecker> checker,
                     std::shared_ptr<ConstraintPenalty> penalty);
    
    /**
     * @brief Handle constraint violations
     * @param state Current state
     * @param control Current control
     * @param t Current time
     * @return Updated control (if violations can be handled)
     */
    Control handleViolations(const State& state, const Control& control, double t) const;
    
    /**
     * @brief Check if violations can be handled
     * @param violations Constraint violations
     * @return True if violations can be handled
     */
    bool canHandleViolations(const std::vector<ConstraintViolation>& violations) const;
    
    /**
     * @brief Get constraint checker
     * @return Shared pointer to constraint checker
     */
    std::shared_ptr<ConstraintChecker> getChecker() const { return checker_; }
    
    /**
     * @brief Get constraint penalty
     * @return Shared pointer to constraint penalty
     */
    std::shared_ptr<ConstraintPenalty> getPenalty() const { return penalty_; }

private:
    std::shared_ptr<ConstraintChecker> checker_;
    std::shared_ptr<ConstraintPenalty> penalty_;
    
    // Violation handling strategies
    Control handleThrustViolation(const State& state, const Control& control) const;
    Control handleGimbalViolation(const State& state, const Control& control, const Control& previous_control) const;
    Control handleAerodynamicViolation(const State& state, const Control& control) const;
};

/**
 * @brief Factory functions
 */

/**
 * @brief Create constraint checker
 * @param limits Operational limits
 * @return Shared pointer to constraint checker
 */
std::shared_ptr<ConstraintChecker> createConstraintChecker(const Limits& limits);

/**
 * @brief Create constraint penalty function
 * @param penalty_weights Penalty weights
 * @return Shared pointer to constraint penalty
 */
std::shared_ptr<ConstraintPenalty> createConstraintPenalty(const std::vector<double>& penalty_weights);

/**
 * @brief Create constraint handler
 * @param checker Constraint checker
 * @param penalty Constraint penalty
 * @return Shared pointer to constraint handler
 */
std::shared_ptr<ConstraintHandler> createConstraintHandler(std::shared_ptr<ConstraintChecker> checker,
                                                          std::shared_ptr<ConstraintPenalty> penalty);

/**
 * @brief Enforce operational limits on state and control
 * @param state State to enforce limits on
 * @param control Control to enforce limits on
 * @param limits Operational limits
 * @param diag Optional diagnostic pointer for logging violations
 */
void enforceLimits(State& state, Control& control, const Limits& limits, Diag* diag = nullptr);

/**
 * @brief Check q-limit with diagnostic update
 * @param q Dynamic pressure [Pa]
 * @param limits Operational limits
 * @param diag Optional diagnostic pointer
 * @return True if limit is exceeded
 */
bool checkQLimit(double q, const Limits& limits, Diag* diag = nullptr);

/**
 * @brief Check n-limit with diagnostic update
 * @param n Load factor [g]
 * @param limits Operational limits
 * @param diag Optional diagnostic pointer
 * @return True if limit is exceeded
 */
bool checkNLimit(double n, const Limits& limits, Diag* diag = nullptr);

} // namespace rocket_physics
