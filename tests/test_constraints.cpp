#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../src/physics/types.hpp"
#include "../src/physics/dynamics.hpp"
#include "../src/physics/constraints.hpp"
#include <cmath>

using namespace rocket_physics;

class ConstraintsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        phys_.Cd = 0.3;
        phys_.Cl = 0.0;
        phys_.S_ref = 0.05;
        phys_.I_b = Matrix3d::Identity() * 1000.0;
        phys_.r_cg = Vec3::Zero();
        phys_.Isp = 300.0;
        phys_.g0 = 9.81;
        phys_.rho0 = 1.225;
        phys_.h_scale = 8400.0;
        phys_.CL_alpha = 3.5;
        phys_.Cm_alpha = -0.8;
        phys_.C_delta = 0.05;
        phys_.l_ref = 1.2;
        phys_.delta_limit = 0.1745; // 10 degrees
        
        limits_.T_max = 1000000.0;
        limits_.m_dry = 1000.0;
        limits_.q_max = 40000.0;
        limits_.w_gimbal_max = 1.0;
        limits_.alpha_max = 0.1;
        limits_.n_max = 5.0;
        
        // Create test state - constant velocity flight
        test_state_.r_i = Vec3(0.0, 0.0, 1000.0); // 1 km altitude
        test_state_.v_i = Vec3(100.0, 0.0, 0.0); // 100 m/s horizontal
        test_state_.q_bi = Quaterniond::Identity();
        test_state_.w_b = Vec3::Zero();
        test_state_.m = 5000.0;
        
        // Create test control
        test_control_.T = 100000.0;
        test_control_.uT_b = Vec3(0.0, 0.0, 1.0);
        test_control_.delta = 0.0;
        
        // Create dynamics and constraint checker
        dynamics_ = createDynamics(phys_, limits_);
        checker_ = createConstraintChecker(limits_);
    }
    
    Phys phys_;
    Limits limits_;
    State test_state_;
    Control test_control_;
    std::shared_ptr<Dynamics> dynamics_;
    std::shared_ptr<ConstraintChecker> checker_;
};

// Test constant-velocity flight: compare analytic q and n
TEST_F(ConstraintsTest, ConstantVelocityFlight) {
    // For constant velocity at given altitude, we can compute q analytically
    double altitude = 1000.0;
    double density = 1.225 * std::exp(-altitude / 8400.0);
    double velocity = 100.0;
    double q_expected = 0.5 * density * velocity * velocity;
    
    // Compute q from dynamics
    double q_computed = dynamics_->computeDynamicPressure(test_state_);
    
    // Should match within 5%
    EXPECT_NEAR(q_computed, q_expected, q_expected * 0.05);
}

// Test over-thrust case: q-limit should trigger
TEST_F(ConstraintsTest, OverThrustQLimit) {
    // Create high-velocity state
    State high_velocity_state = test_state_;
    high_velocity_state.v_i = Vec3(500.0, 0.0, 0.0); // 500 m/s
    
    // Compute dynamic pressure
    double q = dynamics_->computeDynamicPressure(high_velocity_state);
    
    // Should exceed limit
    EXPECT_GT(q, limits_.q_max);
    
    // Check constraint
    auto violations = checker_->checkConstraints(high_velocity_state, test_control_, 0.0);
    
    bool q_violated = false;
    for (const auto& v : violations) {
        if (v.type == ConstraintType::DYNAMIC_PRESSURE && v.is_violated) {
            q_violated = true;
            EXPECT_GT(v.value, v.limit);
        }
    }
    EXPECT_TRUE(q_violated);
}

// Test lift-slope case: increasing α increases n linearly
TEST_F(ConstraintsTest, LiftSlope) {
    // Test with different angles of attack
    std::vector<double> alphas = {0.0, 0.05, 0.1, 0.15};
    std::vector<double> n_values;
    
    for (double alpha : alphas) {
        // Create state with angle of attack
        State alpha_state = test_state_;
        // Rotate velocity vector to create angle of attack
        Quaterniond alpha_rot = Quaterniond(Eigen::AngleAxisd(alpha, Vec3(0, 1, 0)));
        alpha_state.q_bi = alpha_rot.inverse();
        alpha_state.v_i = alpha_rot * Vec3(100.0, 0.0, 0.0);
        
        double n = dynamics_->computeLoadFactor(alpha_state, test_control_, 0.0);
        n_values.push_back(n);
    }
    
    // Check that n increases with alpha (linearly for small angles)
    for (size_t i = 1; i < n_values.size(); ++i) {
        EXPECT_GT(n_values[i], n_values[i-1]);
    }
    
    // For small angles, n should be approximately linear
    double n_slope = (n_values[1] - n_values[0]) / (alphas[1] - alphas[0]);
    double expected_slope = phys_.CL_alpha * 0.5 * phys_.rho0 * test_state_.v_i.norm() * test_state_.v_i.norm() * phys_.S_ref / (test_state_.m * phys_.g0);
    EXPECT_NEAR(n_slope, expected_slope, expected_slope * 0.1);
}

// Test all diagnostics flags behave as expected
TEST_F(ConstraintsTest, DiagnosticsFlags) {
    // Create state that violates constraints
    State violating_state = test_state_;
    violating_state.v_i = Vec3(400.0, 0.0, 0.0); // High velocity
    violating_state.m = 500.0; // Below dry mass
    
    Diag diag = dynamics_->checkConstraints(violating_state, test_control_, 0.0);
    
    // Should detect violations
    EXPECT_GT(diag.q, 0.0);
    EXPECT_GT(diag.n, 0.0);
    
    // Check if violations are flagged
    EXPECT_TRUE(diag.q_violation || diag.q <= limits_.q_max); // Either violated or within limit
    EXPECT_TRUE(diag.m_underflow || violating_state.m >= limits_.m_dry); // Either violated or OK
}

// Test constraint penalty computation
TEST_F(ConstraintsTest, ConstraintPenalty) {
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto penalty = createConstraintPenalty(weights);
    
    // Create violations
    State high_q_state = test_state_;
    high_q_state.v_i = Vec3(500.0, 0.0, 0.0);
    
    auto violations = checker_->checkConstraints(high_q_state, test_control_, 0.0);
    double penalty_value = penalty->computePenalty(violations);
    
    EXPECT_GT(penalty_value, 0.0);
}

// Test constraint handler
TEST_F(ConstraintsTest, ConstraintHandler) {
    auto penalty = createConstraintPenalty({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    auto handler = createConstraintHandler(checker_, penalty);
    
    // Test with excessive thrust
    Control excessive_control(2000000.0, Vec3(0.0, 0.0, 1.0));
    Control handled_control = handler->handleViolations(test_state_, excessive_control, 0.0);
    
    // Thrust should be clamped
    EXPECT_LE(handled_control.T, limits_.T_max);
    EXPECT_EQ(handled_control.T, limits_.T_max);
}

// Test q-limit check function
TEST_F(ConstraintsTest, QLimitCheck) {
    Diag diag;
    
    // Test below limit
    bool violated1 = checkQLimit(30000.0, limits_, &diag);
    EXPECT_FALSE(violated1);
    EXPECT_FALSE(diag.q_violation);
    EXPECT_EQ(diag.q, 30000.0);
    
    // Test above limit
    bool violated2 = checkQLimit(50000.0, limits_, &diag);
    EXPECT_TRUE(violated2);
    EXPECT_TRUE(diag.q_violation);
    EXPECT_EQ(diag.q, 50000.0);
}

// Test n-limit check function
TEST_F(ConstraintsTest, NLimitCheck) {
    Diag diag;
    
    // Test below limit
    bool violated1 = checkNLimit(3.0, limits_, &diag);
    EXPECT_FALSE(violated1);
    EXPECT_FALSE(diag.n_violation);
    EXPECT_EQ(diag.n, 3.0);
    
    // Test above limit
    bool violated2 = checkNLimit(6.0, limits_, &diag);
    EXPECT_TRUE(violated2);
    EXPECT_TRUE(diag.n_violation);
    EXPECT_EQ(diag.n, 6.0);
}

// Test enforce limits function
TEST_F(ConstraintsTest, EnforceLimits) {
    State test_state = test_state_;
    Control test_control = test_control_;
    Diag diag;
    
    // Set values that violate limits
    test_control.T = 2000000.0; // Exceeds T_max
    test_state.m = 500.0; // Below m_dry
    diag.q_violation = true;
    diag.n_violation = true;
    
    // Enforce limits
    enforceLimits(test_state, test_control, limits_, &diag);
    
    // Check that limits are enforced
    EXPECT_LE(test_control.T, limits_.T_max);
    EXPECT_GE(test_state.m, limits_.m_dry);
}

// Test control surface deflection
TEST_F(ConstraintsTest, ControlSurfaceDeflection) {
    // Test with control surface deflection
    Control control_with_delta(100000.0, Vec3(0.0, 0.0, 1.0), 0.1); // 0.1 rad deflection
    
    EXPECT_EQ(control_with_delta.delta, 0.1);
    
    // Compute forces and moments with deflection
    auto [forces, moments] = dynamics_->computeForcesAndMoments(test_state_, control_with_delta, 0.0);
    
    // Moments should be non-zero due to control surface
    EXPECT_NE(moments.y(), 0.0);
}

// Test violation count
TEST_F(ConstraintsTest, ViolationCount) {
    State violating_state = test_state_;
    violating_state.v_i = Vec3(500.0, 0.0, 0.0); // High velocity -> high q
    violating_state.m = 500.0; // Below dry mass
    
    int count = checker_->getViolationCount(violating_state, test_control_, 0.0);
    EXPECT_GT(count, 0);
    
    bool has_violations = checker_->hasViolations(violating_state, test_control_, 0.0);
    EXPECT_TRUE(has_violations);
}

// Test max violation magnitude
TEST_F(ConstraintsTest, MaxViolationMagnitude) {
    State violating_state = test_state_;
    violating_state.v_i = Vec3(500.0, 0.0, 0.0);
    
    double max_violation = checker_->getMaxViolationMagnitude(violating_state, test_control_, 0.0);
    EXPECT_GT(max_violation, 0.0);
}
