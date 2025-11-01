#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../src/physics/types.hpp"
#include "../src/physics/dynamics.hpp"
#include "../src/physics/integrator.hpp"
#include "../src/physics/atmosphere.hpp"
#include "../src/physics/constraints.hpp"

using namespace rocket_physics;

class DynamicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        phys_.Cd = 0.3;
        phys_.Cl = 0.0;
        phys_.S_ref = 1.0;
        phys_.I_b = Matrix3d::Identity() * 1000.0;
        phys_.r_cg = Vec3::Zero();
        phys_.Isp = 300.0;
        phys_.g0 = 9.81;
        phys_.rho0 = 1.225;
        phys_.h_scale = 8400.0;
        
        limits_.T_max = 1000000.0;
        limits_.m_dry = 1000.0;
        limits_.q_max = 50000.0;
        limits_.w_gimbal_max = 1.0;
        limits_.alpha_max = 0.1;
        limits_.n_max = 10.0;
        
        // Create test state
        test_state_.r_i = Vec3(1000.0, 2000.0, 3000.0);
        test_state_.v_i = Vec3(100.0, 200.0, 300.0);
        test_state_.q_bi = Quaterniond::Identity();
        test_state_.w_b = Vec3::Zero();
        test_state_.m = 5000.0;
        
        // Create test control
        test_control_.T = 100000.0;
        test_control_.uT_b = Vec3(0.0, 0.0, 1.0);
        
        // Create dynamics object
        dynamics_ = createDynamics(phys_, limits_);
    }
    
    Phys phys_;
    Limits limits_;
    State test_state_;
    Control test_control_;
    std::shared_ptr<Dynamics> dynamics_;
};

// Test Dynamics class
TEST_F(DynamicsTest, DynamicsConstructor) {
    EXPECT_NE(dynamics_, nullptr);
    EXPECT_EQ(dynamics_->getPhys().Cd, phys_.Cd);
    EXPECT_EQ(dynamics_->getLimits().T_max, limits_.T_max);
}

TEST_F(DynamicsTest, ComputeDerivative) {
    State state_dot = dynamics_->computeDerivative(test_state_, test_control_, 0.0);
    
    // Check that derivative is computed
    EXPECT_NE(state_dot.r_i, Vec3::Zero()); // Position derivative should be velocity
    EXPECT_NE(state_dot.v_i, Vec3::Zero()); // Velocity derivative should be non-zero
    EXPECT_NE(state_dot.m, 0.0); // Mass derivative should be negative (fuel consumption)
    
    // Position derivative should equal velocity
    EXPECT_EQ(state_dot.r_i, test_state_.v_i);
}

TEST_F(DynamicsTest, ComputeForcesAndMoments) {
    auto [forces, moments] = dynamics_->computeForcesAndMoments(test_state_, test_control_, 0.0);
    
    // Forces should be non-zero
    EXPECT_NE(forces, Vec3::Zero());
    
    // Thrust force should be in thrust direction
    Vec3 expected_thrust = test_control_.T * test_control_.uT_b;
    EXPECT_NEAR(forces.norm(), expected_thrust.norm(), 1e-6);
}

TEST_F(DynamicsTest, ComputeAtmosphericProperties) {
    double altitude = 1000.0;
    auto [density, pressure] = dynamics_->computeAtmosphericProperties(altitude);
    
    EXPECT_GT(density, 0.0);
    EXPECT_GT(pressure, 0.0);
    EXPECT_LT(density, phys_.rho0); // Density should decrease with altitude
}

TEST_F(DynamicsTest, ComputeDynamicPressure) {
    double q = dynamics_->computeDynamicPressure(test_state_);
    
    EXPECT_GE(q, 0.0);
    EXPECT_LT(q, 1e6); // Reasonable upper bound
}

TEST_F(DynamicsTest, ComputeAngleOfAttack) {
    double alpha = dynamics_->computeAngleOfAttack(test_state_);
    
    EXPECT_GE(alpha, 0.0);
    EXPECT_LE(alpha, M_PI/2.0); // Should be between 0 and 90 degrees
}

TEST_F(DynamicsTest, ComputeLoadFactor) {
    double n = dynamics_->computeLoadFactor(test_state_, test_control_, 0.0);
    
    EXPECT_GT(n, 0.0);
    EXPECT_LT(n, 100.0); // Reasonable upper bound
}

TEST_F(DynamicsTest, CheckConstraints) {
    Diag diag = dynamics_->checkConstraints(test_state_, test_control_, 0.0);
    
    EXPECT_GE(diag.rho, 0.0);
    EXPECT_GE(diag.q, 0.0);
    EXPECT_GE(diag.alpha, 0.0);
    EXPECT_GE(diag.n, 0.0);
}

// Test Integrator classes
TEST_F(DynamicsTest, RK4Integrator) {
    auto integrator = createRK4Integrator(dynamics_);
    EXPECT_NE(integrator, nullptr);
    
    State new_state = integrator->integrate(test_state_, test_control_, 0.0, 0.1);
    
    // State should be different after integration
    EXPECT_NE(new_state.r_i, test_state_.r_i);
    EXPECT_NE(new_state.v_i, test_state_.v_i);
    EXPECT_NE(new_state.m, test_state_.m);
    
    // Mass should decrease (fuel consumption)
    EXPECT_LT(new_state.m, test_state_.m);
}

TEST_F(DynamicsTest, RK45Integrator) {
    auto integrator = createRK45Integrator(dynamics_, 1e-6, 1e-9, 1.0);
    EXPECT_NE(integrator, nullptr);
    
    State new_state = integrator->integrate(test_state_, test_control_, 0.0, 0.1);
    
    // State should be different after integration
    EXPECT_NE(new_state.r_i, test_state_.r_i);
    EXPECT_NE(new_state.v_i, test_state_.v_i);
    EXPECT_NE(new_state.m, test_state_.m);
    
    // Mass should decrease (fuel consumption)
    EXPECT_LT(new_state.m, test_state_.m);
}

TEST_F(DynamicsTest, IntegratorInterval) {
    auto integrator = createRK4Integrator(dynamics_);
    
    auto control_func = [](double t) -> Control {
        return Control(100000.0, Vec3(0.0, 0.0, 1.0));
    };
    
    std::vector<State> states = integrator->integrateInterval(test_state_, control_func, 0.0, 1.0, 0.1);
    
    EXPECT_GT(states.size(), 1);
    EXPECT_EQ(states[0].r_i, test_state_.r_i);
    
    // Mass should decrease over time
    for (size_t i = 1; i < states.size(); ++i) {
        EXPECT_LT(states[i].m, states[i-1].m);
    }
}

// Test Atmosphere classes
TEST_F(DynamicsTest, ISAAtmosphere) {
    auto atmosphere = createISAAtmosphere();
    EXPECT_NE(atmosphere, nullptr);
    
    double altitude = 1000.0;
    auto [density, pressure] = atmosphere->computeProperties(altitude);
    
    EXPECT_GT(density, 0.0);
    EXPECT_GT(pressure, 0.0);
    EXPECT_LT(density, 1.225); // Should be less than sea level density
}

TEST_F(DynamicsTest, ExponentialAtmosphere) {
    auto atmosphere = createExponentialAtmosphere(1.225, 8400.0, 288.15);
    EXPECT_NE(atmosphere, nullptr);
    
    double altitude = 1000.0;
    double density = atmosphere->computeDensity(altitude);
    double pressure = atmosphere->computePressure(altitude);
    
    EXPECT_GT(density, 0.0);
    EXPECT_GT(pressure, 0.0);
    EXPECT_LT(density, 1.225);
}

TEST_F(DynamicsTest, WindModels) {
    auto no_wind = createNoWindModel();
    EXPECT_NE(no_wind, nullptr);
    
    Vec3 wind = no_wind->computeWind(Vec3::Zero(), 0.0);
    EXPECT_EQ(wind, Vec3::Zero());
    
    auto constant_wind = createConstantWindModel(Vec3(10.0, 5.0, 0.0));
    EXPECT_NE(constant_wind, nullptr);
    
    wind = constant_wind->computeWind(Vec3::Zero(), 0.0);
    EXPECT_EQ(wind, Vec3(10.0, 5.0, 0.0));
}

// Test Constraint classes
TEST_F(DynamicsTest, ConstraintChecker) {
    auto checker = createConstraintChecker(limits_);
    EXPECT_NE(checker, nullptr);
    
    auto violations = checker->checkConstraints(test_state_, test_control_, 0.0);
    EXPECT_GE(violations.size(), 0);
    
    bool has_violations = checker->hasViolations(test_state_, test_control_, 0.0);
    EXPECT_FALSE(has_violations); // Should not have violations with reasonable test data
}

TEST_F(DynamicsTest, ConstraintPenalty) {
    std::vector<double> weights = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    auto penalty = createConstraintPenalty(weights);
    EXPECT_NE(penalty, nullptr);
    
    auto checker = createConstraintChecker(limits_);
    auto violations = checker->checkConstraints(test_state_, test_control_, 0.0);
    
    double penalty_value = penalty->computePenalty(violations);
    EXPECT_GE(penalty_value, 0.0);
}

TEST_F(DynamicsTest, ConstraintHandler) {
    auto checker = createConstraintChecker(limits_);
    auto penalty = createConstraintPenalty({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    auto handler = createConstraintHandler(checker, penalty);
    
    EXPECT_NE(handler, nullptr);
    EXPECT_EQ(handler->getChecker(), checker);
    EXPECT_EQ(handler->getPenalty(), penalty);
    
    Control modified_control = handler->handleViolations(test_state_, test_control_, 0.0);
    EXPECT_EQ(modified_control.T, test_control_.T); // Should be unchanged for valid control
}

// Test error handling
TEST_F(DynamicsTest, InvalidState) {
    State invalid_state;
    invalid_state.m = -1.0; // Invalid mass
    
    EXPECT_THROW(dynamics_->computeDerivative(invalid_state, test_control_, 0.0), std::exception);
}

TEST_F(DynamicsTest, InvalidControl) {
    Control invalid_control;
    invalid_control.T = -1000.0; // Invalid thrust
    
    EXPECT_THROW(dynamics_->computeDerivative(test_state_, invalid_control, 0.0), std::exception);
}

// Test numerical stability
TEST_F(DynamicsTest, NumericalStability) {
    auto integrator = createRK4Integrator(dynamics_);
    
    // Test with very small time step
    State state1 = integrator->integrate(test_state_, test_control_, 0.0, 1e-6);
    State state2 = integrator->integrate(test_state_, test_control_, 0.0, 1e-6);
    
    // Results should be identical for same inputs
    EXPECT_NEAR((state1.r_i - state2.r_i).norm(), 0.0, 1e-10);
    EXPECT_NEAR((state1.v_i - state2.v_i).norm(), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(state1.m - state2.m), 0.0, 1e-10);
}

// Test performance
TEST_F(DynamicsTest, Performance) {
    auto integrator = createRK4Integrator(dynamics_);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < 1000; ++i) {
        integrator->integrate(test_state_, test_control_, 0.0, 0.01);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Should complete 1000 integrations in reasonable time
    EXPECT_LT(duration.count(), 1000000); // Less than 1 second
}