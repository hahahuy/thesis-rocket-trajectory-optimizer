#include <gtest/gtest.h>
#include "physics/dynamics.hpp"
#include <cmath>
#include <vector>
#include <iostream>

using namespace physics;

class DynamicsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard rocket parameters
        params.Cd = 0.3;
        params.A = 1.0;           // 1 m² reference area
        params.Isp = 300.0;       // 300 s specific impulse
        params.Tmax = 50000.0;    // 50 kN max thrust
        params.m_dry = 1000.0;    // 1000 kg dry mass
        params.m_prop = 4000.0;   // 4000 kg propellant
        params.rho0 = 1.225;      // kg/m³ sea level density
        params.H = 8400.0;        // 8.4 km scale height
        params.g0 = 9.81;         // m/s² standard gravity
        params.R_earth = 6.371e6; // Earth radius
        params.enable_wind = false;
        
        // Initial state: on ground, at rest, full fuel
        state0.x = 0.0;
        state0.y = 0.0;
        state0.vx = 0.0;
        state0.vy = 0.0;
        state0.m = params.m_dry + params.m_prop; // 5000 kg total
    }
    
    AscentDynamics::Params params;
    AscentDynamics::State state0;
    
    // Tolerance for floating point comparisons
    const double tol = 1e-6;
};

// Test atmospheric density model
TEST_F(DynamicsTest, AtmosphericDensity) {
    // Sea level density
    double rho_0 = AscentDynamics::atmospheric_density(0.0, params);
    EXPECT_NEAR(rho_0, params.rho0, tol);
    
    // At scale height, density should be 1/e of sea level
    double rho_H = AscentDynamics::atmospheric_density(params.H, params);
    EXPECT_NEAR(rho_H, params.rho0 / M_E, 1e-3);
    
    // At high altitude, density should be very small
    double rho_high = AscentDynamics::atmospheric_density(50000.0, params);
    EXPECT_LT(rho_high, 0.01 * params.rho0);
    
    // Density should decrease monotonically with altitude
    double rho_1km = AscentDynamics::atmospheric_density(1000.0, params);
    double rho_2km = AscentDynamics::atmospheric_density(2000.0, params);
    EXPECT_GT(rho_1km, rho_2km);
}

// Test gravity model
TEST_F(DynamicsTest, GravityModel) {
    // At sea level
    double g_0 = AscentDynamics::gravity(0.0, params);
    EXPECT_NEAR(g_0, params.g0, tol);
    
    // At low altitude (should be constant)
    double g_10km = AscentDynamics::gravity(10000.0, params);
    EXPECT_NEAR(g_10km, params.g0, tol);
    
    // At high altitude (should decrease)
    double g_100km = AscentDynamics::gravity(100000.0, params);
    EXPECT_LT(g_100km, params.g0);
    EXPECT_GT(g_100km, 0.9 * params.g0); // Should still be close for 100 km
}

// Test RHS function with zero control (free fall)
TEST_F(DynamicsTest, FreeFall) {
    AscentDynamics::Control zero_control(0.0, 0.0);
    
    // From rest at sea level
    auto ds_dt = AscentDynamics::rhs(state0, zero_control, params);
    
    // Position derivatives should be velocities
    EXPECT_NEAR(ds_dt.x, state0.vx, tol);
    EXPECT_NEAR(ds_dt.y, state0.vy, tol);
    
    // Horizontal acceleration should be zero (no thrust, no initial velocity)
    EXPECT_NEAR(ds_dt.vx, 0.0, tol);
    
    // Vertical acceleration should be -g (downward)
    EXPECT_NEAR(ds_dt.vy, -params.g0, tol);
    
    // Mass should not change (no thrust)
    EXPECT_NEAR(ds_dt.m, 0.0, tol);
}

// Test RHS function with vertical thrust
TEST_F(DynamicsTest, VerticalThrust) {
    // Thrust exactly balancing weight
    double thrust_mag = state0.m * params.g0;
    AscentDynamics::Control control(thrust_mag, M_PI/2.0); // Vertical thrust
    
    auto ds_dt = AscentDynamics::rhs(state0, control, params);
    
    // Should have near-zero vertical acceleration (thrust balances weight)
    EXPECT_NEAR(ds_dt.vy, 0.0, 1e-3);
    
    // Mass should decrease due to propellant consumption
    double expected_mass_rate = -thrust_mag / (params.Isp * params.g0);
    EXPECT_NEAR(ds_dt.m, expected_mass_rate, tol);
}

// Test mass conservation
TEST_F(DynamicsTest, MassConservation) {
    double thrust = 30000.0; // 30 kN
    AscentDynamics::Control control(thrust, M_PI/2.0);
    
    // Simulate for a short time
    double dt = 0.1; // 0.1 second
    auto rhs_func = [](const AscentDynamics::State& s, 
                      const AscentDynamics::Control& u, 
                      const AscentDynamics::Params& p) {
        return AscentDynamics::rhs(s, u, p);
    };
    
    std::vector<AscentDynamics::Control> controls(10, control); // 1 second total
    auto trajectory = ForwardIntegrator::integrate_rk4(rhs_func, state0, controls, params, dt);
    
    // Check mass decrease
    double initial_mass = trajectory[0].m;
    double final_mass = trajectory.back().m;
    double mass_consumed = initial_mass - final_mass;
    
    // Expected mass consumption: T * t / (Isp * g0)
    double expected_mass_consumed = thrust * 1.0 / (params.Isp * params.g0);
    
    EXPECT_NEAR(mass_consumed, expected_mass_consumed, 0.01); // 1% tolerance
    EXPECT_GE(final_mass, params.m_dry); // Should not go below dry mass
}

// Test energy conservation in vacuum (no drag, no thrust)
TEST_F(DynamicsTest, EnergyConservation) {
    // Start with some velocity
    AscentDynamics::State state_moving = state0;
    state_moving.vy = 100.0; // 100 m/s upward
    
    // No thrust, no atmosphere (set density to zero for this test)
    AscentDynamics::Params params_vacuum = params;
    params_vacuum.rho0 = 0.0; // No atmosphere
    
    AscentDynamics::Control no_control(0.0, 0.0);
    
    auto rhs_func = [](const AscentDynamics::State& s, 
                      const AscentDynamics::Control& u, 
                      const AscentDynamics::Params& p) {
        return AscentDynamics::rhs(s, u, p);
    };
    
    // Simulate trajectory
    double dt = 0.01;
    std::vector<AscentDynamics::Control> controls(1000, no_control); // 10 seconds
    auto trajectory = ForwardIntegrator::integrate_rk4(rhs_func, state_moving, controls, params_vacuum, dt);
    
    // Calculate total energy at different points
    auto total_energy = [&](const AscentDynamics::State& s) {
        double kinetic = 0.5 * s.m * (s.vx * s.vx + s.vy * s.vy);
        double potential = s.m * params.g0 * s.y;
        return kinetic + potential;
    };
    
    double E0 = total_energy(trajectory[0]);
    double E_mid = total_energy(trajectory[trajectory.size()/2]);
    double E_final = total_energy(trajectory.back());
    
    // Energy should be conserved (within numerical tolerance)
    EXPECT_NEAR(E0, E_mid, 0.01 * E0);   // 1% tolerance
    EXPECT_NEAR(E0, E_final, 0.01 * E0); // 1% tolerance
}

// Test integrator accuracy
TEST_F(DynamicsTest, IntegratorAccuracy) {
    // Simple test case: constant acceleration
    AscentDynamics::Control control(state0.m * params.g0 * 1.5, M_PI/2.0); // 50% more than weight for clear upward motion
    
    auto rhs_func = [](const AscentDynamics::State& s, 
                      const AscentDynamics::Control& u, 
                      const AscentDynamics::Params& p) {
        return AscentDynamics::rhs(s, u, p);
    };
    
    // Compare different step sizes
    double dt_coarse = 0.1;
    double dt_fine = 0.01;
    
    std::vector<AscentDynamics::Control> controls_coarse(10, control);  // 1 second
    std::vector<AscentDynamics::Control> controls_fine(100, control);   // 1 second
    
    auto traj_coarse = ForwardIntegrator::integrate_rk4(rhs_func, state0, controls_coarse, params, dt_coarse);
    auto traj_fine = ForwardIntegrator::integrate_rk4(rhs_func, state0, controls_fine, params, dt_fine);
    
    // Both should have positive altitude
    EXPECT_GT(traj_coarse.back().y, 0.0);
    EXPECT_GT(traj_fine.back().y, 0.0);
    
    // Check final positions are reasonably close
    double error_y = std::abs(traj_coarse.back().y - traj_fine.back().y);
    EXPECT_LT(error_y, 1.0); // Less than 1 meter difference
    
    // For this simple case, just check they're close
    EXPECT_NEAR(traj_coarse.back().y, traj_fine.back().y, 1.0);
}

// Test adaptive integrator
TEST_F(DynamicsTest, AdaptiveIntegrator) {
    auto rhs_func = [](const AscentDynamics::State& s, 
                      const AscentDynamics::Control& u, 
                      const AscentDynamics::Params& p) {
        return AscentDynamics::rhs(s, u, p);
    };
    
    // Constant control function with sufficient thrust
    auto control_func = [&](double t) {
        return AscentDynamics::Control(state0.m * params.g0 * 2.0, M_PI/2.0); // 2x weight in thrust
    };
    
    // Test adaptive integrator
    auto trajectory = ForwardIntegrator::integrate_rk45(
        rhs_func, state0, control_func, params, 
        0.0, 2.0,    // 0 to 2 seconds (shorter time for more predictable behavior)
        1e-6, 1e-8,  // Tolerances
        0.1          // Max step
    );
    
    // Should have reasonable number of points
    EXPECT_GT(trajectory.size(), 5);
    EXPECT_LT(trajectory.size(), 1000);
    
    // Time should be monotonically increasing
    for (size_t i = 1; i < trajectory.size(); ++i) {
        EXPECT_GT(trajectory[i].first, trajectory[i-1].first);
    }
    
    // Final time should be close to 2.0
    EXPECT_NEAR(trajectory.back().first, 2.0, 1e-6);
    
    // Rocket should have gained altitude (with 2x thrust, should definitely go up)
    EXPECT_GT(trajectory.back().second.y, 5.0); // At least 5 meters up
    
    // Mass should have decreased
    EXPECT_LT(trajectory.back().second.m, state0.m);
    
    // Rocket should have positive upward velocity
    EXPECT_GT(trajectory.back().second.vy, 0.0);
}

// Test wind effects
TEST_F(DynamicsTest, WindEffects) {
    params.enable_wind = true;
    
    // State with horizontal velocity
    AscentDynamics::State state_horizontal = state0;
    state_horizontal.vx = 50.0; // 50 m/s horizontal
    state_horizontal.y = 1000.0; // 1 km altitude
    
    AscentDynamics::Control no_control(0.0, 0.0);
    
    auto ds_dt_no_wind = AscentDynamics::rhs(state_horizontal, no_control, params);
    
    // With wind enabled, there should be additional drag effects
    // (The wind model creates relative velocity, affecting drag)
    
    // This is more of a smoke test - wind should affect the dynamics
    // The exact values depend on the wind model implementation
    EXPECT_TRUE(std::isfinite(ds_dt_no_wind.vx));
    EXPECT_TRUE(std::isfinite(ds_dt_no_wind.vy));
}

// Test edge cases
TEST_F(DynamicsTest, EdgeCases) {
    // Test with very small mass (near dry mass)
    AscentDynamics::State low_mass_state = state0;
    low_mass_state.m = params.m_dry + 1.0; // Just 1 kg of fuel left
    
    AscentDynamics::Control high_thrust(params.Tmax, M_PI/2.0);
    
    auto ds_dt = AscentDynamics::rhs(low_mass_state, high_thrust, params);
    
    // Should not allow mass to go below dry mass
    EXPECT_GE(low_mass_state.m + ds_dt.m, params.m_dry - tol);
    
    // Test with zero velocity (should not crash)
    auto ds_dt_zero_v = AscentDynamics::rhs(state0, high_thrust, params);
    EXPECT_TRUE(std::isfinite(ds_dt_zero_v.vx));
    EXPECT_TRUE(std::isfinite(ds_dt_zero_v.vy));
    
    // Test at very high altitude
    AscentDynamics::State high_alt_state = state0;
    high_alt_state.y = 200000.0; // 200 km
    
    auto ds_dt_high = AscentDynamics::rhs(high_alt_state, high_thrust, params);
    EXPECT_TRUE(std::isfinite(ds_dt_high.vx));
    EXPECT_TRUE(std::isfinite(ds_dt_high.vy));
}

// Performance test for integrator
TEST_F(DynamicsTest, IntegratorPerformance) {
    auto rhs_func = [](const AscentDynamics::State& s, 
                      const AscentDynamics::Control& u, 
                      const AscentDynamics::Params& p) {
        return AscentDynamics::rhs(s, u, p);
    };
    
    AscentDynamics::Control control(25000.0, M_PI/4.0);
    std::vector<AscentDynamics::Control> controls(1000, control); // 1000 steps
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto trajectory = ForwardIntegrator::integrate_rk4(rhs_func, state0, controls, params, 0.01);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Should complete in reasonable time (less than 1 second for 1000 steps)
    EXPECT_LT(duration.count(), 1000);
    
    // Should produce expected number of points
    EXPECT_EQ(trajectory.size(), 1001); // Initial + 1000 steps
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
