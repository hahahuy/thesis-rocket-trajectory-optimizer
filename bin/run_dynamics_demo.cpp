#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "../src/physics/types.hpp"
#include "../src/physics/dynamics.hpp"
#include "../src/physics/integrator.hpp"
#include "../src/physics/atmosphere.hpp"
#include "../src/physics/constraints.hpp"

using namespace rocket_physics;

int main() {
    std::cout << "=== Rocket Dynamics Demo ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    // Create physical parameters
    Phys phys;
    phys.Cd = 0.3;
    phys.Cl = 0.0;
    phys.S_ref = 1.0;
    phys.I_b = Matrix3d::Identity() * 1000.0;
    phys.r_cg = Vec3::Zero();
    phys.Isp = 300.0;
    phys.g0 = 9.81;
    phys.rho0 = 1.225;
    phys.h_scale = 8400.0;
    
    // Create operational limits
    Limits limits;
    limits.T_max = 1000000.0;
    limits.m_dry = 1000.0;
    limits.q_max = 50000.0;
    limits.w_gimbal_max = 1.0;
    limits.alpha_max = 0.1;
    limits.n_max = 10.0;
    
    // Create dynamics object
    auto dynamics = createDynamics(phys, limits);
    
    // Create integrators
    auto rk4_integrator = createRK4Integrator(dynamics);
    auto rk45_integrator = createRK45Integrator(dynamics, 1e-6, 1e-9, 1.0);
    
    // Create atmosphere models
    auto isa_atmosphere = createISAAtmosphere();
    auto exponential_atmosphere = createExponentialAtmosphere();
    
    // Create wind models
    auto no_wind = createNoWindModel();
    auto constant_wind = createConstantWindModel(Vec3(10.0, 5.0, 0.0));
    
    // Create constraint checker
    auto constraint_checker = createConstraintChecker(limits);
    auto constraint_penalty = createConstraintPenalty({1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
    auto constraint_handler = createConstraintHandler(constraint_checker, constraint_penalty);
    
    // Initial state
    State initial_state;
    initial_state.r_i = Vec3(0.0, 0.0, 0.0);
    initial_state.v_i = Vec3(0.0, 0.0, 0.0);
    initial_state.q_bi = Quaterniond::Identity();
    initial_state.w_b = Vec3::Zero();
    initial_state.m = 5000.0;
    
    // Control function
    auto control_func = [](double t) -> Control {
        if (t < 10.0) {
            return Control(100000.0, Vec3(0.0, 0.0, 1.0));
        } else {
            return Control(0.0, Vec3(0.0, 0.0, 1.0));
        }
    };
    
    std::cout << "\n--- Initial State ---" << std::endl;
    std::cout << "Position [m]: (" << initial_state.r_i.transpose() << ")" << std::endl;
    std::cout << "Velocity [m/s]: (" << initial_state.v_i.transpose() << ")" << std::endl;
    std::cout << "Mass [kg]: " << initial_state.m << std::endl;
    
    // Test single integration step
    std::cout << "\n--- Single Integration Step ---" << std::endl;
    Control test_control = control_func(0.0);
    State state_after_step = rk4_integrator->integrate(initial_state, test_control, 0.0, 0.1);
    
    std::cout << "After 0.1s step:" << std::endl;
    std::cout << "Position [m]: (" << state_after_step.r_i.transpose() << ")" << std::endl;
    std::cout << "Velocity [m/s]: (" << state_after_step.v_i.transpose() << ")" << std::endl;
    std::cout << "Mass [kg]: " << state_after_step.m << std::endl;
    
    // Test atmospheric properties
    std::cout << "\n--- Atmospheric Properties ---" << std::endl;
    double altitude = 1000.0;
    auto [density, pressure] = isa_atmosphere->computeProperties(altitude);
    std::cout << "At altitude " << altitude << "m:" << std::endl;
    std::cout << "Density [kg/m³]: " << density << std::endl;
    std::cout << "Pressure [Pa]: " << pressure << std::endl;
    std::cout << "Temperature [K]: " << isa_atmosphere->computeTemperature(altitude) << std::endl;
    std::cout << "Speed of sound [m/s]: " << isa_atmosphere->computeSpeedOfSound(altitude) << std::endl;
    
    // Test wind models
    std::cout << "\n--- Wind Models ---" << std::endl;
    Vec3 wind_no = no_wind->computeWind(Vec3::Zero(), 0.0);
    Vec3 wind_constant = constant_wind->computeWind(Vec3::Zero(), 0.0);
    std::cout << "No wind: (" << wind_no.transpose() << ") m/s" << std::endl;
    std::cout << "Constant wind: (" << wind_constant.transpose() << ") m/s" << std::endl;
    
    // Test constraint checking
    std::cout << "\n--- Constraint Checking ---" << std::endl;
    auto violations = constraint_checker->checkConstraints(initial_state, test_control, 0.0);
    std::cout << "Number of constraint violations: " << violations.size() << std::endl;
    
    bool has_violations = constraint_checker->hasViolations(initial_state, test_control, 0.0);
    std::cout << "Has violations: " << (has_violations ? "Yes" : "No") << std::endl;
    
    if (has_violations) {
        int violation_count = constraint_checker->getViolationCount(initial_state, test_control, 0.0);
        double max_violation = constraint_checker->getMaxViolationMagnitude(initial_state, test_control, 0.0);
        std::cout << "Violation count: " << violation_count << std::endl;
        std::cout << "Max violation magnitude: " << max_violation << std::endl;
    }
    
    // Test trajectory integration
    std::cout << "\n--- Trajectory Integration ---" << std::endl;
    double t_start = 0.0;
    double t_end = 5.0;
    double dt = 0.1;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<State> trajectory = rk4_integrator->integrateInterval(initial_state, control_func, t_start, t_end, dt);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Integration time: " << duration.count() << " μs" << std::endl;
    std::cout << "Number of states: " << trajectory.size() << std::endl;
    
    // Display trajectory summary
    if (!trajectory.empty()) {
        State final_state = trajectory.back();
        std::cout << "\nFinal state:" << std::endl;
        std::cout << "Position [m]: (" << final_state.r_i.transpose() << ")" << std::endl;
        std::cout << "Velocity [m/s]: (" << final_state.v_i.transpose() << ")" << std::endl;
        std::cout << "Mass [kg]: " << final_state.m << std::endl;
        
        double altitude = final_state.r_i.norm() - 6371000.0; // Earth radius
        std::cout << "Altitude [m]: " << altitude << std::endl;
        
        double velocity = final_state.v_i.norm();
        std::cout << "Velocity magnitude [m/s]: " << velocity << std::endl;
        
        double dynamic_pressure = dynamics->computeDynamicPressure(final_state);
        std::cout << "Dynamic pressure [Pa]: " << dynamic_pressure << std::endl;
        
        double angle_of_attack = dynamics->computeAngleOfAttack(final_state);
        std::cout << "Angle of attack [rad]: " << angle_of_attack << std::endl;
        
        double load_factor = dynamics->computeLoadFactor(final_state, test_control, t_end);
        std::cout << "Load factor [g]: " << load_factor << std::endl;
    }
    
    // Test RK45 integrator
    std::cout << "\n--- RK45 Adaptive Integration ---" << std::endl;
    auto rk45_start = std::chrono::high_resolution_clock::now();
    std::vector<State> rk45_trajectory = rk45_integrator->integrateInterval(initial_state, control_func, t_start, t_end, dt);
    auto rk45_end = std::chrono::high_resolution_clock::now();
    
    auto rk45_duration = std::chrono::duration_cast<std::chrono::microseconds>(rk45_end - rk45_start);
    
    std::cout << "RK45 integration time: " << rk45_duration.count() << " μs" << std::endl;
    std::cout << "RK45 function evaluations: " << rk45_integrator->getFunctionEvaluations() << std::endl;
    std::cout << "RK45 last step size: " << rk45_integrator->getLastStepSize() << std::endl;
    
    // Compare RK4 and RK45 results
    if (!trajectory.empty() && !rk45_trajectory.empty()) {
        State rk4_final = trajectory.back();
        State rk45_final = rk45_trajectory.back();
        
        double position_error = (rk4_final.r_i - rk45_final.r_i).norm();
        double velocity_error = (rk4_final.v_i - rk45_final.v_i).norm();
        double mass_error = std::abs(rk4_final.m - rk45_final.m);
        
        std::cout << "\n--- Integration Comparison ---" << std::endl;
        std::cout << "Position error [m]: " << position_error << std::endl;
        std::cout << "Velocity error [m/s]: " << velocity_error << std::endl;
        std::cout << "Mass error [kg]: " << mass_error << std::endl;
    }
    
    // Test constraint handling
    std::cout << "\n--- Constraint Handling ---" << std::endl;
    Control excessive_control(2000000.0, Vec3(0.0, 0.0, 1.0)); // Exceeds thrust limit
    Control handled_control = constraint_handler->handleViolations(initial_state, excessive_control, 0.0);
    
    std::cout << "Original thrust [N]: " << excessive_control.T << std::endl;
    std::cout << "Handled thrust [N]: " << handled_control.T << std::endl;
    std::cout << "Thrust limit [N]: " << limits.T_max << std::endl;
    
    // Test diagnostic information
    std::cout << "\n--- Diagnostic Information ---" << std::endl;
    Diag diag = dynamics->checkConstraints(initial_state, test_control, 0.0);
    std::cout << "Atmospheric density [kg/m³]: " << diag.rho << std::endl;
    std::cout << "Dynamic pressure [Pa]: " << diag.q << std::endl;
    std::cout << "Q violation: " << (diag.q_violation ? "Yes" : "No") << std::endl;
    std::cout << "Mass underflow: " << (diag.m_underflow ? "Yes" : "No") << std::endl;
    std::cout << "Angle of attack [rad]: " << diag.alpha << std::endl;
    std::cout << "Load factor [g]: " << diag.n << std::endl;
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    
    return 0;
}
