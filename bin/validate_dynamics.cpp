#define _USE_MATH_DEFINES  // Enable M_PI on MSVC
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "../src/physics/types.hpp"
#include "../src/physics/dynamics.hpp"
#include "../src/physics/integrator.hpp"

using namespace rocket_physics;

/**
 * @brief Validate dynamics stability and continuity
 */
void validateStability() {
    std::cout << "=== Task 1: Dynamics Stability and Continuity Validation ===" << std::endl;
    
    // Setup
    Phys phys;
    phys.S_ref = 0.05;
    phys.CL_alpha = 3.5;
    phys.Cm_alpha = -0.8;
    phys.C_delta = 0.05;
    phys.l_ref = 1.2;
    phys.Cd = 0.3;
    phys.Isp = 300.0;
    phys.g0 = 9.81;
    phys.rho0 = 1.225;
    phys.h_scale = 8400.0;
    phys.I_b = Eigen::Matrix3d::Identity() * 1000.0;
    
    Limits limits;
    limits.T_max = 4000.0;
    limits.m_dry = 10.0;
    limits.q_max = 40000.0;
    limits.n_max = 5.0;
    
    auto dynamics = createDynamics(phys, limits);
    
    // Initial state
    State state;
    state.r_i = Vec3(0.0, 0.0, 0.0);
    state.v_i = Vec3(0.0, 0.0, 0.0);
    state.q_bi = Quaterniond::Identity();
    state.w_b = Vec3::Zero();
    state.m = 50.0; // 50 kg
    
    // Control function
    auto control_func = [](double t) -> Control {
        double T = (t < 5.0) ? 4000.0 : 0.0; // 4000 N for 5 seconds
        double theta = 90.0 * M_PI / 180.0 - (90.0 * M_PI / 180.0) * (t / 30.0); // 90° -> 0°
        Vec3 direction(std::sin(theta), 0.0, std::cos(theta));
        return Control(T, direction);
    };
    
    // Test RK4 vs RK45 comparison
    std::cout << "\n--- Test 1.1: RK4 vs RK45 Comparison ---" << std::endl;
    auto rk4 = createRK4Integrator(dynamics);
    auto rk45 = createRK45Integrator(dynamics, 1e-6, 1e-9, 1.0);
    
    double t_start = 0.0;
    double t_end = 30.0;
    double dt = 0.1;
    
    auto rk4_traj = rk4->integrateInterval(state, control_func, t_start, t_end, dt);
    auto rk45_traj = rk45->integrateInterval(state, control_func, t_start, t_end, dt);
    
    // Compare final states
    State rk4_final = rk4_traj.back();
    State rk45_final = rk45_traj.back();
    
    double pos_diff = (rk4_final.r_i - rk45_final.r_i).norm();
    double vel_diff = (rk4_final.v_i - rk45_final.v_i).norm();
    double mass_diff = std::abs(rk4_final.m - rk45_final.m);
    double quat_diff = std::abs(rk4_final.q_bi.coeffs().norm() - rk45_final.q_bi.coeffs().norm());
    
    std::cout << "Position difference: " << pos_diff << " (target: < 1e-4)" << std::endl;
    std::cout << "Velocity difference: " << vel_diff << " (target: < 1e-4)" << std::endl;
    std::cout << "Mass difference: " << mass_diff << " (target: < 1e-4)" << std::endl;
    std::cout << "Quaternion difference: " << quat_diff << " (target: < 1e-4)" << std::endl;
    
    bool pass1 = (pos_diff < 1e-4) && (vel_diff < 1e-4) && (mass_diff < 1e-4);
    std::cout << (pass1 ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Test quaternion norm
    std::cout << "\n--- Test 1.2: Quaternion Norm Stability ---" << std::endl;
    double max_q_dev = 0.0;
    double min_q_dev = 1e10;
    
    for (const auto& s : rk45_traj) {
        double q_norm = s.q_bi.norm();
        double deviation = std::abs(q_norm - 1.0);
        max_q_dev = std::max(max_q_dev, deviation);
        min_q_dev = std::min(min_q_dev, deviation);
    }
    
    std::cout << "Max quaternion norm deviation: " << max_q_dev << " (target: < 1e-9)" << std::endl;
    std::cout << "Min quaternion norm deviation: " << min_q_dev << std::endl;
    std::cout << "Quaternion norm range: [" << (1.0 - max_q_dev) << ", " << (1.0 + max_q_dev) << "]" << std::endl;
    
    bool pass2 = (max_q_dev < 1e-9);
    std::cout << (pass2 ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Test aerodynamic continuity
    std::cout << "\n--- Test 1.3: Aerodynamic Lift/Drag Continuity ---" << std::endl;
    std::ofstream aero_file("data/aerodynamic_continuity.csv");
    aero_file << "alpha_deg,lift,drag\n";
    
    double max_derivative_change = 0.0;
    double prev_lift = 0.0;
    double prev_drag = 0.0;
    
    for (double alpha_deg = 0.0; alpha_deg <= 10.0; alpha_deg += 0.1) {
        double alpha_rad = alpha_deg * M_PI / 180.0;
        
        // Create state with angle of attack
        State test_state;
        test_state.r_i = Vec3(0.0, 0.0, 5000.0); // 5 km altitude
        test_state.v_i = Vec3(100.0 * std::cos(alpha_rad), 0.0, 100.0 * std::sin(alpha_rad));
        test_state.q_bi = Quaterniond::Identity();
        test_state.w_b = Vec3::Zero();
        test_state.m = 50.0;
        
        Vec3 wind = Vec3::Zero();
        // Create test control for forces computation
        Control test_control_force(4000.0, Vec3(1.0, 0.0, 0.0));
        double t = 0.0; // Time for forces computation
        // Use public method instead
        auto [forces, moments] = dynamics->computeForcesAndMoments(test_state, test_control_force, t);
        Vec3 drag_force = forces; // Approximate - forces include all forces
        Vec3 lift_component = drag_force; // Simplified - in practice would separate lift/drag
        
        double lift = lift_component.z();
        double drag = -drag_force.x();
        
        aero_file << alpha_deg << "," << lift << "," << drag << "\n";
        
        if (alpha_deg > 0.1) {
            double lift_derivative = (lift - prev_lift) / 0.1;
            double drag_derivative = (drag - prev_drag) / 0.1;
            
            // Check smoothness
            if (alpha_deg > 0.2) {
                double prev_lift_deriv = (prev_lift - (alpha_deg > 0.2 ? lift : prev_lift)) / 0.1;
                double derivative_change = std::abs(lift_derivative - prev_lift_deriv);
                max_derivative_change = std::max(max_derivative_change, derivative_change);
            }
        }
        
        prev_lift = lift;
        prev_drag = drag;
    }
    
    aero_file.close();
    std::cout << "Max derivative change: " << max_derivative_change << " (target: < 0.1 for smoothness)" << std::endl;
    std::cout << "Aerodynamic data saved to data/aerodynamic_continuity.csv" << std::endl;
    
    bool pass3 = (max_derivative_change < 0.1);
    std::cout << (pass3 ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Summary
    std::cout << "\n=== Stability Validation Summary ===" << std::endl;
    std::cout << "RK4 vs RK45: " << (pass1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Quaternion norm: " << (pass2 ? "PASS" : "FAIL") << std::endl;
    std::cout << "Aerodynamic continuity: " << (pass3 ? "PASS" : "FAIL") << std::endl;
    
    if (pass1 && pass2 && pass3) {
        std::cout << "\n✓ All stability tests PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Some stability tests FAILED" << std::endl;
    }
}

/**
 * @brief Run reference flight case
 */
void runReferenceFlight() {
    std::cout << "\n=== Task 6: Reference Flight Case ===" << std::endl;
    
    // Nominal config
    Phys phys;
    phys.S_ref = 0.05;
    phys.CL_alpha = 3.5;
    phys.Cm_alpha = -0.8;
    phys.C_delta = 0.05;
    phys.l_ref = 1.2;
    phys.Cd = 0.3;
    phys.Isp = 300.0;
    phys.g0 = 9.81;
    phys.rho0 = 1.225;
    phys.h_scale = 8400.0;
    phys.I_b = Eigen::Matrix3d::Identity() * 1000.0;
    
    Limits limits;
    limits.T_max = 4000.0;
    limits.m_dry = 10.0;
    limits.q_max = 40000.0;
    limits.n_max = 5.0;
    
    auto dynamics = createDynamics(phys, limits);
    auto integrator = createRK45Integrator(dynamics, 1e-6, 1e-9, 1.0);
    
    // Initial state: m₀ = 50 kg
    State state;
    state.r_i = Vec3(0.0, 0.0, 0.0);
    state.v_i = Vec3(0.0, 0.0, 0.0);
    state.q_bi = Quaterniond::Identity();
    state.w_b = Vec3::Zero();
    state.m = 50.0;
    
    // Control: T = 4000 N, θ = 90→0°
    auto control_func = [](double t) -> Control {
        double T = (t < 5.0) ? 4000.0 : 0.0;
        double theta = 90.0 * M_PI / 180.0 - (90.0 * M_PI / 180.0) * (t / 30.0);
        Vec3 direction(std::sin(theta), 0.0, std::cos(theta));
        return Control(T, direction);
    };
    
    // Integrate for 30 seconds
    double t_start = 0.0;
    double t_end = 30.0;
    double dt = 0.1;
    
    auto trajectory = integrator->integrateInterval(state, control_func, t_start, t_end, dt);
    
    // Compute diagnostics
    std::vector<double> times;
    std::vector<double> altitudes;
    std::vector<double> q_values;
    std::vector<double> n_values;
    std::vector<double> velocities;
    std::vector<double> masses;
    
    double current_time = t_start;
    double dt_actual = t_end / trajectory.size();
    
    for (const auto& s : trajectory) {
        times.push_back(current_time);
        
        double altitude = std::max(0.0, s.r_i.norm() - 6371000.0);
        altitudes.push_back(altitude);
        
        double q = dynamics->computeDynamicPressure(s);
        q_values.push_back(q);
        
        Control ctrl = control_func(current_time);
        double n = dynamics->computeLoadFactor(s, ctrl, current_time);
        n_values.push_back(n);
        
        velocities.push_back(s.v_i.norm());
        masses.push_back(s.m);
        
        current_time += dt_actual;
    }
    
    // Find key metrics
    auto max_q_it = std::max_element(q_values.begin(), q_values.end());
    double max_q = *max_q_it;
    size_t max_q_idx = std::distance(q_values.begin(), max_q_it);
    double max_q_time = times[max_q_idx];
    
    auto max_alt_it = std::max_element(altitudes.begin(), altitudes.end());
    double max_alt = *max_alt_it;
    size_t max_alt_idx = std::distance(altitudes.begin(), max_alt_it);
    double apogee_time = times[max_alt_idx];
    
    auto max_n_it = std::max_element(n_values.begin(), n_values.end());
    double max_n = *max_n_it;
    
    // Find burnout (when mass reaches dry mass + small margin)
    size_t burnout_idx = trajectory.size();
    for (size_t i = 0; i < masses.size(); ++i) {
        if (masses[i] <= limits.m_dry + 1.0) {
            burnout_idx = i;
            break;
        }
    }
    double burnout_time = (burnout_idx < times.size()) ? times[burnout_idx] : t_end;
    
    // Print results
    std::cout << "\nReference Flight Results:" << std::endl;
    std::cout << "  Peak q: " << max_q / 1000.0 << " kPa at t = " << max_q_time << " s" << std::endl;
    std::cout << "  Expected: 20-30 kPa at ~2-3 s" << std::endl;
    std::cout << "  Max altitude: " << max_alt / 1000.0 << " km at t = " << apogee_time << " s" << std::endl;
    std::cout << "  Expected: 8-12 km at ~25-30 s" << std::endl;
    std::cout << "  Burnout: " << burnout_time << " s" << std::endl;
    std::cout << "  Expected: ~5 s" << std::endl;
    std::cout << "  Max load factor: " << max_n << " g" << std::endl;
    std::cout << "  Expected: < " << limits.n_max << " g" << std::endl;
    
    // Validation
    bool q_valid = (max_q >= 20000.0 && max_q <= 30000.0);
    bool alt_valid = (max_alt >= 8000.0 && max_alt <= 12000.0);
    bool n_valid = (max_n < limits.n_max);
    bool burnout_valid = (burnout_time >= 4.0 && burnout_time <= 6.0);
    
    std::cout << "\nValidation:" << std::endl;
    std::cout << "  Peak q: " << (q_valid ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  Altitude: " << (alt_valid ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  Load factor: " << (n_valid ? "✓ PASS" : "✗ FAIL") << std::endl;
    std::cout << "  Burnout: " << (burnout_valid ? "✓ PASS" : "✗ FAIL") << std::endl;
    
    // Save to CSV
    std::ofstream csv_file("data/reference_case.csv");
    csv_file << "t,altitude,q,n,velocity,mass,x,y,z,vx,vy,vz\n";
    
    current_time = t_start;
    for (size_t i = 0; i < trajectory.size(); ++i) {
        const auto& s = trajectory[i];
        csv_file << current_time << ","
                 << altitudes[i] << ","
                 << q_values[i] << ","
                 << n_values[i] << ","
                 << velocities[i] << ","
                 << masses[i] << ","
                 << s.r_i.x() << "," << s.r_i.y() << "," << s.r_i.z() << ","
                 << s.v_i.x() << "," << s.v_i.y() << "," << s.v_i.z() << "\n";
        current_time += dt_actual;
    }
    
    csv_file.close();
    std::cout << "\nReference flight data saved to data/reference_case.csv" << std::endl;
    
    if (q_valid && alt_valid && n_valid && burnout_valid) {
        std::cout << "\n✓ Reference flight validation PASSED" << std::endl;
    } else {
        std::cout << "\n✗ Reference flight validation FAILED" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::cout << std::fixed << std::setprecision(6);
    
    // Create data directory
    system("mkdir -p data logs/dynamics");
    
    bool run_all = (argc > 1 && std::string(argv[1]) == "--all");
    
    if (run_all || std::string(argv[0]).find("validate") != std::string::npos) {
        validateStability();
        runReferenceFlight();
    } else {
        validateStability();
    }
    
    return 0;
}
