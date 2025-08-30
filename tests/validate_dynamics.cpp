#include "physics/dynamics.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <cassert>

using namespace physics;

/**
 * @brief Validation test for rocket dynamics
 * 
 * This program validates the physics implementation by:
 * 1. Running a simple vertical ascent trajectory
 * 2. Checking physical behavior (energy, mass conservation)
 * 3. Comparing RK4 vs RK45 integrators
 * 4. Outputting trajectory data for plotting
 */

void print_state(const AscentDynamics::State& s, double t) {
    std::cout << std::fixed << std::setprecision(2)
              << "t=" << std::setw(6) << t 
              << " x=" << std::setw(8) << s.x/1000.0 << "km"
              << " y=" << std::setw(8) << s.y/1000.0 << "km"
              << " vx=" << std::setw(7) << s.vx << "m/s"
              << " vy=" << std::setw(7) << s.vy << "m/s"
              << " m=" << std::setw(6) << s.m/1000.0 << "t" << std::endl;
}

void save_trajectory_csv(const std::vector<std::pair<double, AscentDynamics::State>>& trajectory,
                        const std::string& filename) {
    std::ofstream file(filename);
    file << "time,x,y,vx,vy,mass,altitude_km,speed_ms,energy_mj\n";
    
    for (const auto& [t, s] : trajectory) {
        double speed = std::sqrt(s.vx*s.vx + s.vy*s.vy);
        double kinetic_energy = 0.5 * s.m * speed * speed / 1e6; // MJ
        double potential_energy = s.m * 9.81 * s.y / 1e6; // MJ
        double total_energy = kinetic_energy + potential_energy;
        
        file << std::fixed << std::setprecision(6)
             << t << ","
             << s.x << ","
             << s.y << ","
             << s.vx << ","
             << s.vy << ","
             << s.m << ","
             << s.y/1000.0 << ","
             << speed << ","
             << total_energy << "\n";
    }
    file.close();
    std::cout << "Saved trajectory to " << filename << std::endl;
}

int main() {
    std::cout << "=== Rocket Dynamics Validation ===" << std::endl;
    
    // Setup rocket parameters (ensure T/W > 1 for ascent)
    AscentDynamics::Params params;
    params.Cd = 0.3;
    params.A = 1.0;           // 1 m² reference area
    params.Isp = 300.0;       // 300 s specific impulse
    params.Tmax = 150000.0;   // 150 kN max thrust (increased for T/W > 1)
    params.m_dry = 2000.0;    // 2000 kg dry mass
    params.m_prop = 8000.0;   // 8000 kg propellant
    params.rho0 = 1.225;      // kg/m³ sea level density
    params.H = 8400.0;        // 8.4 km scale height
    params.g0 = 9.81;         // m/s² standard gravity
    params.enable_wind = false;
    
    std::cout << "Rocket parameters:" << std::endl;
    std::cout << "  Total mass: " << (params.m_dry + params.m_prop)/1000.0 << " tons" << std::endl;
    std::cout << "  Max thrust: " << params.Tmax/1000.0 << " kN" << std::endl;
    std::cout << "  Thrust-to-weight: " << params.Tmax/(params.m_dry + params.m_prop)/params.g0 << std::endl;
    std::cout << "  Specific impulse: " << params.Isp << " s" << std::endl;
    
    // Initial state
    AscentDynamics::State state0;
    state0.x = 0.0;
    state0.y = 0.0;
    state0.vx = 0.0;
    state0.vy = 0.0;
    state0.m = params.m_dry + params.m_prop;
    
    auto rhs_func = [](const AscentDynamics::State& s, 
                      const AscentDynamics::Control& u, 
                      const AscentDynamics::Params& p) {
        return AscentDynamics::rhs(s, u, p);
    };
    
    // Test 1: Vertical ascent with gravity turn
    std::cout << "\n=== Test 1: Vertical Ascent with Gravity Turn ===" << std::endl;
    
    auto control_func = [&](double t) -> AscentDynamics::Control {
        // Launch profile:
        // 0-10s: Vertical ascent at 80% thrust
        // 10-30s: Gravity turn, reducing thrust angle
        // 30-60s: More horizontal, reduced thrust
        
        double thrust_fraction = 0.8; // 80% of max thrust
        double thrust = params.Tmax * thrust_fraction;
        double angle;
        
        if (t < 10.0) {
            angle = M_PI/2.0; // Vertical
        } else if (t < 30.0) {
            // Gravity turn: linearly reduce angle from 90° to 45°
            double progress = (t - 10.0) / 20.0;
            angle = M_PI/2.0 * (1.0 - 0.5 * progress);
        } else {
            // Continue turning more horizontal
            double progress = std::min(1.0, (t - 30.0) / 30.0);
            angle = M_PI/4.0 * (1.0 - 0.7 * progress); // Down to ~13°
            thrust_fraction = 0.6; // Reduce thrust
            thrust = params.Tmax * thrust_fraction;
        }
        
        return AscentDynamics::Control(thrust, angle);
    };
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Simulate using adaptive RK45
    auto trajectory = ForwardIntegrator::integrate_rk45(
        rhs_func, state0, control_func, params,
        0.0, 60.0,    // 0 to 60 seconds
        1e-6, 1e-8,   // Tolerances
        0.5           // Max step
    );
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Simulation completed in " << duration.count() << " ms" << std::endl;
    std::cout << "Generated " << trajectory.size() << " trajectory points" << std::endl;
    
    // Print key trajectory points
    std::cout << "\nTrajectory milestones:" << std::endl;
    print_state(trajectory[0].second, trajectory[0].first);
    
    for (size_t i = 0; i < trajectory.size(); ++i) {
        double t = trajectory[i].first;
        if (std::abs(t - 10.0) < 0.1 || std::abs(t - 30.0) < 0.1 || std::abs(t - 60.0) < 0.1) {
            print_state(trajectory[i].second, t);
        }
    }
    
    // Validate physical behavior
    std::cout << "\n=== Physical Validation ===" << std::endl;
    
    const auto& final_state = trajectory.back().second;
    double final_time = trajectory.back().first;
    
    // Mass conservation check
    double initial_mass = trajectory[0].second.m;
    double mass_consumed = initial_mass - final_state.m;
    std::cout << "Mass consumed: " << mass_consumed/1000.0 << " tons" << std::endl;
    std::cout << "Remaining fuel: " << (final_state.m - params.m_dry)/1000.0 << " tons" << std::endl;
    
    // Performance metrics
    double max_altitude = 0.0;
    double max_speed = 0.0;
    double max_horizontal_distance = 0.0;
    
    for (const auto& [t, s] : trajectory) {
        max_altitude = std::max(max_altitude, s.y);
        double speed = std::sqrt(s.vx*s.vx + s.vy*s.vy);
        max_speed = std::max(max_speed, speed);
        max_horizontal_distance = std::max(max_horizontal_distance, s.x);
    }
    
    std::cout << "Max altitude: " << max_altitude/1000.0 << " km" << std::endl;
    std::cout << "Max speed: " << max_speed << " m/s (" << max_speed*3.6 << " km/h)" << std::endl;
    std::cout << "Horizontal range: " << max_horizontal_distance/1000.0 << " km" << std::endl;
    std::cout << "Final velocity: " << std::sqrt(final_state.vx*final_state.vx + final_state.vy*final_state.vy) << " m/s" << std::endl;
    
    // Energy analysis
    double initial_energy = 0.5 * initial_mass * 0.0; // Started at rest
    double final_kinetic = 0.5 * final_state.m * (final_state.vx*final_state.vx + final_state.vy*final_state.vy);
    double final_potential = final_state.m * params.g0 * final_state.y;
    double final_energy = final_kinetic + final_potential;
    
    std::cout << "Final kinetic energy: " << final_kinetic/1e6 << " MJ" << std::endl;
    std::cout << "Final potential energy: " << final_potential/1e6 << " MJ" << std::endl;
    std::cout << "Total final energy: " << final_energy/1e6 << " MJ" << std::endl;
    
    // Test 2: Compare integrators
    std::cout << "\n=== Test 2: Integrator Comparison ===" << std::endl;
    
    // Shorter simulation for fair comparison
    auto short_control_func = [&](double t) -> AscentDynamics::Control {
        return AscentDynamics::Control(params.Tmax * 0.8, M_PI/2.0); // Constant vertical thrust
    };
    
    // RK4 with fixed step
    double dt = 0.1;
    int num_steps = 100; // 10 seconds
    std::vector<AscentDynamics::Control> controls(num_steps, short_control_func(0.0));
    
    auto start_rk4 = std::chrono::high_resolution_clock::now();
    auto traj_rk4 = ForwardIntegrator::integrate_rk4(rhs_func, state0, controls, params, dt);
    auto end_rk4 = std::chrono::high_resolution_clock::now();
    
    // RK45 adaptive
    auto start_rk45 = std::chrono::high_resolution_clock::now();
    auto traj_rk45 = ForwardIntegrator::integrate_rk45(
        rhs_func, state0, short_control_func, params, 0.0, 10.0, 1e-6, 1e-8, 0.1);
    auto end_rk45 = std::chrono::high_resolution_clock::now();
    
    auto duration_rk4 = std::chrono::duration_cast<std::chrono::microseconds>(end_rk4 - start_rk4);
    auto duration_rk45 = std::chrono::duration_cast<std::chrono::microseconds>(end_rk45 - start_rk45);
    
    std::cout << "RK4 (fixed step): " << traj_rk4.size() << " points, " 
              << duration_rk4.count() << " μs" << std::endl;
    std::cout << "RK45 (adaptive): " << traj_rk45.size() << " points, " 
              << duration_rk45.count() << " μs" << std::endl;
    
    // Compare final states
    const auto& final_rk4 = traj_rk4.back();
    const auto& final_rk45 = traj_rk45.back().second;
    
    double altitude_diff = std::abs(final_rk4.y - final_rk45.y);
    double velocity_diff = std::abs(std::sqrt(final_rk4.vx*final_rk4.vx + final_rk4.vy*final_rk4.vy) -
                                  std::sqrt(final_rk45.vx*final_rk45.vx + final_rk45.vy*final_rk45.vy));
    
    std::cout << "Altitude difference: " << altitude_diff << " m" << std::endl;
    std::cout << "Velocity difference: " << velocity_diff << " m/s" << std::endl;
    
    // Test 3: Energy conservation in free fall
    std::cout << "\n=== Test 3: Energy Conservation (Free Fall) ===" << std::endl;
    
    AscentDynamics::State ballistic_state = state0;
    ballistic_state.vy = 200.0; // Initial upward velocity
    ballistic_state.y = 1000.0; // Start at 1 km
    
    // No atmosphere for this test
    AscentDynamics::Params vacuum_params = params;
    vacuum_params.rho0 = 0.0;
    
    auto no_thrust_func = [](double t) -> AscentDynamics::Control {
        return AscentDynamics::Control(0.0, 0.0);
    };
    
    auto ballistic_traj = ForwardIntegrator::integrate_rk45(
        rhs_func, ballistic_state, no_thrust_func, vacuum_params,
        0.0, 40.0, 1e-8, 1e-10, 0.1
    );
    
    // Check energy conservation
    auto calc_energy = [&](const AscentDynamics::State& s) {
        double ke = 0.5 * s.m * (s.vx*s.vx + s.vy*s.vy);
        double pe = s.m * params.g0 * s.y;
        return ke + pe;
    };
    
    double initial_energy_ballistic = calc_energy(ballistic_traj[0].second);
    double max_energy_error = 0.0;
    
    for (const auto& [t, s] : ballistic_traj) {
        double energy = calc_energy(s);
        double error = std::abs(energy - initial_energy_ballistic) / initial_energy_ballistic;
        max_energy_error = std::max(max_energy_error, error);
    }
    
    std::cout << "Energy conservation error: " << max_energy_error * 100.0 << "%" << std::endl;
    std::cout << "Max altitude reached: " << std::max_element(ballistic_traj.begin(), ballistic_traj.end(),
        [](const auto& a, const auto& b) { return a.second.y < b.second.y; })->second.y/1000.0 << " km" << std::endl;
    
    // Save trajectory data for plotting
    save_trajectory_csv(trajectory, "rocket_trajectory.csv");
    
    // Validation summary
    std::cout << "\n=== Validation Summary ===" << std::endl;
    
    bool mass_reasonable = (mass_consumed > 0) && (final_state.m >= params.m_dry);
    bool altitude_reasonable = (max_altitude > 1000.0) && (max_altitude < 200000.0);
    bool speed_reasonable = (max_speed > 50.0) && (max_speed < 5000.0);
    bool energy_conserved = (max_energy_error < 0.01); // 1% error
    bool integrators_agree = (altitude_diff < 10.0) && (velocity_diff < 5.0);
    
    std::cout << "✓ Mass conservation: " << (mass_reasonable ? "PASS" : "FAIL") << std::endl;
    std::cout << "✓ Altitude reasonable: " << (altitude_reasonable ? "PASS" : "FAIL") << std::endl;
    std::cout << "✓ Speed reasonable: " << (speed_reasonable ? "PASS" : "FAIL") << std::endl;
    std::cout << "✓ Energy conservation: " << (energy_conserved ? "PASS" : "FAIL") << std::endl;
    std::cout << "✓ Integrator agreement: " << (integrators_agree ? "PASS" : "FAIL") << std::endl;
    
    bool all_tests_pass = mass_reasonable && altitude_reasonable && speed_reasonable && 
                         energy_conserved && integrators_agree;
    
    std::cout << "\nOverall validation: " << (all_tests_pass ? "PASS" : "FAIL") << std::endl;
    
    if (all_tests_pass) {
        std::cout << "🚀 Physics dynamics implementation validated successfully!" << std::endl;
    } else {
        std::cout << "❌ Some validation tests failed. Check implementation." << std::endl;
        return 1;
    }
    
    return 0;
}
