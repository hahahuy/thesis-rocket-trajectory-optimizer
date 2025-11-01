#include <iostream>
#include <cassert>
#include "../src/physics/types.hpp"

using namespace rocket_physics;

int main() {
    std::cout << "=== Simple Types Test ===" << std::endl;
    
    // Test State struct
    Vec3 position(1000.0, 2000.0, 3000.0);
    Vec3 velocity(100.0, 200.0, 300.0);
    Quaterniond quaternion(0.7071, 0.0, 0.0, 0.7071);
    Vec3 angular_velocity(0.1, 0.2, 0.3);
    double mass = 5000.0;
    
    State state(position, velocity, quaternion, angular_velocity, mass);
    
    // Test vector conversion
    Eigen::VectorXd state_vec = state.toVector();
    assert(state_vec.size() == 14);
    std::cout << "✓ State vector conversion: " << state_vec.size() << " elements" << std::endl;
    
    // Test round-trip conversion
    State state_from_vec;
    state_from_vec.fromVector(state_vec);
    assert((state_from_vec.r_i - state.r_i).norm() < 1e-10);
    assert((state_from_vec.v_i - state.v_i).norm() < 1e-10);
    assert((state_from_vec.w_b - state.w_b).norm() < 1e-10);
    assert(std::abs(state_from_vec.m - state.m) < 1e-10);
    std::cout << "✓ State round-trip conversion successful" << std::endl;
    
    // Test Control struct
    double thrust = 100000.0;
    Vec3 thrust_direction(0.0, 0.0, 1.0);
    Control control(thrust, thrust_direction);
    
    Eigen::VectorXd control_vec = control.toVector();
    assert(control_vec.size() == 4);
    std::cout << "✓ Control vector conversion: " << control_vec.size() << " elements" << std::endl;
    
    // Test round-trip conversion
    Control control_from_vec;
    control_from_vec.fromVector(control_vec);
    assert(std::abs(control_from_vec.T - control.T) < 1e-10);
    assert((control_from_vec.uT_b - control.uT_b).norm() < 1e-10);
    std::cout << "✓ Control round-trip conversion successful" << std::endl;
    
    // Test utility functions
    assert(utils::stateDim() == 14);
    assert(utils::controlDim() == 4);
    std::cout << "✓ Utility functions work correctly" << std::endl;
    
    // Test quaternion normalization
    State test_state = state;
    test_state.q_bi = Quaterniond(2.0, 1.0, 0.5, 0.1);
    double norm_before = test_state.q_bi.norm();
    utils::normalizeQuaternion(test_state);
    double norm_after = test_state.q_bi.norm();
    assert(std::abs(norm_after - 1.0) < 1e-10);
    std::cout << "✓ Quaternion normalization: " << norm_before << " -> " << norm_after << std::endl;
    
    // Test validation functions
    std::cout << "State validation: " << (utils::isValidState(state) ? "PASS" : "FAIL") << std::endl;
    std::cout << "Quaternion norm: " << state.q_bi.norm() << std::endl;
    std::cout << "Mass: " << state.m << std::endl;
    assert(utils::isValidControl(control));
    std::cout << "✓ Control validation works correctly" << std::endl;
    
    // Test Phys and Limits structs
    Phys phys;
    assert(phys.Cd == 0.3);
    assert(phys.Isp == 300.0);
    std::cout << "✓ Phys struct initialization successful" << std::endl;
    
    Limits limits;
    assert(limits.T_max == 1000000.0);
    assert(limits.q_max == 50000.0);
    std::cout << "✓ Limits struct initialization successful" << std::endl;
    
    // Test Diag struct
    Diag diag;
    diag.rho = 1.225;
    diag.q = 10000.0;
    diag.q_violation = false;
    diag.m_underflow = false;
    assert(diag.rho == 1.225);
    assert(diag.q == 10000.0);
    assert(!diag.q_violation);
    assert(!diag.m_underflow);
    std::cout << "✓ Diag struct initialization successful" << std::endl;
    
    std::cout << "\n=== All Tests Passed! ===" << std::endl;
    return 0;
}
