#include <iostream>
#include <iomanip>
#include "../src/physics/types.hpp"

using namespace rocket_physics;

int main() {
    std::cout << "=== Rocket Physics Types Demo ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    
    // Create a sample state
    Vec3 position(1000.0, 2000.0, 3000.0);
    Vec3 velocity(100.0, 200.0, 300.0);
    Quaterniond quaternion(0.7071, 0.0, 0.0, 0.7071); // 90° rotation around Z
    Vec3 angular_velocity(0.1, 0.2, 0.3);
    double mass = 5000.0;
    
    State state(position, velocity, quaternion, angular_velocity, mass);
    
    std::cout << "\n--- State Information ---" << std::endl;
    std::cout << "Position [m]: (" << state.r_i.transpose() << ")" << std::endl;
    std::cout << "Velocity [m/s]: (" << state.v_i.transpose() << ")" << std::endl;
    std::cout << "Quaternion (w,x,y,z): (" << state.q_bi.w() << ", " << state.q_bi.x() 
              << ", " << state.q_bi.y() << ", " << state.q_bi.z() << ")" << std::endl;
    std::cout << "Angular velocity [rad/s]: (" << state.w_b.transpose() << ")" << std::endl;
    std::cout << "Mass [kg]: " << state.m << std::endl;
    
    // Convert to vector and back
    Eigen::VectorXd state_vec = state.toVector();
    std::cout << "\n--- State Vector (14 elements) ---" << std::endl;
    std::cout << "State vector: [" << state_vec.transpose() << "]" << std::endl;
    
    State state_from_vec;
    state_from_vec.fromVector(state_vec);
    std::cout << "\n--- State from Vector ---" << std::endl;
    std::cout << "Position [m]: (" << state_from_vec.r_i.transpose() << ")" << std::endl;
    std::cout << "Quaternion norm: " << state_from_vec.q_bi.norm() << std::endl;
    
    // Create a sample control
    double thrust = 100000.0;
    Vec3 thrust_direction(0.0, 0.0, 1.0);
    Control control(thrust, thrust_direction);
    
    std::cout << "\n--- Control Information ---" << std::endl;
    std::cout << "Thrust [N]: " << control.T << std::endl;
    std::cout << "Thrust direction: (" << control.uT_b.transpose() << ")" << std::endl;
    
    Eigen::VectorXd control_vec = control.toVector();
    std::cout << "\n--- Control Vector (4 elements) ---" << std::endl;
    std::cout << "Control vector: [" << control_vec.transpose() << "]" << std::endl;
    
    // Create physical parameters
    Phys phys;
    std::cout << "\n--- Physical Parameters ---" << std::endl;
    std::cout << "Drag coefficient: " << phys.Cd << std::endl;
    std::cout << "Reference area [m²]: " << phys.S_ref << std::endl;
    std::cout << "Specific impulse [s]: " << phys.Isp << std::endl;
    std::cout << "Inertia tensor [kg⋅m²]:\n" << phys.I_b << std::endl;
    
    // Create limits
    Limits limits;
    std::cout << "\n--- Operational Limits ---" << std::endl;
    std::cout << "Max thrust [N]: " << limits.T_max << std::endl;
    std::cout << "Dry mass [kg]: " << limits.m_dry << std::endl;
    std::cout << "Max dynamic pressure [Pa]: " << limits.q_max << std::endl;
    std::cout << "Max gimbal rate [rad/s]: " << limits.w_gimbal_max << std::endl;
    
    // Create diagnostic information
    Diag diag;
    diag.rho = 1.225;
    diag.q = 10000.0;
    diag.q_violation = false;
    diag.m_underflow = false;
    diag.alpha = 0.05;
    diag.n = 3.0;
    
    std::cout << "\n--- Diagnostic Information ---" << std::endl;
    std::cout << "Atmospheric density [kg/m³]: " << diag.rho << std::endl;
    std::cout << "Dynamic pressure [Pa]: " << diag.q << std::endl;
    std::cout << "Q violation: " << (diag.q_violation ? "Yes" : "No") << std::endl;
    std::cout << "Mass underflow: " << (diag.m_underflow ? "Yes" : "No") << std::endl;
    std::cout << "Angle of attack [rad]: " << diag.alpha << std::endl;
    std::cout << "Load factor [g]: " << diag.n << std::endl;
    
    // Test utility functions
    std::cout << "\n--- Utility Functions ---" << std::endl;
    std::cout << "State dimension: " << utils::stateDim() << std::endl;
    std::cout << "Control dimension: " << utils::controlDim() << std::endl;
    std::cout << "State is valid: " << (utils::isValidState(state) ? "Yes" : "No") << std::endl;
    std::cout << "Control is valid: " << (utils::isValidControl(control) ? "Yes" : "No") << std::endl;
    
    // Test quaternion normalization
    State test_state = state;
    test_state.q_bi = Quaterniond(2.0, 1.0, 0.5, 0.1); // Non-normalized
    std::cout << "\n--- Quaternion Normalization ---" << std::endl;
    std::cout << "Before normalization: " << test_state.q_bi.norm() << std::endl;
    utils::normalizeQuaternion(test_state);
    std::cout << "After normalization: " << test_state.q_bi.norm() << std::endl;
    
    std::cout << "\n=== Demo Complete ===" << std::endl;
    
    return 0;
}
