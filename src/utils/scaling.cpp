#include "scaling.hpp"
#include <fstream>
#include <iostream>

namespace rocket_physics {
namespace scaling {

Eigen::VectorXd nondimensionalize(const State& state, const Scales& scales) {
    Eigen::VectorXd state_nd(14);
    
    // Position: divide by length scale
    state_nd.segment<3>(0) = state.r_i / scales.L_ref;
    
    // Velocity: divide by velocity scale
    state_nd.segment<3>(3) = state.v_i / scales.V_ref;
    
    // Quaternion: already normalized, copy as-is
    state_nd(6) = state.q_bi.w();
    state_nd(7) = state.q_bi.x();
    state_nd(8) = state.q_bi.y();
    state_nd(9) = state.q_bi.z();
    
    // Angular velocity: divide by (V_ref / L_ref) = angular velocity scale
    double omega_scale = scales.V_ref / scales.L_ref;
    state_nd.segment<3>(10) = state.w_b / omega_scale;
    
    // Mass: divide by mass scale
    state_nd(13) = state.m / scales.M_ref;
    
    return state_nd;
}

State dimensionalize(const Eigen::VectorXd& state_nd, const Scales& scales) {
    State state;
    
    // Position
    state.r_i = state_nd.segment<3>(0) * scales.L_ref;
    
    // Velocity
    state.v_i = state_nd.segment<3>(3) * scales.V_ref;
    
    // Quaternion
    state.q_bi = Quaterniond(state_nd(6), state_nd(7), state_nd(8), state_nd(9));
    state.q_bi.normalize();
    
    // Angular velocity
    double omega_scale = scales.V_ref / scales.L_ref;
    state.w_b = state_nd.segment<3>(10) * omega_scale;
    
    // Mass
    state.m = state_nd(13) * scales.M_ref;
    
    return state;
}

Eigen::VectorXd nondimensionalize(const Control& control, const Scales& scales) {
    Eigen::VectorXd control_nd(5);
    
    // Thrust: divide by force scale
    control_nd(0) = control.T / scales.F_ref;
    
    // Thrust direction: already unit vector, copy as-is
    control_nd.segment<3>(1) = control.uT_b;
    
    // Control surface deflection: divide by rad (already dimensionless, but scale by reference)
    control_nd(4) = control.delta; // Keep in radians
    
    return control_nd;
}

Control dimensionalizeControl(const Eigen::VectorXd& control_nd, const Scales& scales) {
    Control control;
    
    // Thrust
    control.T = control_nd(0) * scales.F_ref;
    
    // Thrust direction
    control.uT_b = control_nd.segment<3>(1).normalized();
    
    // Control surface deflection
    control.delta = control_nd(4);
    
    return control;
}

bool checkScaling(const Eigen::VectorXd& state_nd) {
    for (int i = 0; i < state_nd.size(); ++i) {
        double val = std::abs(state_nd(i));
        if (val > 10.0 || (val < 0.1 && i != 6 && i != 7 && i != 8 && i != 9)) {
            // Quaternion components can be small, ignore them
            if (i >= 6 && i <= 9) continue;
            return false;
        }
    }
    return true;
}

Scales loadScales(const std::string& filename) {
    // For now, return default scales
    // TODO: Parse YAML file if yaml-cpp is available
    Scales scales;
    scales.L_ref = 1e4;
    scales.V_ref = 1e3;
    scales.M_ref = 50.0;
    scales.F_ref = 5e3;
    scales.T_ref = 50.0;
    scales.Q_ref = 1e4;
    return scales;
}

} // namespace scaling
} // namespace rocket_physics
