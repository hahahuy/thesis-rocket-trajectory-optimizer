#pragma once

#include "../physics/types.hpp"
#include <Eigen/Dense>

namespace rocket_physics {
namespace scaling {

/**
 * @brief Scaling factors for non-dimensionalization
 */
struct Scales {
    double L_ref = 1e4;      // Length scale [m]
    double V_ref = 1e3;      // Velocity scale [m/s]
    double M_ref = 50.0;     // Mass scale [kg]
    double F_ref = 5e3;      // Force scale [N]
    double T_ref = 50.0;     // Time scale [s]
    double Q_ref = 1e4;      // Dynamic pressure scale [Pa]
    
    Scales() = default;
    Scales(double L, double V, double M, double F, double T, double Q)
        : L_ref(L), V_ref(V), M_ref(M), F_ref(F), T_ref(T), Q_ref(Q) {}
};

/**
 * @brief Non-dimensionalize state
 * @param state Dimensional state
 * @param scales Scaling factors
 * @return Non-dimensional state vector
 */
Eigen::VectorXd nondimensionalize(const State& state, const Scales& scales);

/**
 * @brief Dimensionalize state
 * @param state_nd Non-dimensional state vector
 * @param scales Scaling factors
 * @return Dimensional state
 */
State dimensionalize(const Eigen::VectorXd& state_nd, const Scales& scales);

/**
 * @brief Non-dimensionalize control
 * @param control Dimensional control
 * @param scales Scaling factors
 * @return Non-dimensional control vector
 */
Eigen::VectorXd nondimensionalize(const Control& control, const Scales& scales);

/**
 * @brief Dimensionalize control
 * @param control_nd Non-dimensional control vector
 * @param scales Scaling factors
 * @return Dimensional control
 */
Control dimensionalizeControl(const Eigen::VectorXd& control_nd, const Scales& scales);

/**
 * @brief Check if scaled values are near O(1)
 * @param state_nd Non-dimensional state
 * @return True if all values are between 0.1 and 10.0
 */
bool checkScaling(const Eigen::VectorXd& state_nd);

/**
 * @brief Load scales from YAML configuration
 * @param filename Path to scales.yaml
 * @return Scales structure
 */
Scales loadScales(const std::string& filename = "configs/scales.yaml");

} // namespace scaling
} // namespace rocket_physics
