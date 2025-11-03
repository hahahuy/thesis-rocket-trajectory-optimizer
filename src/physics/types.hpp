#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <functional>
#include <vector>

namespace rocket_physics {

// Forward declarations
using Vec3 = Eigen::Vector3d;
using Quaterniond = Eigen::Quaterniond;
using Matrix3d = Eigen::Matrix3d;

/**
 * @brief State vector indices for standardization
 */
enum class StateIndex {
    X = 0, Y, Z,                    // Position (0-2)
    VX, VY, VZ,                     // Velocity (3-5)
    Q0, Q1, Q2, Q3,                 // Quaternion (6-9)
    WX, WY, WZ,                     // Angular velocity (10-12)
    M                               // Mass (13)
};

/**
 * @brief Control vector indices for standardization
 */
enum class ControlIndex {
    T = 0,                          // Thrust magnitude (0)
    THETA,                          // Gimbal angle theta (1) - or uT_x
    PHI,                            // Gimbal angle phi (2) - or uT_y
    UT_Z,                           // Thrust direction z component (3)
    DELTA                           // Control surface deflection (4)
};

/**
 * @brief 6-DOF rocket state representation
 * 
 * Contains 14 variables:
 * - r_i(3): Position in inertial frame [m]
 * - v_i(3): Velocity in inertial frame [m/s]
 * - q_bi(4): Quaternion from body to inertial frame
 * - ω_b(3): Angular velocity in body frame [rad/s]
 * - m(1): Mass [kg]
 */
struct State {
    Vec3 r_i;           // Position in inertial frame [m]
    Vec3 v_i;           // Velocity in inertial frame [m/s]
    Quaterniond q_bi;   // Quaternion from body to inertial frame
    Vec3 w_b;           // Angular velocity in body frame [rad/s]
    double m;           // Mass [kg]
    
    // Default constructor
    State() : r_i(Vec3::Zero()), v_i(Vec3::Zero()), q_bi(Quaterniond::Identity()), 
              w_b(Vec3::Zero()), m(0.0) {}
    
    // Constructor with parameters
    State(const Vec3& r, const Vec3& v, const Quaterniond& q, const Vec3& w, double mass)
        : r_i(r), v_i(v), q_bi(q), w_b(w), m(mass) {}
    
    // Convert to Eigen vector (14 elements)
    Eigen::VectorXd toVector() const {
        Eigen::VectorXd state_vec(14);
        state_vec.segment<3>(0) = r_i;
        state_vec.segment<3>(3) = v_i;
        state_vec.segment<4>(6) = Eigen::Vector4d(q_bi.w(), q_bi.x(), q_bi.y(), q_bi.z());
        state_vec.segment<3>(10) = w_b;
        state_vec(13) = m;
        return state_vec;
    }
    
    // Initialize from Eigen vector
    void fromVector(const Eigen::VectorXd& state_vec) {
        if (state_vec.size() != 14) {
            throw std::invalid_argument("State vector must have 14 elements");
        }
        r_i = state_vec.segment<3>(0);
        v_i = state_vec.segment<3>(3);
        q_bi = Quaterniond(state_vec(6), state_vec(7), state_vec(8), state_vec(9));
        w_b = state_vec.segment<3>(10);
        m = state_vec(13);
    }
};

/**
 * @brief Control input representation
 * 
 * Contains thrust magnitude and gimbal direction
 */
struct Control {
    double T;        // Thrust magnitude [N]
    Vec3 uT_b;       // Thrust direction unit vector in body frame
    double delta;    // Optional control surface deflection (rad)
    
    // Default constructor
    Control() : T(0.0), uT_b(Vec3::Zero()), delta(0.0) {}
    
    // Constructor with parameters
    Control(double thrust, const Vec3& direction) : T(thrust), uT_b(direction.normalized()), delta(0.0) {}
    
    // Constructor with all parameters
    Control(double thrust, const Vec3& direction, double control_surface) 
        : T(thrust), uT_b(direction.normalized()), delta(control_surface) {}
    
    // Convert to Eigen vector (5 elements: T, uT_b[0], uT_b[1], uT_b[2], delta)
    Eigen::VectorXd toVector() const {
        Eigen::VectorXd control_vec(5);
        control_vec(0) = T;
        control_vec.segment<3>(1) = uT_b;
        control_vec(4) = delta;
        return control_vec;
    }
    
    // Initialize from Eigen vector
    void fromVector(const Eigen::VectorXd& control_vec) {
        if (control_vec.size() < 4) {
            throw std::invalid_argument("Control vector must have at least 4 elements");
        }
        T = control_vec(0);
        uT_b = control_vec.segment<3>(1).normalized();
        if (control_vec.size() >= 5) {
            delta = control_vec(4);
        } else {
            delta = 0.0;
        }
    }
};

/**
 * @brief Physical parameters for rocket simulation
 * 
 * Contains aerodynamic, inertia, and propulsion parameters
 */
struct Phys {
    // Aerodynamic parameters
    double Cd;              // Drag coefficient
    double Cl;              // Lift coefficient
    double S_ref;           // Reference area [m²]
    
    // Inertia parameters
    Matrix3d I_b;           // Inertia tensor in body frame [kg⋅m²]
    Vec3 r_cg;              // Center of gravity offset from body origin [m]
    
    // Propulsion parameters
    double Isp;             // Specific impulse [s]
    double g0;              // Standard gravity [m/s²]
    
    // Atmospheric parameters
    double rho0;            // Sea level density [kg/m³]
    double h_scale;         // Atmospheric scale height [m]
    
    // Aerodynamic parameters (for tail/wing effects)
    double CL_alpha;        // Lift curve slope [1/rad]
    double Cm_alpha;        // Pitch moment coefficient [1/rad]
    double C_delta;         // Control surface authority [1/rad]
    double l_ref;           // Reference length for moments [m]
    double delta_limit;     // Maximum control surface deflection [rad]
    
    // Default constructor
    Phys() : Cd(0.3), Cl(0.0), S_ref(1.0), I_b(Matrix3d::Identity()), 
             r_cg(Vec3::Zero()), Isp(300.0), g0(9.81), rho0(1.225), h_scale(8400.0),
             CL_alpha(3.5), Cm_alpha(-0.8), C_delta(0.05), l_ref(1.2), delta_limit(0.1745) {}
};

/**
 * @brief Operational limits and constraints
 * 
 * Contains maximum values and safety limits
 */
struct Limits {
    double T_max;           // Maximum thrust [N]
    double m_dry;           // Dry mass [kg]
    double q_max;           // Maximum dynamic pressure [Pa]
    double w_gimbal_max;    // Maximum gimbal rate [rad/s]
    double alpha_max;       // Maximum angle of attack [rad]
    double n_max;           // Maximum load factor [g]
    
    // Default constructor
    Limits() : T_max(1000000.0), m_dry(1000.0), q_max(50000.0), 
               w_gimbal_max(1.0), alpha_max(0.1), n_max(10.0) {}
};

/**
 * @brief Function callbacks for external forces
 */
using GravityFunc = std::function<Vec3(const Vec3&)>;
using WindFunc = std::function<Vec3(const Vec3&, double)>;

/**
 * @brief Diagnostic information for monitoring
 * 
 * Contains atmospheric and constraint violation information
 */
struct Diag {
    double rho;             // Atmospheric density [kg/m³]
    double q;               // Dynamic pressure [Pa]
    bool q_violation;       // Dynamic pressure constraint violation
    bool n_violation;       // Normal load constraint violation
    bool m_underflow;       // Mass underflow detection
    double alpha;           // Angle of attack [rad]
    double n;               // Load factor [g]
    
    // Default constructor
    Diag() : rho(0.0), q(0.0), q_violation(false), n_violation(false), m_underflow(false), 
             alpha(0.0), n(0.0) {}
    
    // Reset all values
    void reset() {
        rho = 0.0;
        q = 0.0;
        q_violation = false;
        n_violation = false;
        m_underflow = false;
        alpha = 0.0;
        n = 0.0;
    }
};

/**
 * @brief Utility functions for state manipulation
 */
namespace utils {
    
    /**
     * @brief Convert state to vector for numerical integration
     */
    inline Eigen::VectorXd stateToVector(const State& state) {
        return state.toVector();
    }
    
    /**
     * @brief Convert vector back to state
     */
    inline State vectorToState(const Eigen::VectorXd& vec) {
        State state;
        state.fromVector(vec);
        return state;
    }
    
    /**
     * @brief Get state dimension
     */
    inline constexpr int stateDim() { return 14; }
    
    /**
     * @brief Get control dimension
     */
    inline constexpr int controlDim() { return 4; }
    
    /**
     * @brief Normalize quaternion to ensure unit length
     */
    inline void normalizeQuaternion(State& state) {
        state.q_bi.normalize();
    }
    
    /**
     * @brief Check if state is valid
     */
    inline bool isValidState(const State& state) {
        return state.m > 0.0 && 
               std::abs(state.q_bi.norm() - 1.0) < 1e-5 &&
               !std::isnan(state.r_i.norm()) &&
               !std::isnan(state.v_i.norm()) &&
               !std::isnan(state.w_b.norm());
    }
    
    /**
     * @brief Check if control is valid
     */
    inline bool isValidControl(const Control& control) {
        return control.T >= 0.0 && 
               std::abs(control.uT_b.norm() - 1.0) < 1e-6 &&
               !std::isnan(control.T) &&
               !std::isnan(control.uT_b.norm());
    }
}

} // namespace rocket_physics
