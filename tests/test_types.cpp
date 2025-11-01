#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "../src/physics/types.hpp"

using namespace rocket_physics;

class TypesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test data
        test_position = Vec3(1000.0, 2000.0, 3000.0);
        test_velocity = Vec3(100.0, 200.0, 300.0);
        test_quaternion = Quaterniond(0.7071, 0.0, 0.0, 0.7071); // 90° rotation around Z
        test_angular_velocity = Vec3(0.1, 0.2, 0.3);
        test_mass = 5000.0;
        
        test_thrust = 100000.0;
        test_thrust_direction = Vec3(0.0, 0.0, 1.0).normalized();
    }
    
    Vec3 test_position;
    Vec3 test_velocity;
    Quaterniond test_quaternion;
    Vec3 test_angular_velocity;
    double test_mass;
    
    double test_thrust;
    Vec3 test_thrust_direction;
};

// Test State struct
TEST_F(TypesTest, StateDefaultConstructor) {
    State state;
    
    EXPECT_EQ(state.r_i, Vec3::Zero());
    EXPECT_EQ(state.v_i, Vec3::Zero());
    EXPECT_EQ(state.q_bi.w(), 1.0);
    EXPECT_EQ(state.q_bi.x(), 0.0);
    EXPECT_EQ(state.q_bi.y(), 0.0);
    EXPECT_EQ(state.q_bi.z(), 0.0);
    EXPECT_EQ(state.w_b, Vec3::Zero());
    EXPECT_EQ(state.m, 0.0);
}

TEST_F(TypesTest, StateParameterConstructor) {
    State state(test_position, test_velocity, test_quaternion, test_angular_velocity, test_mass);
    
    EXPECT_EQ(state.r_i, test_position);
    EXPECT_EQ(state.v_i, test_velocity);
    EXPECT_EQ(state.q_bi.w(), test_quaternion.w());
    EXPECT_EQ(state.q_bi.x(), test_quaternion.x());
    EXPECT_EQ(state.q_bi.y(), test_quaternion.y());
    EXPECT_EQ(state.q_bi.z(), test_quaternion.z());
    EXPECT_EQ(state.w_b, test_angular_velocity);
    EXPECT_EQ(state.m, test_mass);
}

TEST_F(TypesTest, StateToVectorConversion) {
    State state(test_position, test_velocity, test_quaternion, test_angular_velocity, test_mass);
    Eigen::VectorXd state_vec = state.toVector();
    
    EXPECT_EQ(state_vec.size(), 14);
    EXPECT_EQ(state_vec.segment<3>(0), test_position);
    EXPECT_EQ(state_vec.segment<3>(3), test_velocity);
    EXPECT_EQ(state_vec.segment<4>(6), Eigen::Vector4d(test_quaternion.w(), test_quaternion.x(), 
                                                      test_quaternion.y(), test_quaternion.z()));
    EXPECT_EQ(state_vec.segment<3>(10), test_angular_velocity);
    EXPECT_EQ(state_vec(13), test_mass);
}

TEST_F(TypesTest, StateFromVectorConversion) {
    Eigen::VectorXd state_vec(14);
    state_vec.segment<3>(0) = test_position;
    state_vec.segment<3>(3) = test_velocity;
    state_vec.segment<4>(6) = Eigen::Vector4d(test_quaternion.w(), test_quaternion.x(), 
                                             test_quaternion.y(), test_quaternion.z());
    state_vec.segment<3>(10) = test_angular_velocity;
    state_vec(13) = test_mass;
    
    State state;
    state.fromVector(state_vec);
    
    EXPECT_EQ(state.r_i, test_position);
    EXPECT_EQ(state.v_i, test_velocity);
    EXPECT_NEAR(state.q_bi.w(), test_quaternion.w(), 1e-10);
    EXPECT_NEAR(state.q_bi.x(), test_quaternion.x(), 1e-10);
    EXPECT_NEAR(state.q_bi.y(), test_quaternion.y(), 1e-10);
    EXPECT_NEAR(state.q_bi.z(), test_quaternion.z(), 1e-10);
    EXPECT_EQ(state.w_b, test_angular_velocity);
    EXPECT_EQ(state.m, test_mass);
}

TEST_F(TypesTest, StateRoundTripConversion) {
    State original_state(test_position, test_velocity, test_quaternion, test_angular_velocity, test_mass);
    Eigen::VectorXd state_vec = original_state.toVector();
    State converted_state;
    converted_state.fromVector(state_vec);
    
    EXPECT_EQ(converted_state.r_i, original_state.r_i);
    EXPECT_EQ(converted_state.v_i, original_state.v_i);
    EXPECT_NEAR(converted_state.q_bi.w(), original_state.q_bi.w(), 1e-10);
    EXPECT_NEAR(converted_state.q_bi.x(), original_state.q_bi.x(), 1e-10);
    EXPECT_NEAR(converted_state.q_bi.y(), original_state.q_bi.y(), 1e-10);
    EXPECT_NEAR(converted_state.q_bi.z(), original_state.q_bi.z(), 1e-10);
    EXPECT_EQ(converted_state.w_b, original_state.w_b);
    EXPECT_EQ(converted_state.m, original_state.m);
}

// Test Control struct
TEST_F(TypesTest, ControlDefaultConstructor) {
    Control control;
    
    EXPECT_EQ(control.T, 0.0);
    EXPECT_EQ(control.uT_b, Vec3::Zero());
}

TEST_F(TypesTest, ControlParameterConstructor) {
    Control control(test_thrust, test_thrust_direction);
    
    EXPECT_EQ(control.T, test_thrust);
    EXPECT_EQ(control.uT_b, test_thrust_direction.normalized());
}

TEST_F(TypesTest, ControlToVectorConversion) {
    Control control(test_thrust, test_thrust_direction);
    Eigen::VectorXd control_vec = control.toVector();
    
    EXPECT_EQ(control_vec.size(), 4);
    EXPECT_EQ(control_vec(0), test_thrust);
    EXPECT_EQ(control_vec.segment<3>(1), test_thrust_direction.normalized());
}

TEST_F(TypesTest, ControlFromVectorConversion) {
    Eigen::VectorXd control_vec(4);
    control_vec(0) = test_thrust;
    control_vec.segment<3>(1) = test_thrust_direction;
    
    Control control;
    control.fromVector(control_vec);
    
    EXPECT_EQ(control.T, test_thrust);
    EXPECT_EQ(control.uT_b, test_thrust_direction.normalized());
}

// Test Phys struct
TEST_F(TypesTest, PhysDefaultConstructor) {
    Phys phys;
    
    EXPECT_EQ(phys.Cd, 0.3);
    EXPECT_EQ(phys.Cl, 0.0);
    EXPECT_EQ(phys.S_ref, 1.0);
    EXPECT_EQ(phys.I_b, Matrix3d::Identity());
    EXPECT_EQ(phys.r_cg, Vec3::Zero());
    EXPECT_EQ(phys.Isp, 300.0);
    EXPECT_EQ(phys.g0, 9.81);
    EXPECT_EQ(phys.rho0, 1.225);
    EXPECT_EQ(phys.h_scale, 8400.0);
}

// Test Limits struct
TEST_F(TypesTest, LimitsDefaultConstructor) {
    Limits limits;
    
    EXPECT_EQ(limits.T_max, 1000000.0);
    EXPECT_EQ(limits.m_dry, 1000.0);
    EXPECT_EQ(limits.q_max, 50000.0);
    EXPECT_EQ(limits.w_gimbal_max, 1.0);
    EXPECT_EQ(limits.alpha_max, 0.1);
    EXPECT_EQ(limits.n_max, 10.0);
}

// Test Diag struct
TEST_F(TypesTest, DiagDefaultConstructor) {
    Diag diag;
    
    EXPECT_EQ(diag.rho, 0.0);
    EXPECT_EQ(diag.q, 0.0);
    EXPECT_FALSE(diag.q_violation);
    EXPECT_FALSE(diag.m_underflow);
    EXPECT_EQ(diag.alpha, 0.0);
    EXPECT_EQ(diag.n, 0.0);
}

TEST_F(TypesTest, DiagReset) {
    Diag diag;
    diag.rho = 1.0;
    diag.q = 1000.0;
    diag.q_violation = true;
    diag.m_underflow = true;
    diag.alpha = 0.05;
    diag.n = 5.0;
    
    diag.reset();
    
    EXPECT_EQ(diag.rho, 0.0);
    EXPECT_EQ(diag.q, 0.0);
    EXPECT_FALSE(diag.q_violation);
    EXPECT_FALSE(diag.m_underflow);
    EXPECT_EQ(diag.alpha, 0.0);
    EXPECT_EQ(diag.n, 0.0);
}

// Test utility functions
TEST_F(TypesTest, StateDimension) {
    EXPECT_EQ(utils::stateDim(), 14);
}

TEST_F(TypesTest, ControlDimension) {
    EXPECT_EQ(utils::controlDim(), 4);
}

TEST_F(TypesTest, StateToVectorUtility) {
    State state(test_position, test_velocity, test_quaternion, test_angular_velocity, test_mass);
    Eigen::VectorXd state_vec = utils::stateToVector(state);
    
    EXPECT_EQ(state_vec.size(), 14);
    EXPECT_EQ(state_vec, state.toVector());
}

TEST_F(TypesTest, VectorToStateUtility) {
    Eigen::VectorXd state_vec(14);
    state_vec.segment<3>(0) = test_position;
    state_vec.segment<3>(3) = test_velocity;
    state_vec.segment<4>(6) = Eigen::Vector4d(test_quaternion.w(), test_quaternion.x(), 
                                             test_quaternion.y(), test_quaternion.z());
    state_vec.segment<3>(10) = test_angular_velocity;
    state_vec(13) = test_mass;
    
    State state = utils::vectorToState(state_vec);
    
    EXPECT_EQ(state.r_i, test_position);
    EXPECT_EQ(state.v_i, test_velocity);
    EXPECT_NEAR(state.q_bi.w(), test_quaternion.w(), 1e-10);
    EXPECT_NEAR(state.q_bi.x(), test_quaternion.x(), 1e-10);
    EXPECT_NEAR(state.q_bi.y(), test_quaternion.y(), 1e-10);
    EXPECT_NEAR(state.q_bi.z(), test_quaternion.z(), 1e-10);
    EXPECT_EQ(state.w_b, test_angular_velocity);
    EXPECT_EQ(state.m, test_mass);
}

TEST_F(TypesTest, QuaternionNormalization) {
    State state;
    state.q_bi = Quaterniond(2.0, 1.0, 0.5, 0.1); // Non-normalized quaternion
    utils::normalizeQuaternion(state);
    
    EXPECT_NEAR(state.q_bi.norm(), 1.0, 1e-10);
}

TEST_F(TypesTest, ValidStateCheck) {
    State valid_state(test_position, test_velocity, test_quaternion, test_angular_velocity, test_mass);
    EXPECT_TRUE(utils::isValidState(valid_state));
    
    State invalid_state;
    invalid_state.m = -1.0; // Negative mass
    EXPECT_FALSE(utils::isValidState(invalid_state));
}

TEST_F(TypesTest, ValidControlCheck) {
    Control valid_control(test_thrust, test_thrust_direction);
    EXPECT_TRUE(utils::isValidControl(valid_control));
    
    Control invalid_control;
    invalid_control.T = -1000.0; // Negative thrust
    EXPECT_FALSE(utils::isValidControl(invalid_control));
}

// Test error handling
TEST_F(TypesTest, StateFromVectorWrongSize) {
    Eigen::VectorXd wrong_size_vec(10);
    State state;
    
    EXPECT_THROW(state.fromVector(wrong_size_vec), std::invalid_argument);
}

TEST_F(TypesTest, ControlFromVectorWrongSize) {
    Eigen::VectorXd wrong_size_vec(2);
    Control control;
    
    EXPECT_THROW(control.fromVector(wrong_size_vec), std::invalid_argument);
}
