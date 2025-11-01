#pragma once

#include "types.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>

namespace rocket_physics {

/**
 * @brief Mathematical utility functions
 */
namespace math {
    
    /**
     * @brief Normalize quaternion to unit length
     * @param q Quaternion to normalize
     * @return Normalized quaternion
     */
    Quaterniond normalizeQuaternion(const Quaterniond& q);
    
    /**
     * @brief Convert quaternion to rotation matrix
     * @param q Quaternion
     * @return Rotation matrix
     */
    Matrix3d quaternionToRotationMatrix(const Quaterniond& q);
    
    /**
     * @brief Convert rotation matrix to quaternion
     * @param R Rotation matrix
     * @return Quaternion
     */
    Quaterniond rotationMatrixToQuaternion(const Matrix3d& R);
    
    /**
     * @brief Convert Euler angles to quaternion
     * @param roll Roll angle [rad]
     * @param pitch Pitch angle [rad]
     * @param yaw Yaw angle [rad]
     * @return Quaternion
     */
    Quaterniond eulerToQuaternion(double roll, double pitch, double yaw);
    
    /**
     * @brief Convert quaternion to Euler angles
     * @param q Quaternion
     * @return Vector of [roll, pitch, yaw] in radians
     */
    Vec3 quaternionToEuler(const Quaterniond& q);
    
    /**
     * @brief Compute quaternion derivative
     * @param q Quaternion
     * @param omega Angular velocity
     * @return Quaternion derivative
     */
    Quaterniond quaternionDerivative(const Quaterniond& q, const Vec3& omega);
    
    /**
     * @brief Compute angle between two vectors
     * @param v1 First vector
     * @param v2 Second vector
     * @return Angle in radians
     */
    double angleBetweenVectors(const Vec3& v1, const Vec3& v2);
    
    /**
     * @brief Compute cross product matrix
     * @param v Vector
     * @return Skew-symmetric matrix
     */
    Matrix3d crossProductMatrix(const Vec3& v);
    
    /**
     * @brief Clamp value to range
     * @param value Value to clamp
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Clamped value
     */
    double clamp(double value, double min_val, double max_val);
    
    /**
     * @brief Clamp vector to range
     * @param vec Vector to clamp
     * @param min_val Minimum value
     * @param max_val Maximum value
     * @return Clamped vector
     */
    Vec3 clampVector(const Vec3& vec, double min_val, double max_val);
    
    /**
     * @brief Linear interpolation
     * @param t Interpolation parameter [0, 1]
     * @param start Start value
     * @param end End value
     * @return Interpolated value
     */
    double lerp(double t, double start, double end);
    
    /**
     * @brief Vector linear interpolation
     * @param t Interpolation parameter [0, 1]
     * @param start Start vector
     * @param end End vector
     * @return Interpolated vector
     */
    Vec3 lerpVector(double t, const Vec3& start, const Vec3& end);
    
    /**
     * @brief Quaternion spherical linear interpolation
     * @param t Interpolation parameter [0, 1]
     * @param start Start quaternion
     * @param end End quaternion
     * @return Interpolated quaternion
     */
    Quaterniond slerp(double t, const Quaterniond& start, const Quaterniond& end);
}

/**
 * @brief Coordinate transformation utilities
 */
namespace coord {
    
    /**
     * @brief Transform vector from body to inertial frame
     * @param body_vector Vector in body frame
     * @param q_bi Quaternion from body to inertial
     * @return Vector in inertial frame
     */
    Vec3 bodyToInertial(const Vec3& body_vector, const Quaterniond& q_bi);
    
    /**
     * @brief Transform vector from inertial to body frame
     * @param inertial_vector Vector in inertial frame
     * @param q_bi Quaternion from body to inertial
     * @return Vector in body frame
     */
    Vec3 inertialToBody(const Vec3& inertial_vector, const Quaterniond& q_bi);
    
    /**
     * @brief Transform angular velocity from body to inertial frame
     * @param omega_body Angular velocity in body frame
     * @param q_bi Quaternion from body to inertial
     * @return Angular velocity in inertial frame
     */
    Vec3 angularVelocityBodyToInertial(const Vec3& omega_body, const Quaterniond& q_bi);
    
    /**
     * @brief Transform angular velocity from inertial to body frame
     * @param omega_inertial Angular velocity in inertial frame
     * @param q_bi Quaternion from body to inertial
     * @return Angular velocity in body frame
     */
    Vec3 angularVelocityInertialToBody(const Vec3& omega_inertial, const Quaterniond& q_bi);
}

/**
 * @brief File I/O utilities
 */
namespace io {
    
    /**
     * @brief Save state trajectory to file
     * @param states Vector of states
     * @param times Vector of times
     * @param filename Output filename
     */
    void saveTrajectory(const std::vector<State>& states, const std::vector<double>& times, const std::string& filename);
    
    /**
     * @brief Load state trajectory from file
     * @param filename Input filename
     * @return Pair of (states, times)
     */
    std::pair<std::vector<State>, std::vector<double>> loadTrajectory(const std::string& filename);
    
    /**
     * @brief Save control trajectory to file
     * @param controls Vector of controls
     * @param times Vector of times
     * @param filename Output filename
     */
    void saveControlTrajectory(const std::vector<Control>& controls, const std::vector<double>& times, const std::string& filename);
    
    /**
     * @brief Load control trajectory from file
     * @param filename Input filename
     * @return Pair of (controls, times)
     */
    std::pair<std::vector<Control>, std::vector<double>> loadControlTrajectory(const std::string& filename);
    
    /**
     * @brief Save diagnostic data to file
     * @param diagnostics Vector of diagnostic data
     * @param times Vector of times
     * @param filename Output filename
     */
    void saveDiagnostics(const std::vector<Diag>& diagnostics, const std::vector<double>& times, const std::string& filename);
    
    /**
     * @brief Load diagnostic data from file
     * @param filename Input filename
     * @return Pair of (diagnostics, times)
     */
    std::pair<std::vector<Diag>, std::vector<double>> loadDiagnostics(const std::string& filename);
}

/**
 * @brief Logging utilities
 */
namespace logging {
    
    /**
     * @brief Log level enumeration
     */
    enum class LogLevel {
        DEBUG = 0,
        INFO = 1,
        WARNING = 2,
        ERROR = 3
    };
    
    /**
     * @brief Logger class
     */
    class Logger {
    public:
        /**
         * @brief Constructor
         * @param filename Log filename
         * @param level Log level
         */
        Logger(const std::string& filename, LogLevel level = LogLevel::INFO);
        
        /**
         * @brief Destructor
         */
        ~Logger();
        
        /**
         * @brief Log message
         * @param level Log level
         * @param message Message to log
         */
        void log(LogLevel level, const std::string& message);
        
        /**
         * @brief Log debug message
         * @param message Message to log
         */
        void debug(const std::string& message);
        
        /**
         * @brief Log info message
         * @param message Message to log
         */
        void info(const std::string& message);
        
        /**
         * @brief Log warning message
         * @param message Message to log
         */
        void warning(const std::string& message);
        
        /**
         * @brief Log error message
         * @param message Message to log
         */
        void error(const std::string& message);
        
        /**
         * @brief Set log level
         * @param level Log level
         */
        void setLevel(LogLevel level);
        
        /**
         * @brief Get log level
         * @return Log level
         */
        LogLevel getLevel() const { return level_; }

    private:
        std::ofstream file_;
        LogLevel level_;
        
        std::string levelToString(LogLevel level) const;
        std::string getTimestamp() const;
    };
}

/**
 * @brief Validation utilities
 */
namespace validation {
    
    /**
     * @brief Validate state
     * @param state State to validate
     * @return True if state is valid
     */
    bool validateState(const State& state);
    
    /**
     * @brief Validate control
     * @param control Control to validate
     * @return True if control is valid
     */
    bool validateControl(const Control& control);
    
    /**
     * @brief Validate physical parameters
     * @param phys Physical parameters to validate
     * @return True if parameters are valid
     */
    bool validatePhys(const Phys& phys);
    
    /**
     * @brief Validate limits
     * @param limits Limits to validate
     * @return True if limits are valid
     */
    bool validateLimits(const Limits& limits);
    
    /**
     * @brief Get validation errors
     * @param state State to validate
     * @return Vector of error messages
     */
    std::vector<std::string> getStateValidationErrors(const State& state);
    
    /**
     * @brief Get validation errors
     * @param control Control to validate
     * @return Vector of error messages
     */
    std::vector<std::string> getControlValidationErrors(const Control& control);
}

/**
 * @brief Unit conversion utilities
 */
namespace units {
    
    /**
     * @brief Convert degrees to radians
     * @param degrees Angle in degrees
     * @return Angle in radians
     */
    double degToRad(double degrees);
    
    /**
     * @brief Convert radians to degrees
     * @param radians Angle in radians
     * @return Angle in degrees
     */
    double radToDeg(double radians);
    
    /**
     * @brief Convert feet to meters
     * @param feet Distance in feet
     * @return Distance in meters
     */
    double feetToMeters(double feet);
    
    /**
     * @brief Convert meters to feet
     * @param meters Distance in meters
     * @return Distance in feet
     */
    double metersToFeet(double meters);
    
    /**
     * @brief Convert pounds to kilograms
     * @param pounds Mass in pounds
     * @return Mass in kilograms
     */
    double poundsToKilograms(double pounds);
    
    /**
     * @brief Convert kilograms to pounds
     * @param kilograms Mass in kilograms
     * @return Mass in pounds
     */
    double kilogramsToPounds(double kilograms);
}

} // namespace rocket_physics
