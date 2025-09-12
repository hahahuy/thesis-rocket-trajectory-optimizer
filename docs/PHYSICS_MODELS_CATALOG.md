# Physics Models Catalog for Rocket Trajectory Optimization

This document provides a comprehensive overview of all physics models that can be implemented in the rocket trajectory optimization project. Each model is marked with its current implementation status and includes detailed code implementation notes.

## Legend
- ✅ **Implemented** - Currently available in the codebase
- 🔄 **Partially Implemented** - Basic version exists, can be enhanced
- ❌ **Not Implemented** - Not currently available
- 🔥 **High Priority** - Recommended for immediate implementation
- ⚠️ **Complex** - Requires significant implementation effort

---

## 1. Dynamics and Kinematics Models

### 1.1 Basic Dynamics ✅
**Status**: Fully implemented in `src/physics/dynamics.cpp`

**Description**: 3-DOF planar rocket dynamics with position, velocity, and mass state variables.

**Current Implementation**:
```cpp
struct State {
    double x, y;      // Position [m]
    double vx, vy;    // Velocity [m/s]
    double m;         // Mass [kg]
};

State rhs(const State& s, const Control& u, const Params& p) {
    // Implemented: dx/dt = vx, dy/dt = vy
    // Implemented: dvx/dt, dvy/dt from forces
    // Implemented: dm/dt from thrust
}
```

**Code Location**: `src/physics/dynamics.hpp` lines 27-51, `src/physics/dynamics.cpp` lines 8-78

### 1.2 6-DOF Dynamics ❌🔥
**Status**: Not implemented

**Description**: Full 6 degrees of freedom including attitude dynamics (pitch, yaw, roll).

**Implementation Requirements**:
```cpp
struct State6DOF {
    double x, y, z;           // Position [m]
    double vx, vy, vz;        // Velocity [m/s]
    double phi, theta, psi;   // Euler angles [rad]
    double p, q, r;           // Angular rates [rad/s]
    double m;                 // Mass [kg]
};

struct Control6DOF {
    double Tx, Ty, Tz;       // Thrust components [N]
    double Mx, My, Mz;       // Moment components [N⋅m]
};

// New files needed:
// src/physics/dynamics_6dof.hpp
// src/physics/dynamics_6dof.cpp
// src/physics/quaternions.hpp (for attitude representation)
```

**Code Impact**:
- New RHS function with attitude dynamics
- Quaternion mathematics for attitude
- Rotation matrices for force/moment transformations
- Extended integrator support

### 1.3 Flexible Body Dynamics ❌⚠️
**Status**: Not implemented

**Description**: Models structural flexibility using modal analysis approach.

**Implementation Requirements**:
```cpp
struct FlexibleState {
    State rigid_body;
    std::vector<double> modal_coords;     // Generalized coordinates
    std::vector<double> modal_velocities; // Generalized velocities
};

struct StructuralParams {
    std::vector<double> natural_frequencies; // Hz
    std::vector<double> damping_ratios;      // -
    std::vector<Eigen::Vector3d> mode_shapes; // Mode shape at key points
    double structural_damping = 0.02;        // Structural damping coefficient
};

// Implementation in new file:
// src/physics/flexible_dynamics.hpp
```

**Code Impact**:
- Modal analysis mathematics
- Coupling between rigid body and flexible modes
- Additional DOFs in state vector
- Aerodynamic force distribution along vehicle

---

## 2. Atmospheric Models

### 2.1 Exponential Atmosphere ✅
**Status**: Implemented in `atmospheric_density()` function

**Current Implementation**:
```cpp
double atmospheric_density(double altitude, const Params& p) {
    return p.rho0 * std::exp(-altitude / p.H);
}
```

**Code Location**: `src/physics/dynamics.cpp` lines 80-83

### 2.2 International Standard Atmosphere (ISA) ❌🔥
**Status**: Not implemented

**Description**: Industry-standard atmospheric model with temperature, pressure, and density variations.

**Implementation Requirements**:
```cpp
struct ISA_Layer {
    double base_altitude;    // [m]
    double base_temperature; // [K]
    double base_pressure;    // [Pa]
    double lapse_rate;       // [K/m]
};

struct AtmosphereState {
    double temperature;      // [K]
    double pressure;         // [Pa]
    double density;          // [kg/m³]
    double speed_of_sound;   // [m/s]
    double viscosity;        // [Pa⋅s]
};

class ISA_Model {
public:
    static AtmosphereState compute_properties(double altitude);
private:
    static const std::vector<ISA_Layer> isa_layers;
};

// New file: src/physics/isa_atmosphere.hpp
//           src/physics/isa_atmosphere.cpp
```

**Code Impact**:
- Replace single density function with full atmosphere state
- Update drag calculations to use proper density
- Enable Mach number calculations
- Support for compressibility effects

### 2.3 Real-Time Weather Data ❌
**Status**: Not implemented

**Description**: Integration with weather APIs for real atmospheric conditions.

**Implementation Requirements**:
```cpp
class WeatherInterface {
public:
    virtual AtmosphereState get_conditions(double lat, double lon, 
                                         double altitude, double time) = 0;
};

class NOAA_Interface : public WeatherInterface {
    // Implementation for NOAA weather data
};

// New files: src/physics/weather/
//            weather_interface.hpp
//            noaa_interface.cpp
```

---

## 3. Gravitational Models

### 3.1 Variable Gravity ✅
**Status**: Implemented with altitude variation

**Current Implementation**:
```cpp
double gravity(double altitude, const Params& p) {
    if (altitude < 100000.0) {
        return p.g0;  // Constant below 100km
    } else {
        double R = p.R_earth;
        return p.g0 * (R * R) / ((R + altitude) * (R + altitude));
    }
}
```

**Code Location**: `src/physics/dynamics.cpp` lines 104-115

### 3.2 Earth Gravity Model (EGM2008) ❌🔥
**Status**: Not implemented

**Description**: High-fidelity Earth gravity field with gravitational harmonics.

**Implementation Requirements**:
```cpp
class EarthGravityModel {
public:
    static Eigen::Vector3d gravity_vector(double lat, double lon, double altitude);
    
private:
    static const int max_degree = 20;  // Spherical harmonic degree
    static const int max_order = 20;   // Spherical harmonic order
    static const std::vector<std::vector<double>> C_nm; // Coefficients
    static const std::vector<std::vector<double>> S_nm;
};

// New file: src/physics/earth_gravity.hpp
//           src/physics/earth_gravity.cpp
//           data/egm2008_coefficients.dat
```

**Code Impact**:
- 3D gravity vector instead of scalar
- Geographic coordinate transformations
- Significant data file for coefficients
- More complex gravity calculations

### 3.3 Multi-Body Gravitational Effects ❌
**Status**: Not implemented

**Description**: Gravitational perturbations from Moon, Sun, and planets.

**Implementation Requirements**:
```cpp
struct CelestialBody {
    std::string name;
    double mass;              // [kg]
    Eigen::Vector3d position; // [m] in Earth-centered frame
};

class CelestialMechanics {
public:
    static std::vector<CelestialBody> get_ephemeris(double julian_date);
    static Eigen::Vector3d third_body_acceleration(
        const Eigen::Vector3d& satellite_pos,
        const CelestialBody& body
    );
};
```

---

## 4. Aerodynamic Models

### 4.1 Simple Drag ✅
**Status**: Implemented with constant drag coefficient

**Current Implementation**:
```cpp
// In rhs() function:
double drag_mag = 0.5 * rho * p.Cd * p.A * v_rel_mag * v_rel_mag;
```

**Code Location**: `src/physics/dynamics.cpp` lines 35-48

### 4.2 Mach-Dependent Drag ❌🔥
**Status**: Not implemented

**Description**: Variable drag coefficient based on Mach number and angle of attack.

**Implementation Requirements**:
```cpp
struct AeroParams {
    double Cd_subsonic = 0.3;
    double Cd_transonic_peak = 1.2;
    double Cd_supersonic = 0.8;
    double mach_transonic_start = 0.8;
    double mach_transonic_end = 1.2;
    
    // Angle of attack effects
    double Cd_alpha_slope = 0.5;  // per radian
    double Cl_alpha_slope = 2.0;  // per radian
};

double compute_drag_coefficient(double mach, double alpha, const AeroParams& aero) {
    double Cd_mach = interpolate_mach_drag(mach, aero);
    double Cd_alpha = aero.Cd_alpha_slope * std::abs(alpha);
    return Cd_mach + Cd_alpha;
}

// New file: src/physics/aerodynamics.hpp
//           src/physics/aerodynamics.cpp
```

**Code Impact**:
- Require Mach number calculation (need speed of sound)
- Angle of attack calculation
- More complex drag force computation
- Lift force addition to equations

### 4.3 Panel Method Aerodynamics ❌⚠️
**Status**: Not implemented

**Description**: High-fidelity aerodynamics using computational fluid dynamics approach.

**Implementation Requirements**:
```cpp
class PanelMethod {
public:
    struct AeroCoefficients {
        double Cd, Cl, Cm;  // Drag, lift, moment coefficients
    };
    
    AeroCoefficients compute_coefficients(
        double mach, double alpha, double beta,
        const VehicleGeometry& geometry
    );
    
private:
    void solve_potential_flow();
    void apply_boundary_conditions();
};

// Requires: significant CFD implementation
// New directory: src/physics/cfd/
```

---

## 5. Propulsion Models

### 5.1 Simple Thrust ✅
**Status**: Implemented with constant specific impulse

**Current Implementation**:
```cpp
struct Control {
    double T;      // Thrust magnitude [N]
    double theta;  // Thrust angle [rad]
};

// Mass flow: dm/dt = -T / (Isp * g0)
```

**Code Location**: `src/physics/dynamics.hpp` lines 56-62, `src/physics/dynamics.cpp` lines 65-76

### 5.2 Performance-Based Thrust ❌🔥
**Status**: Not implemented

**Description**: Thrust and Isp variation with altitude, throttle setting, and mixture ratio.

**Implementation Requirements**:
```cpp
struct EngineParams {
    double sea_level_thrust;      // [N]
    double sea_level_isp;         // [s]
    double vacuum_thrust;         // [N]
    double vacuum_isp;            // [s]
    double min_throttle = 0.4;    // Minimum throttle setting
    double max_throttle = 1.0;    // Maximum throttle setting
    
    // Performance curves
    std::vector<double> altitude_points;  // [m]
    std::vector<double> thrust_ratios;    // [-]
    std::vector<double> isp_ratios;       // [-]
};

struct ThrustState {
    double thrust_available;     // [N]
    double isp_current;          // [s]
    double mass_flow_rate;       // [kg/s]
};

ThrustState compute_engine_performance(double altitude, double throttle, 
                                     const EngineParams& engine);

// New file: src/physics/propulsion.hpp
//           src/physics/propulsion.cpp
```

**Code Impact**:
- Replace constant Isp with variable performance
- Add throttle control to Control struct
- Altitude-dependent thrust calculations
- More realistic mass flow rates

### 5.3 Multi-Stage Rocket ❌🔥
**Status**: Not implemented

**Description**: Discrete staging events with different engine characteristics per stage.

**Implementation Requirements**:
```cpp
struct Stage {
    double dry_mass;           // [kg]
    double propellant_mass;    // [kg]
    EngineParams engine;
    double burn_time;          // [s]
    double coast_time;         // [s] optional coast phase
};

struct MultiStageVehicle {
    std::vector<Stage> stages;
    int current_stage = 0;
    std::vector<double> separation_times;
};

class StagingLogic {
public:
    static bool should_separate(const State& s, double time, 
                              const MultiStageVehicle& vehicle);
    static void perform_separation(State& s, MultiStageVehicle& vehicle);
};

// New file: src/physics/staging.hpp
//           src/physics/staging.cpp
```

**Code Impact**:
- Event detection for staging
- Discontinuous mass changes
- Stage-specific parameters
- Complex trajectory planning

### 5.4 Thrust Vector Control (TVC) ❌
**Status**: Not implemented

**Description**: Realistic actuator dynamics for thrust vectoring.

**Implementation Requirements**:
```cpp
struct TVCParams {
    double max_gimbal_angle;      // [rad]
    double gimbal_rate_limit;     // [rad/s]
    double actuator_bandwidth;    // [Hz]
    double hydraulic_pressure;    // [Pa]
    double power_consumption;     // [W]
};

struct TVCState {
    double current_angle_x;       // [rad]
    double current_angle_y;       // [rad]
    double angular_velocity_x;    // [rad/s]
    double angular_velocity_y;    // [rad/s]
};

class TVCActuator {
public:
    TVCState update(const TVCState& current, 
                   double commanded_angle_x, double commanded_angle_y,
                   double dt, const TVCParams& params);
};
```

---

## 6. Environmental Effects

### 6.1 Basic Wind Model 🔄
**Status**: Partially implemented - simple linear + sinusoidal

**Current Implementation**:
```cpp
std::pair<double, double> wind_velocity(double altitude, double time, const Params& p) {
    double wind_x = 0.1 * altitude / 1000.0; // Linear profile
    wind_x += 5.0 * std::sin(2.0 * M_PI * 0.1 * time); // Gust
    return {wind_x, 0.0};
}
```

**Code Location**: `src/physics/dynamics.cpp` lines 85-102

**Enhancement Needed**:
```cpp
enum class WindModel {
    NONE, LINEAR, HWM93, CUSTOM_PROFILE, TURBULENCE
};

struct WindProfile {
    WindModel model_type;
    std::vector<double> altitudes;         // [m]
    std::vector<Eigen::Vector2d> winds;    // [m/s]
    double turbulence_intensity;          // [m/s]
    double correlation_length;            // [m]
    double time_correlation;              // [s]
};

class WindModel {
public:
    static Eigen::Vector2d get_wind(double altitude, double time, 
                                   const WindProfile& profile);
    static Eigen::Vector2d get_turbulence(double altitude, double time,
                                         const WindProfile& profile);
};
```

### 6.2 Earth Rotation Effects ❌🔥
**Status**: Not implemented

**Description**: Coriolis and centrifugal accelerations due to Earth's rotation.

**Implementation Requirements**:
```cpp
struct EarthRotationParams {
    double omega = 7.2921159e-5;  // Earth's angular velocity [rad/s]
    double latitude;              // Launch site latitude [rad]
    double longitude;             // Launch site longitude [rad]
    bool enable_coriolis = true;
    bool enable_centrifugal = true;
};

Eigen::Vector3d coriolis_acceleration(const Eigen::Vector3d& velocity,
                                     const EarthRotationParams& earth) {
    Eigen::Vector3d omega_vec(0, 0, earth.omega);
    return -2.0 * omega_vec.cross(velocity);
}

Eigen::Vector3d centrifugal_acceleration(const Eigen::Vector3d& position,
                                        const EarthRotationParams& earth) {
    Eigen::Vector3d omega_vec(0, 0, earth.omega);
    return -omega_vec.cross(omega_vec.cross(position));
}

// Integration into rhs() function needed
```

**Code Impact**:
- Add latitude/longitude to parameters
- 3D vector calculations
- Geographic coordinate transformations
- Significant for long-range trajectories

### 6.3 Atmospheric Turbulence ❌
**Status**: Not implemented

**Description**: Stochastic atmospheric disturbances using Dryden or von Karman models.

**Implementation Requirements**:
```cpp
class TurbulenceModel {
public:
    enum Type { NONE, DRYDEN, VON_KARMAN };
    
    struct TurbulenceParams {
        Type model_type = DRYDEN;
        double intensity_low = 1.0;    // [m/s] at low altitude
        double intensity_high = 5.0;   // [m/s] at high altitude
        double scale_length = 1000.0;  // [m]
        unsigned int random_seed = 42;
    };
    
    Eigen::Vector3d generate_turbulence(double altitude, double time,
                                       const TurbulenceParams& params);
    
private:
    void initialize_filters();
    std::vector<double> filter_states;
};

// Requires: random number generation, digital filters
// New file: src/physics/turbulence.hpp
```

---

## 7. Specialized Physics Models

### 7.1 Aerodynamic Heating ❌🔥
**Status**: Not implemented

**Description**: Heat transfer calculations for thermal protection system design.

**Implementation Requirements**:
```cpp
struct ThermalState {
    double surface_temperature;    // [K]
    double heat_flux;             // [W/m²]
    double integrated_heat_load;  // [J/m²]
};

struct ThermalParams {
    double emissivity = 0.8;          // Surface emissivity
    double absorptivity = 0.9;        // Solar absorptivity
    double thermal_conductivity;      // [W/(m⋅K)]
    double specific_heat;             // [J/(kg⋅K)]
    double material_thickness;        // [m]
};

class AerodynamicHeating {
public:
    static double compute_stagnation_heating(double velocity, double density);
    static ThermalState update_thermal_state(const ThermalState& current,
                                           double heat_flux, double dt,
                                           const ThermalParams& params);
};

// New file: src/physics/thermal.hpp
//           src/physics/thermal.cpp
```

### 7.2 Propellant Sloshing ❌
**Status**: Not implemented

**Description**: Liquid propellant movement effects on vehicle dynamics.

**Implementation Requirements**:
```cpp
struct SloshParams {
    double tank_radius;           // [m]
    double fill_ratio;           // [0-1]
    double liquid_density;       // [kg/m³]
    std::vector<double> slosh_frequencies; // [Hz] for multiple modes
    std::vector<double> damping_ratios;    // [-]
};

struct SloshState {
    std::vector<double> modal_displacements; // [m]
    std::vector<double> modal_velocities;    // [m/s]
};

class PropellantSlosh {
public:
    static SloshState integrate_slosh(const SloshState& current,
                                    const Eigen::Vector3d& acceleration,
                                    double dt, const SloshParams& params);
    
    static Eigen::Vector3d compute_slosh_force(const SloshState& slosh,
                                             const SloshParams& params);
};
```

### 7.3 Vehicle Flexibility ❌⚠️
**Status**: Not implemented

**Description**: Structural dynamics using finite element or modal analysis.

**Implementation Requirements** (see section 1.3 for details)

### 7.4 Plume Impingement ❌
**Status**: Not implemented

**Description**: Rocket exhaust interaction with vehicle structure and payload.

**Implementation Requirements**:
```cpp
struct PlumeParams {
    double nozzle_exit_diameter;  // [m]
    double exit_velocity;         // [m/s]
    double exit_temperature;      // [K]
    double expansion_angle;       // [rad]
};

class PlumeImpingement {
public:
    static std::vector<double> compute_pressure_distribution(
        const Eigen::Vector3d& nozzle_position,
        const Eigen::Vector3d& nozzle_direction,
        const std::vector<Eigen::Vector3d>& surface_points,
        const PlumeParams& plume
    );
};
```

---

## 8. Advanced Integration and Numerical Methods

### 8.1 Fixed-Step RK4 ✅
**Status**: Implemented in `ForwardIntegrator` class

**Code Location**: `src/physics/dynamics.cpp` lines 204-218

### 8.2 Adaptive RK45 ✅
**Status**: Implemented with Dormand-Prince method

**Code Location**: `src/physics/dynamics.cpp` lines 140-202

### 8.3 Symplectic Integrators ❌
**Status**: Not implemented

**Description**: Energy-conserving integrators for long-term orbital mechanics.

**Implementation Requirements**:
```cpp
class SymplecticIntegrator {
public:
    // Störmer-Verlet method
    static State integrate_verlet(const State& s0, 
                                const std::function<Eigen::Vector3d(Eigen::Vector3d)>& force_func,
                                double dt);
    
    // Leapfrog method
    static State integrate_leapfrog(const State& s0,
                                  const std::function<Eigen::Vector3d(Eigen::Vector3d)>& force_func,
                                  double dt);
};
```

### 8.4 Variable-Order Methods ❌
**Status**: Not implemented

**Description**: Automatic order selection for optimal accuracy/efficiency.

---

## 9. Implementation Priority Matrix

| Model               | Impact | Complexity | Priority | Files to Create            |
|---------------------|--------|------------|----------|----------------------------|
| ISA Atmosphere      |  High  | Low        | 🔥🔥🔥 | `isa_atmosphere.hpp/cpp`    |
| Mach-dependent Drag |  High  | Medium     | 🔥🔥🔥 | `aerodynamics.hpp/cpp`      |
| Earth Rotation      |  High  | Medium     | 🔥🔥   | `earth_rotation.hpp/cpp`    |
| Multi-stage         |  High  | High       | 🔥🔥   | `staging.hpp/cpp`           |
| Performance Thrust  | Medium | Medium     | 🔥      | `propulsion.hpp/cpp`        |
| 6-DOF Dynamics      |  High  | High       | 🔥      | `dynamics_6dof.hpp/cpp`     |
| Aerodynamic Heating | Medium | Low        | 🔥      | `thermal.hpp/cpp`           |
| TVC Dynamics        | Medium | Medium     | ⚠️      | `tvc.hpp/cpp`               |
| Flexible Body       |   Low  | Very High  | ⚠️      | `flexible_dynamics.hpp/cpp` |
| Propellant Slosh    |   Low  | High       | ⚠️      | `slosh.hpp/cpp`             |

## 10. Code Architecture Recommendations

### Directory Structure
```
src/physics/
├── core/
│   ├── dynamics.hpp/cpp          ✅ (existing)
│   ├── dynamics_6dof.hpp/cpp     ❌ (new)
│   └── integrator.hpp/cpp        ✅ (existing)
├── environment/
│   ├── atmosphere.hpp/cpp        🔄 (enhance existing)
│   ├── isa_atmosphere.hpp/cpp    ❌ (new)
│   ├── wind.hpp/cpp             🔄 (enhance existing)
│   ├── turbulence.hpp/cpp       ❌ (new)
│   └── earth_rotation.hpp/cpp   ❌ (new)
├── aerodynamics/
│   ├── aerodynamics.hpp/cpp     ❌ (new)
│   ├── drag_models.hpp/cpp      ❌ (new)
│   └── panel_method.hpp/cpp     ❌ (future)
├── propulsion/
│   ├── propulsion.hpp/cpp       ❌ (new)
│   ├── staging.hpp/cpp          ❌ (new)
│   └── tvc.hpp/cpp              ❌ (new)
├── structures/
│   ├── flexible_dynamics.hpp/cpp ❌ (new)
│   ├── slosh.hpp/cpp            ❌ (new)
│   └── thermal.hpp/cpp          ❌ (new)
└── utils/
    ├── constants.hpp            ❌ (new)
    ├── coordinates.hpp/cpp      ❌ (new)
    └── interpolation.hpp/cpp    ❌ (new)
```

### Integration Points
1. **Main RHS Function**: Update to call specialized models
2. **Parameter Structures**: Extend with new model parameters
3. **State Vectors**: Support for additional state variables
4. **Control Inputs**: Enhanced control vector definitions
5. **Integrator Interface**: Support for event detection (staging)

### Testing Strategy
Each new physics model should include:
- Unit tests in `tests/test_[model_name].cpp`
- Validation against analytical solutions where possible
- Performance benchmarks
- Integration tests with existing dynamics

---

## Summary

Your current implementation provides a solid foundation with basic 3-DOF dynamics, exponential atmosphere, and variable gravity. The highest-impact improvements would be:

1. **ISA atmosphere model** - Industry standard
2. **Mach-dependent aerodynamics** - Critical for realism
3. **Earth rotation effects** - Essential for accuracy
4. **Multi-stage support** - Necessary for realistic missions

These enhancements would significantly improve simulation fidelity while maintaining computational efficiency for your optimization and machine learning workflows.
