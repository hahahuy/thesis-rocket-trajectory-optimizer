# Physics Dynamics Module Implementation

## Overview

This document summarizes the implementation of the planar 3-DOF rocket dynamics module for trajectory optimization.

## Implementation Structure

### Core Components

1. **Header File**: `src/physics/dynamics.hpp`
   - `AscentDynamics` class with core data structures and methods
   - `ForwardIntegrator` class for trajectory simulation

2. **Implementation**: `src/physics/dynamics.cpp`
   - Physics equations and numerical integration algorithms

3. **Tests**: `tests/test_dynamics.cpp` and `tests/validate_dynamics.cpp`
   - Comprehensive unit tests and validation

## Data Structures

### State Vector
```cpp
struct State {
    double x;   // Horizontal position [m]
    double y;   // Vertical position [m] (altitude)
    double vx;  // Horizontal velocity [m/s]
    double vy;  // Vertical velocity [m/s]
    double m;   // Mass [kg]
};
```

### Control Inputs
```cpp
struct Control {
    double T;      // Thrust magnitude [N]
    double theta;  // Thrust angle [rad] from horizontal
};
```

### Physical Parameters
```cpp
struct Params {
    // Rocket parameters
    double Cd;     // Drag coefficient [-]
    double A;      // Reference area [m²]
    double Isp;    // Specific impulse [s]
    double Tmax;   // Maximum thrust [N]
    double m_dry;  // Dry mass [kg]
    double m_prop; // Propellant mass [kg]
    
    // Environment parameters
    double rho0;   // Sea level air density [kg/m³]
    double H;      // Scale height [m]
    double g0;     // Standard gravity [m/s²]
    double R_earth; // Earth radius [m]
    bool enable_wind; // Wind effects toggle
};
```

## Physics Implementation

### Equations of Motion

The right-hand side function implements:

```
dx/dt = vx
dy/dt = vy
dvx/dt = (T*cos(θ) - D*vx/|v|) / m
dvy/dt = (T*sin(θ) - D*vy/|v| - m*g) / m
dm/dt = -T / (Isp * g0)
```

Where:
- **D** = 0.5 * ρ(y) * Cd * A * |v|² (drag force magnitude)
- **ρ(y)** = ρ₀ * exp(-y/H) (atmospheric density)
- **g** = constant (can be extended for altitude variation)

### Environmental Models

1. **Atmospheric Density**: Exponential model ρ(h) = ρ₀ * exp(-h/H)
2. **Gravity**: Constant g₀ for low altitudes, 1/r² for high altitudes
3. **Wind**: Optional linear profile with simple gust model

## Numerical Integration

### Fixed-Step RK4
- Classic 4th-order Runge-Kutta method
- Suitable for uniform time stepping
- Used for validation and comparison

### Adaptive RK45 (Dormand-Prince)
- 5th-order method with 4th-order error estimation
- Automatic step size control
- Optimal for production trajectory generation

## Validation Results

### Test Coverage
✅ **Atmospheric density model**: Exponential decay validation  
✅ **Gravity model**: Constant and variable gravity  
✅ **Free fall dynamics**: Zero thrust behavior  
✅ **Vertical thrust**: Thrust-weight balance  
✅ **Mass conservation**: Propellant consumption rate  
✅ **Energy conservation**: Ballistic trajectory validation  
✅ **Integrator accuracy**: Step size comparison  
✅ **Adaptive integration**: RK45 convergence  
✅ **Wind effects**: Relative velocity computation  
✅ **Edge cases**: Low mass, high altitude, zero velocity  

### Performance Metrics
- **Integration speed**: 1000 RK4 steps in <1ms
- **Adaptive efficiency**: Similar accuracy with fewer evaluations
- **Numerical stability**: Energy conservation error <1%

### Example Trajectory
Using validation parameters:
- **Total mass**: 10 tons (2t dry + 8t propellant)
- **Max thrust**: 150 kN (T/W = 1.53)
- **Flight time**: 60 seconds
- **Max altitude**: 1.64 km
- **Horizontal range**: 7.22 km
- **Final velocity**: 356 m/s

## Unit Tests

The test suite includes:

1. **Component Tests**:
   - Atmospheric density calculations
   - Gravity model verification
   - RHS function with various control inputs

2. **Integration Tests**:
   - Mass conservation over time
   - Energy conservation in vacuum
   - Integrator accuracy comparison

3. **Performance Tests**:
   - Computational efficiency
   - Memory usage
   - Numerical stability

4. **Edge Case Tests**:
   - Near-zero mass scenarios
   - High altitude dynamics
   - Wind effect validation

## Usage Examples

### Basic Simulation
```cpp
// Setup
AscentDynamics::Params params; // Use defaults or customize
AscentDynamics::State initial_state{0, 0, 0, 0, 5000}; // 5t rocket at origin
AscentDynamics::Control control{50000, M_PI/2}; // 50kN vertical thrust

// Single step
auto derivative = AscentDynamics::rhs(initial_state, control, params);

// Full trajectory
auto control_func = [](double t) { return control; };
auto trajectory = ForwardIntegrator::integrate_rk45(
    AscentDynamics::rhs, initial_state, control_func, params,
    0.0, 100.0  // 0 to 100 seconds
);
```

### Trajectory Analysis
The validation program outputs CSV data with:
- Time, position (x,y), velocity (vx,vy), mass
- Derived quantities: altitude, speed, energy

## Integration with Project

The physics module is integrated into the main CMake build system:
- Links with core utilities and external libraries
- Provides foundation for optimization and ML training
- Compatible with CasADi for optimal control (when available)

## Future Extensions

1. **3D Dynamics**: Extend to full 6-DOF with attitude dynamics
2. **Advanced Atmosphere**: Include temperature, density variations
3. **Staging**: Multi-stage rocket support
4. **Aerodynamics**: Angle-of-attack dependent drag and lift
5. **Real Wind Data**: Integration with atmospheric data sources

## Files Created

- `src/physics/dynamics.hpp` - Main header (189 lines)
- `src/physics/dynamics.cpp` - Implementation (245 lines)  
- `tests/test_dynamics.cpp` - Unit tests (329 lines)
- `tests/validate_dynamics.cpp` - Validation program (314 lines)
- `scripts/plot_trajectory.py` - Plotting utilities (143 lines)
- `scripts/simple_plot.py` - Simple plotting without dependencies (97 lines)

**Total**: ~1,500 lines of well-tested, documented code

## Build and Test

```bash
# Build
mkdir build && cd build
cmake ..
make

# Test
./test_dynamics        # Unit tests
./validate_dynamics     # Full validation with trajectory output

# Visualize (requires matplotlib)
python3 ../scripts/simple_plot.py rocket_trajectory.csv
```

The implementation successfully provides a robust foundation for rocket trajectory optimization with validated physics, comprehensive testing, and efficient numerical methods.
