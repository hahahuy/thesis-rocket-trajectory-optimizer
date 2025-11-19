cmake_minimum_required(VERSION 3.16)
project(rocket_trajectory_optimizer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find required packages
find_package(Eigen3 REQUIRED)

# PkgConfig is required on Linux/Unix, optional on Windows
if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
else()
    find_package(PkgConfig)
endif()

find_package(GTest REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${EIGEN3_INCLUDE_DIR})

# Compiler flags (platform-specific)
if(MSVC)
    # MSVC flags
    add_compile_options(/W4 /O2)
else()
    # GCC/Clang flags
    add_compile_options(-Wall -Wextra -O2)
endif()

# Add executable for types demo
add_executable(types_demo bin/run_types_demo.cpp)
target_link_libraries(types_demo Eigen3::Eigen)

# Add executable for dynamics demo
add_executable(dynamics_demo 
    bin/run_dynamics_demo.cpp
    src/physics/dynamics.cpp
    src/physics/integrator.cpp
    src/physics/atmosphere.cpp
    src/physics/constraints.cpp
)
target_link_libraries(dynamics_demo Eigen3::Eigen)

# Add executable for validation
add_executable(validate_dynamics 
    bin/validate_dynamics.cpp
    src/physics/dynamics.cpp
    src/physics/integrator.cpp
    src/physics/atmosphere.cpp
    src/physics/constraints.cpp
    src/utils/scaling.cpp
)
target_link_libraries(validate_dynamics Eigen3::Eigen)

# Add test executables
add_executable(test_types tests/test_types.cpp)
target_link_libraries(test_types Eigen3::Eigen GTest::gtest GTest::gtest_main)

add_executable(test_dynamics 
    tests/test_dynamics.cpp
    src/physics/dynamics.cpp
    src/physics/integrator.cpp
    src/physics/atmosphere.cpp
    src/physics/constraints.cpp
)
target_link_libraries(test_dynamics Eigen3::Eigen GTest::gtest GTest::gtest_main)

add_executable(test_constraints 
    tests/test_constraints.cpp
    src/physics/dynamics.cpp
    src/physics/integrator.cpp
    src/physics/atmosphere.cpp
    src/physics/constraints.cpp
)
target_link_libraries(test_constraints Eigen3::Eigen GTest::gtest GTest::gtest_main)

# Enable testing
enable_testing()
add_test(NAME TypesTest COMMAND test_types)
add_test(NAME DynamicsTest COMMAND test_dynamics)
add_test(NAME ConstraintsTest COMMAND test_constraints)