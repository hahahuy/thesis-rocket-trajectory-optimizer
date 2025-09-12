cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pinn_rocket_trajectory LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Platform-specific flags
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Configure header (kept minimal)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h"
)

# Main library target (exclude logging.cpp to avoid spdlog dependency)
add_library(pinn_rocket_lib
    src/utils/io.cpp
    src/physics/aerodynamics/aerodynamics.cpp
    src/physics/environment/earth_rotation.cpp
    src/physics/environment/isa_atmosphere.cpp
    src/physics/propulsion/staging.cpp
    src/physics/dynamics.cpp
)

target_include_directories(pinn_rocket_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Testing setup with FetchContent for GoogleTest
include(FetchContent)
set(BUILD_GMOCK OFF CACHE BOOL "Disable gmock" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "Disable install for gtest" FORCE)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_dynamics tests/test_dynamics.cpp)
target_link_libraries(test_dynamics
    PRIVATE pinn_rocket_lib GTest::gtest_main
)
add_test(NAME DynamicsTests COMMAND test_dynamics)

add_executable(validate_dynamics tests/validate_dynamics.cpp)
target_link_libraries(validate_dynamics PRIVATE pinn_rocket_lib)

add_executable(run_trajectory_demo tests/validate_dynamics.cpp)
target_link_libraries(run_trajectory_demo PRIVATE pinn_rocket_lib)

# Installation (library only)
install(TARGETS pinn_rocket_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include/pinn_rocket
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "GoogleTest: Fetched via FetchContent")
message(STATUS "===============================")
message(STATUS "")