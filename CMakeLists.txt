cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(pinn_rocket_trajectory LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable position independent code
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

# Find system packages
find_package(PkgConfig REQUIRED)

# Find Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(Eigen3_FOUND)
    message(STATUS "Found Eigen3 ${Eigen3_VERSION}")
endif()

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS CXX)
if(HDF5_FOUND)
    message(STATUS "Found HDF5 ${HDF5_VERSION}")
endif()

# Find spdlog
pkg_check_modules(SPDLOG REQUIRED spdlog)

# Find nlohmann_json
find_package(nlohmann_json 3.2.0 REQUIRED)

# Find GTest
find_package(GTest REQUIRED)

# Find LibTorch
set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/external/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Try to find CasADi
find_package(casadi QUIET)
if(casadi_FOUND)
    message(STATUS "Found CasADi ${casadi_VERSION}")
    set(CASADI_AVAILABLE ON)
else()
    message(STATUS "CasADi not found - optimal control features will be limited")
    message(STATUS "Run ./scripts/build_casadi_ipopt.sh to build CasADi and IPOPT")
    set(CASADI_AVAILABLE OFF)
endif()

# Create header files to check what libraries are available
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h"
)

# Create main library target
add_library(pinn_rocket_lib
    # Core utilities that work with current libraries
    src/utils/io.cpp
    src/utils/logging.cpp
    # Physics dynamics module
    src/physics/dynamics.cpp
)

# Create placeholder source files if they don't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/utils)

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/io.cpp)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/io.cpp
        "#include <iostream>\n// Placeholder for I/O utilities\nnamespace utils { void placeholder() {} }\n")
endif()

if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/logging.cpp)
    file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/logging.cpp
        "#include <spdlog/spdlog.h>\n// Placeholder for logging utilities\nnamespace utils { void setup_logging() { spdlog::info(\"Logging initialized\"); } }\n")
endif()

# Add include directories
target_include_directories(pinn_rocket_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries
target_link_libraries(pinn_rocket_lib PUBLIC
    Eigen3::Eigen
    ${TORCH_LIBRARIES}
    ${HDF5_CXX_LIBRARIES}
    nlohmann_json::nlohmann_json
    ${SPDLOG_LIBRARIES}
)

# Add CasADi if available
if(CASADI_AVAILABLE)
    target_link_libraries(pinn_rocket_lib PUBLIC casadi)
endif()

# Include directories for dependencies
target_include_directories(pinn_rocket_lib SYSTEM PUBLIC
    ${HDF5_CXX_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

# Set compiler definitions
target_compile_definitions(pinn_rocket_lib PRIVATE
    ${HDF5_CXX_DEFINITIONS}
)

if(CASADI_AVAILABLE)
    target_compile_definitions(pinn_rocket_lib PRIVATE CASADI_AVAILABLE)
endif()

# Testing
enable_testing()

# Create tests directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Physics dynamics tests
add_executable(test_dynamics tests/test_dynamics.cpp)
target_link_libraries(test_dynamics 
    pinn_rocket_lib 
    GTest::gtest_main
)
add_test(NAME DynamicsTests COMMAND test_dynamics)

# Physics dynamics validation
add_executable(validate_dynamics tests/validate_dynamics.cpp)
target_link_libraries(validate_dynamics pinn_rocket_lib)

# Installation
install(TARGETS pinn_rocket_lib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include/pinn_rocket
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Print summary
message(STATUS "")
message(STATUS "=== Configuration Summary ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Eigen3: ${Eigen3_VERSION}")
message(STATUS "HDF5: ${HDF5_VERSION}")
message(STATUS "LibTorch: Found")
if(CASADI_AVAILABLE)
    message(STATUS "CasADi: ${casadi_VERSION} ✓")
else()
    message(STATUS "CasADi: Not found - run build script")
endif()
message(STATUS "nlohmann/json: Found")
message(STATUS "spdlog: Found")
message(STATUS "GoogleTest: Found")
message(STATUS "===============================")
message(STATUS "")
message(STATUS "Ready to build! Core libraries verified.")
if(NOT CASADI_AVAILABLE)
    message(STATUS "For complete optimization features, run:")
    message(STATUS "  ./scripts/build_casadi_ipopt.sh")
endif()
message(STATUS "")