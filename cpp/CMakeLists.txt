cmake_minimum_required(VERSION 3.18)
project(ThesisTorchApp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow user to specify LibTorch root via -DTorch_DIR or TORCH_HOME
if(NOT DEFINED Torch_DIR AND DEFINED ENV{TORCH_HOME})
  file(TO_CMAKE_PATH "$ENV{TORCH_HOME}" _TORCH_HOME)
  set(Torch_DIR "${_TORCH_HOME}/share/cmake/Torch" CACHE PATH "Path to Torch cmake package")
endif()

# If user provides TORCH_HOME at configure time
if(DEFINED TORCH_HOME)
  file(TO_CMAKE_PATH "${TORCH_HOME}" _TORCH_HOME_ARG)
  set(Torch_DIR "${_TORCH_HOME_ARG}/share/cmake/Torch" CACHE PATH "Path to Torch cmake package")
endif()

find_package(Torch REQUIRED PATHS ${Torch_DIR} NO_DEFAULT_PATH)

add_executable(thesis_torch_app src/main.cpp)

# Link Torch
if(TORCH_FOUND)
  target_link_libraries(thesis_torch_app PRIVATE "${TORCH_LIBRARIES}")
  target_compile_definitions(thesis_torch_app PRIVATE -D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)
  target_compile_features(thesis_torch_app PRIVATE cxx_std_17)
  target_include_directories(thesis_torch_app PRIVATE ${TORCH_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "LibTorch not found. Set Torch_DIR to <libtorch>/share/cmake/Torch")
endif()

# On Windows, copy DLLs next to the exe for ease of running
if(WIN32)
  set_property(TARGET thesis_torch_app PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  add_custom_command(TARGET thesis_torch_app POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_INSTALL_PREFIX}/lib"
            "$<TARGET_FILE_DIR:thesis_torch_app>")
endif()

message(STATUS "TORCH_INSTALL_PREFIX: ${TORCH_INSTALL_PREFIX}")
message(STATUS "TORCH_LIBRARIES: ${TORCH_LIBRARIES}")

