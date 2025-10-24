cmake_minimum_required(VERSION 3.18)
project(amp_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/eigen")
if (NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
    message(FATAL_ERROR "Eigen not found in ${EIGEN3_INCLUDE_DIR}. Initialise the third_party/eigen submodule.")
endif()

add_library(amp_native SHARED
    graph_runtime.cpp
    amp_kernels.c
)

target_include_directories(amp_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

target_compile_definitions(amp_native PRIVATE AMP_NATIVE_BUILD)

option(AMP_NATIVE_ENABLE_LOGGING "Enable native runtime logging" OFF)
if (AMP_NATIVE_ENABLE_LOGGING)
    target_compile_definitions(amp_native PRIVATE AMP_NATIVE_ENABLE_LOGGING)
endif()

set_target_properties(amp_native PROPERTIES
    OUTPUT_NAME "amp_native"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
