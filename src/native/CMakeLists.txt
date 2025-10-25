cmake_minimum_required(VERSION 3.18)
project(amp_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(EIGEN3_SUBMODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/eigen")
set(EIGEN3_INCLUDE_DIR "${EIGEN3_SUBMODULE_DIR}")
if (NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
    include(FetchContent)
    message(STATUS "Eigen not found at ${EIGEN3_INCLUDE_DIR}; fetching via FetchContent")
    FetchContent_Declare(
        eigen
        URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
    )
    FetchContent_GetProperties(eigen)
    if (NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    if (DEFINED eigen_SOURCE_DIR AND EXISTS "${eigen_SOURCE_DIR}/Eigen/Core")
        set(EIGEN3_INCLUDE_DIR "${eigen_SOURCE_DIR}")
    else()
        message(FATAL_ERROR "Failed to fetch Eigen and locate Eigen/Core")
    endif()
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

# Optional: build native-only unit tests for the graph runtime (KPN)
option(AMP_NATIVE_BUILD_TESTS "Build native unit tests for the graph runtime" ON)
if (AMP_NATIVE_BUILD_TESTS)
    enable_testing()
    add_executable(kpn_unit_test
        tests/kpn_unit_test.cpp
    )
    target_include_directories(kpn_unit_test PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(kpn_unit_test PRIVATE amp_native)
    add_test(NAME kpn_unit_test COMMAND kpn_unit_test)

    add_executable(test_thermo_param
        tests/test_thermo_param.cpp
    )
    target_include_directories(test_thermo_param PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_thermo_param PRIVATE amp_native)
    add_test(NAME test_thermo_param COMMAND test_thermo_param)

    add_executable(test_amp_run_node_v2
        tests/test_amp_run_node_v2.cpp
    )
    target_include_directories(test_amp_run_node_v2 PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_amp_run_node_v2 PRIVATE amp_native)
    add_test(NAME test_amp_run_node_v2 COMMAND test_amp_run_node_v2)

    add_executable(test_fft_noise_gradient
        tests/test_fft_noise_gradient.cpp
    )
    target_include_directories(test_fft_noise_gradient PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_fft_noise_gradient PRIVATE amp_native)
    add_test(NAME test_fft_noise_gradient COMMAND test_fft_noise_gradient --output output.wav)
    set_tests_properties(test_fft_noise_gradient PROPERTIES WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../..")

    add_executable(test_fft_division_node
        tests/test_fft_division_node.cpp
    )
    target_include_directories(test_fft_division_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_fft_division_node PRIVATE amp_native)
    add_test(NAME test_fft_division_node COMMAND test_fft_division_node)

    add_executable(test_pitch_shift_node
        tests/test_pitch_shift_node.cpp
    )
    target_include_directories(test_pitch_shift_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_pitch_shift_node PRIVATE amp_native)
    add_test(NAME test_pitch_shift_node COMMAND test_pitch_shift_node)
endif()
