cmake_minimum_required(VERSION 3.18)
project(amp_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(GNUInstallDirs)

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

set_source_files_properties(amp_kernels.c PROPERTIES LANGUAGE CXX)
set_source_files_properties(amp_debug_alloc.c PROPERTIES LANGUAGE CXX)

add_library(amp_native SHARED
    graph_runtime.cpp
    fft_backend.cpp
    amp_debug_alloc.c
    amp_kernels.c
    amp_descriptor_builder.c
)
set_property(TARGET amp_native PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
set_property(TARGET amp_native PROPERTY MSVC_DEBUG_INFORMATION_FORMAT None)

find_package(Threads REQUIRED)

target_include_directories(amp_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

target_compile_definitions(amp_native PRIVATE AMP_NATIVE_BUILD)
target_link_libraries(amp_native PRIVATE Threads::Threads)

add_executable(demo_kpn_native
    demo_kpn_native.cpp
)
target_link_libraries(demo_kpn_native PRIVATE amp_native)
set_property(TARGET demo_kpn_native PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
set_property(TARGET demo_kpn_native PROPERTY MSVC_DEBUG_INFORMATION_FORMAT None)
if (_AMP_NATIVE_EXTRA_COMPILE_OPTIONS)
    target_compile_options(demo_kpn_native PRIVATE ${_AMP_NATIVE_EXTRA_COMPILE_OPTIONS})
endif()
if (_AMP_NATIVE_EXTRA_LINK_OPTIONS)
    target_link_options(demo_kpn_native PRIVATE ${_AMP_NATIVE_EXTRA_LINK_OPTIONS})
endif()
if (MSVC)
    target_compile_options(demo_kpn_native PRIVATE $<$<CONFIG:Release>:/Od> /FS)
    target_link_options(demo_kpn_native PRIVATE $<$<CONFIG:Release>:/DEBUG:NONE>)
endif()

option(AMP_NATIVE_ENABLE_LOGGING "Enable native runtime logging" OFF)
if (AMP_NATIVE_ENABLE_LOGGING)
    target_compile_definitions(amp_native PRIVATE AMP_NATIVE_ENABLE_LOGGING)
endif()

set(_AMP_NATIVE_NODE_ASSETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nodes")
if (EXISTS "${_AMP_NATIVE_NODE_ASSETS_DIR}")
    set(_AMP_NATIVE_NODE_ASSETS_OUTPUT "${CMAKE_BINARY_DIR}/node_assets")
    set(_AMP_NATIVE_NODE_ASSETS_STAMP "${_AMP_NATIVE_NODE_ASSETS_OUTPUT}/.stamp")
    add_custom_command(
        OUTPUT "${_AMP_NATIVE_NODE_ASSETS_STAMP}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${_AMP_NATIVE_NODE_ASSETS_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${_AMP_NATIVE_NODE_ASSETS_DIR}" "${_AMP_NATIVE_NODE_ASSETS_OUTPUT}"
        COMMAND ${CMAKE_COMMAND} -E touch "${_AMP_NATIVE_NODE_ASSETS_STAMP}"
        COMMENT "Syncing native node asset bundles"
        VERBATIM
    )
    add_custom_target(amp_native_node_assets ALL
        DEPENDS "${_AMP_NATIVE_NODE_ASSETS_STAMP}"
    )
    add_dependencies(amp_native amp_native_node_assets)
    add_dependencies(demo_kpn_native amp_native_node_assets)
    install(
        DIRECTORY "${_AMP_NATIVE_NODE_ASSETS_DIR}/"
        DESTINATION "${CMAKE_INSTALL_DATAROOTDIR}/amp/nodes"
    )
endif()

set(_AMP_NATIVE_EXTRA_COMPILE_OPTIONS "$ENV{AMP_NATIVE_EXTRA_COMPILE_ARGS}")
if (_AMP_NATIVE_EXTRA_COMPILE_OPTIONS)
    separate_arguments(_AMP_NATIVE_EXTRA_COMPILE_OPTIONS NATIVE_COMMAND ${_AMP_NATIVE_EXTRA_COMPILE_OPTIONS})
    target_compile_options(amp_native PRIVATE ${_AMP_NATIVE_EXTRA_COMPILE_OPTIONS})
endif()

set(_AMP_NATIVE_EXTRA_LINK_OPTIONS "$ENV{AMP_NATIVE_EXTRA_LINK_ARGS}")
if (_AMP_NATIVE_EXTRA_LINK_OPTIONS)
    separate_arguments(_AMP_NATIVE_EXTRA_LINK_OPTIONS NATIVE_COMMAND ${_AMP_NATIVE_EXTRA_LINK_OPTIONS})
    target_link_options(amp_native PRIVATE ${_AMP_NATIVE_EXTRA_LINK_OPTIONS})
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

    set(_AMP_NATIVE_TEST_TARGETS
        kpn_unit_test
        test_thermo_param
        test_amp_run_node_v2
        test_fft_noise_gradient
        test_fft_division_node
        test_pitch_shift_node
    )

    foreach(_target IN LISTS _AMP_NATIVE_TEST_TARGETS)
        if(TARGET ${_target})
            if(_AMP_NATIVE_EXTRA_COMPILE_OPTIONS)
                target_compile_options(${_target} PRIVATE ${_AMP_NATIVE_EXTRA_COMPILE_OPTIONS})
            endif()
            if(_AMP_NATIVE_EXTRA_LINK_OPTIONS)
                target_link_options(${_target} PRIVATE ${_AMP_NATIVE_EXTRA_LINK_OPTIONS})
            endif()
        endif()
    endforeach()
endif()
