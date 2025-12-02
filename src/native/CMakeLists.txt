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

if (NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}")
endif()

option(AMP_NATIVE_USE_SYSTEM_FFTFREE "Use system-provided fftfree instead of vendored sources" OFF)

set(_FFTFREE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/fftfree")
set(_FFTFREE_TARGET "")
set(_FFTFREE_INCLUDE_DIRS "")

set(_FFTFREE_FETCHED FALSE)
set(_FFTFREE_EXTERNAL_SOURCE "")
set(_FFTFREE_EXTERNAL_BINARY "")

if (AMP_NATIVE_USE_SYSTEM_FFTFREE)
    # Explicitly request the system/package-provided fftfree. If not found we fail
    # rather than silently pulling a different copy, keeping the choice intentional.
    find_package(fftfree CONFIG REQUIRED)
    if (TARGET fftfree::fft_cffi)
        set(_FFTFREE_TARGET fftfree::fft_cffi)
    elseif (TARGET fft_cffi)
        set(_FFTFREE_TARGET fft_cffi)
    else()
        message(FATAL_ERROR "fftfree package located but no fft_cffi target was exported")
    endif()
    get_target_property(_FFTFREE_INCLUDE_DIRS ${_FFTFREE_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    if (NOT _FFTFREE_INCLUDE_DIRS OR _FFTFREE_INCLUDE_DIRS STREQUAL "_FFTFREE_INCLUDE_DIRS-NOTFOUND")
        set(_FFTFREE_INCLUDE_DIRS "")
    endif()
else()
    if (EXISTS "${_FFTFREE_SOURCE_DIR}/CMakeLists.txt")
        set(_FFTFREE_EXTERNAL_SOURCE "${_FFTFREE_SOURCE_DIR}")
    else()
        include(FetchContent)
        message(STATUS "fftfree sources missing; fetching from upstream repository")
        FetchContent_Declare(
            fftfree_external
            GIT_REPOSITORY https://github.com/sangderenard/fftfree.git
            GIT_TAG 9e875520f2c1fc2eda067f125595af21f1355364
        )
        FetchContent_GetProperties(fftfree_external)
        if (NOT fftfree_external_POPULATED)
            FetchContent_Populate(fftfree_external)
        endif()
        set(_FFTFREE_EXTERNAL_SOURCE "${fftfree_external_SOURCE_DIR}")
        set(_FFTFREE_EXTERNAL_BINARY "${fftfree_external_BINARY_DIR}")
        set(_FFTFREE_FETCHED TRUE)
    endif()

    if (_FFTFREE_EXTERNAL_SOURCE)
        set(FFTFREE_BUILD_SHARED OFF CACHE BOOL "Build fft_cffi as a shared library" FORCE)
        set(FFTFREE_BUILD_APPS OFF CACHE BOOL "Build fftfree standalone demos and tools" FORCE)
        set(FFTFREE_BUILD_TESTS OFF CACHE BOOL "Build fftfree test executables" FORCE)
        set(FFTFREE_OPENMP OFF CACHE BOOL "Enable OpenMP in fftfree" FORCE)
        if (_FFTFREE_FETCHED)
            add_subdirectory("${_FFTFREE_EXTERNAL_SOURCE}" "${_FFTFREE_EXTERNAL_BINARY}")
        else()
            add_subdirectory(fftfree)
        endif()
        set(_FFTFREE_TARGET fft_cffi)
        list(APPEND _FFTFREE_INCLUDE_DIRS "${_FFTFREE_EXTERNAL_SOURCE}")
    else()
        message(FATAL_ERROR "fftfree sources not present. Either initialise the submodule, or allow FetchContent to download it (default). To opt into a system package, set AMP_NATIVE_USE_SYSTEM_FFTFREE=ON and ensure fftfree is installed.")
    endif()
endif()

# Ensure consistent output directories for multi-config generators (Visual Studio)
if(CMAKE_CONFIGURATION_TYPES)
    foreach(_cfg IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER ${_cfg} _ucfg)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_ucfg} "${CMAKE_BINARY_DIR}/${_cfg}" CACHE PATH "" FORCE)
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_ucfg} "${CMAKE_BINARY_DIR}/${_cfg}" CACHE PATH "" FORCE)
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_ucfg} "${CMAKE_BINARY_DIR}/${_cfg}" CACHE PATH "" FORCE)
    endforeach()
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" CACHE PATH "" FORCE)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" CACHE PATH "" FORCE)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" CACHE PATH "" FORCE)
endif()

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/amp_kernels.c PROPERTIES LANGUAGE CXX)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/amp_debug_alloc.c PROPERTIES LANGUAGE CXX)

set(_AMP_NATIVE_SOURCES
    graph_runtime.cpp
    fft_backend.cpp
    amp_debug_alloc.c
    amp_kernels_wrapper.cpp
    amp_mailbox_cpp.cpp
    amp_descriptor_builder.c
    kpn_cffi.cpp
    kpn_stream_session.cpp
    kpn_sampler_registry.cpp
)

add_library(amp_native SHARED ${_AMP_NATIVE_SOURCES})
set_property(TARGET amp_native PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
set_property(TARGET amp_native PROPERTY MSVC_DEBUG_INFORMATION_FORMAT None)

find_package(Threads REQUIRED)

target_include_directories(amp_native
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
    PRIVATE
        ${_FFTFREE_INCLUDE_DIRS}
)

# Allow configuring FFT division diagnostics at CMake configure time.
# If the user passes -DFFTDIV_ENABLE_DIAGNOSTIC=1 when invoking CMake,
# the AMP native target will be compiled with diagnostic prints enabled.
if(NOT DEFINED FFTDIV_ENABLE_DIAGNOSTIC)
    set(FFTDIV_ENABLE_DIAGNOSTIC 0 CACHE BOOL "Enable FFT division diagnostic prints" FORCE)
endif()
if(NOT DEFINED FFTDIV_TRACE_ENABLED)
    set(FFTDIV_TRACE_ENABLED 0 CACHE BOOL "Enable FFT division trace prints" FORCE)
endif()

target_compile_definitions(amp_native PRIVATE AMP_NATIVE_BUILD AMP_NATIVE_USE_FFTFREE FFTDIV_ENABLE_DIAGNOSTIC=${FFTDIV_ENABLE_DIAGNOSTIC} FFTDIV_TRACE_ENABLED=${FFTDIV_TRACE_ENABLED})
target_link_libraries(amp_native PRIVATE Threads::Threads ${_FFTFREE_TARGET})

# Optionally build a diagnostic-enabled variant of the native library that
# compiles FFT division nodes with diagnostic prints enabled. Default is OFF
# (quiet build). Enabling this will create a second shared library
# `amp_native_diagnostic` with `FFTDIV_ENABLE_DIAGNOSTIC` defined.
option(AMP_NATIVE_BUILD_DIAGNOSTIC "Build diagnostic variant of amp_native (with FFT division diagnostics enabled)" OFF)
if (AMP_NATIVE_BUILD_DIAGNOSTIC)
    add_library(amp_native_diagnostic SHARED ${_AMP_NATIVE_SOURCES})
    set_property(TARGET amp_native_diagnostic PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    set_property(TARGET amp_native_diagnostic PROPERTY MSVC_DEBUG_INFORMATION_FORMAT None)

    target_include_directories(amp_native_diagnostic
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${EIGEN3_INCLUDE_DIR}
        PRIVATE
            ${_FFTFREE_INCLUDE_DIRS}
    )

    target_compile_definitions(amp_native_diagnostic PRIVATE AMP_NATIVE_BUILD AMP_NATIVE_USE_FFTFREE FFTDIV_ENABLE_DIAGNOSTIC)
    target_link_libraries(amp_native_diagnostic PRIVATE Threads::Threads ${_FFTFREE_TARGET})

    if (_AMP_NATIVE_EXTRA_COMPILE_OPTIONS)
        target_compile_options(amp_native_diagnostic PRIVATE ${_AMP_NATIVE_EXTRA_COMPILE_OPTIONS})
    endif()
    if (_AMP_NATIVE_EXTRA_LINK_OPTIONS)
        target_link_options(amp_native_diagnostic PRIVATE ${_AMP_NATIVE_EXTRA_LINK_OPTIONS})
    endif()
    if (MSVC)
        target_compile_options(amp_native_diagnostic PRIVATE $<$<CONFIG:Release>:/Od> /FS)
        target_link_options(amp_native_diagnostic PRIVATE $<$<CONFIG:Release>:/DEBUG:NONE>)
    endif()

    set_target_properties(amp_native_diagnostic PROPERTIES
        OUTPUT_NAME "amp_native_diagnostic"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
endif()

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
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
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

    add_executable(test_fft_division_kpn
        tests/test_fft_division_kpn.cpp
    )
    target_include_directories(test_fft_division_kpn PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_fft_division_kpn PRIVATE amp_native)
    add_test(NAME test_fft_division_kpn COMMAND test_fft_division_kpn)

    add_executable(test_fft_division_node
        tests/test_fft_division_node.cpp
    )
    target_include_directories(test_fft_division_node PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_fft_division_node PRIVATE amp_native)
    # Run the FFT division node test quietly by default to avoid noisy diagnostics
    # during ctest runs. Developers can still run the binary directly for verbose
    # output (e.g. `test_fft_division_node --verbosity trace`).
    add_test(NAME test_fft_division_node COMMAND test_fft_division_node --quiet)

    # Minimal KPN harness to run a single FFTDivisionNode via the graph runtime.
    add_executable(kpn_run_fft_division
        tests/kpn_run_fft_division.cpp
    )
    target_include_directories(kpn_run_fft_division PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(kpn_run_fft_division PRIVATE amp_native)
    add_test(NAME kpn_run_fft_division COMMAND kpn_run_fft_division)

    # Diagnostic: compare Eigen FFT vs backend (fftfree) outputs
    add_executable(test_fft_backend_compare
        tests/fft_backend_compare.cpp
    )
    target_include_directories(test_fft_backend_compare PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${_FFTFREE_INCLUDE_DIRS}
    )
    target_link_libraries(test_fft_backend_compare PRIVATE amp_native ${_FFTFREE_TARGET})
    add_test(NAME test_fft_backend_compare COMMAND test_fft_backend_compare)

    add_executable(test_fft_backend_transform
        tests/test_fft_backend_transform.cpp
    )
    target_include_directories(test_fft_backend_transform PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${_FFTFREE_INCLUDE_DIRS}
    )
    target_link_libraries(test_fft_backend_transform PRIVATE ${_FFTFREE_TARGET})
    add_test(NAME test_fft_backend_transform COMMAND test_fft_backend_transform)

    add_executable(test_fft_backend_stream
        tests/test_fft_backend_stream.cpp
    )
    target_include_directories(test_fft_backend_stream PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${_FFTFREE_INCLUDE_DIRS}
    )
    target_link_libraries(test_fft_backend_stream PRIVATE ${_FFTFREE_TARGET})
    add_test(NAME test_fft_backend_stream COMMAND test_fft_backend_stream)

    add_executable(test_fft_windowing_roundtrip
        tests/test_fft_windowing_roundtrip.cpp
    )
    target_include_directories(test_fft_windowing_roundtrip PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests
        ${_FFTFREE_INCLUDE_DIRS}
    )
    target_link_libraries(test_fft_windowing_roundtrip PRIVATE amp_native ${_FFTFREE_TARGET})
    add_test(NAME test_fft_windowing_roundtrip COMMAND test_fft_windowing_roundtrip)

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
        kpn_run_fft_division
        test_fft_division_kpn
        test_fft_division_node
        test_fft_backend_compare
        test_fft_backend_transform
        test_fft_backend_stream
        test_fft_windowing_roundtrip
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

# Ensure that the demo and native test executables are built by a simple
# "cmake --build <builddir> --config <cfg>" invocation (add an ALL target).
set(_AMP_NATIVE_DEFAULT_BINARIES demo_kpn_native)
if(AMP_NATIVE_BUILD_TESTS)
    list(APPEND _AMP_NATIVE_DEFAULT_BINARIES ${_AMP_NATIVE_TEST_TARGETS})
endif()
# Create an aggregate ALL target that depends on the demo and all test binaries.
add_custom_target(build_native_binaries ALL DEPENDS ${_AMP_NATIVE_DEFAULT_BINARIES})

# Ensure the kpn_run_fft_division target is placed in the same runtime output
# directory as the other native binaries and is pulled into the aggregate
# `build_native_binaries` target. This guards against generator caching or
# ordering causing the target to be omitted from the aggregate build.
if(TARGET kpn_run_fft_division)
    set_target_properties(kpn_run_fft_division PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>"
    )
    add_dependencies(build_native_binaries kpn_run_fft_division)
endif()
