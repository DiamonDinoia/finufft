cmake_minimum_required(VERSION 3.24...3.31)
project(FINUFFT VERSION 2.5.0 LANGUAGES C CXX)

# Use modern CMake policies by default
cmake_policy(VERSION 3.24)

include(CMakeDependentOption)

# gersemi: off
# All options go here sphinx tag (don't remove): @cmake_opts_start
option(FINUFFT_BUILD_FORTRAN "Whether to build the FINUFFT Fortran examples" OFF)
option(FINUFFT_BUILD_MATLAB "Whether to build the FINUFFT Matlab interface" OFF)
option(FINUFFT_BUILD_PYTHON "Whether the Python wrapper should be built." OFF)
option(FINUFFT_ENABLE_SANITIZERS "Whether to enable sanitizers, only effective for Debug configuration." OFF)
option(FINUFFT_USE_OPENMP "Whether to use OpenMP for parallelization. If disabled, the finufft library will be single threaded. This does not affect the choice of FFTW library." ON)
option(FINUFFT_USE_CPU "Whether to build the ordinary FINUFFT library (libfinufft)." ON)
option(FINUFFT_USE_CUDA "Whether to build CUDA accelerated FINUFFT library (libcufinufft). This is completely independent of the main FINUFFT library" OFF)
# Replace legacy BUILD_SHARED_LIBS toggle with explicit FINUFFT_STATIC_LINKING user option
option(FINUFFT_STATIC_LINKING "If ON builds the static finufft library, if OFF build a shared finufft library." ON)
option(FINUFFT_POSITION_INDEPENDENT_CODE "Whether to build with position independent code (-fPIC). Forced ON if shared." ON)
option(FINUFFT_BUILD_DEVEL "Whether to build development executables" OFF)
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_USE_DUCC0 "Whether to use DUCC0 (instead of FFTW) for CPU FFTs" OFF)
option(FINUFFT_BUILD_DOCS "Whether to build the FINUFFT documentation" OFF)
# Windows fallback (OFF = minimal exports; ON only if headers lack dllexport annotations for CUDA lib)
option(FINUFFT_WINDOWS_EXPORT_ALL "Export all symbols on Windows (fallback for missing dllexport annotations)" OFF)

# if FINUFFT_USE_DUCC0 is ON, the following options are ignored
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_FFTW_SUFFIX "DEFAULT" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.) defaults to empty string if OpenMP is disabled, else uses OpenMP. Ignored if DUCC0 is used.")
# if FINUFFT_USE_CPU is OFF, the following options are ignored
set(FINUFFT_ARCH_FLAGS "native" CACHE STRING "Compiler flags for specifying target architecture, defaults to -march=native")
# sphinx tag (don't remove): @cmake_opts_end
cmake_dependent_option(FINUFFT_ENABLE_INSTALL "Disable installation in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
# gersemi: on

# Map FINUFFT_STATIC_LINKING -> CMake's BUILD_SHARED_LIBS
if(FINUFFT_STATIC_LINKING)
    set(BUILD_SHARED_LIBS OFF)
else()
    set(BUILD_SHARED_LIBS ON)
endif()

# PIC: force ON when building shared; otherwise honor the option variable only per-target
if(NOT FINUFFT_STATIC_LINKING)
    set(FINUFFT_POSITION_INDEPENDENT_CODE ON)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE})

include(cmake/utils.cmake)
include(cmake/finufft-flags.cmake)
include(cmake/finufft-targets.cmake)

# Dependency versions for CPM
set(CPM_DOWNLOAD_VERSION "0.40.5" CACHE STRING "Version of CPM.cmake to use")
set(FFTW_VERSION "3.3.10" CACHE STRING "Version of FFTW to use")
set(XTL_VERSION "0.7.7" CACHE STRING "Version of xtl to use")
set(XSIMD_VERSION "13.2.0" CACHE STRING "Version of xsimd to use")
set(DUCC0_VERSION "ducc0_0_36_0" CACHE STRING "Version of ducc0 to use")
set(CUDA11_CCCL_VERSION "2.8.5" CACHE STRING "Version of FINUFFT-cccl for cuda 11 to use")
set(CUDA12_CCCL_VERSION "3.0.1" CACHE STRING "Version of FINUFFT-cccl for cuda 12 to use")
mark_as_advanced(
    CPM_DOWNLOAD_VERSION
    FFTW_VERSION
    XTL_VERSION
    XSIMD_VERSION
    DUCC0_VERSION
    CUDA11_CCCL_VERSION
    CUDA12_CCCL_VERSION
)

include(cmake/setupCPM.cmake)

if(FINUFFT_USE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(WARNING "No CMAKE_CUDA_ARCHITECTURES set; attempting auto-detect")
        detect_cuda_architecture()
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart cufft)
    # Ensure CCCL is available for CUDA targets (defines CCCL::CCCL)
    include(${PROJECT_SOURCE_DIR}/cmake/setupCCCL.cmake)
endif()

if(PROJECT_IS_TOP_LEVEL)
    include(CTest)
    if(FINUFFT_BUILD_TESTS)
        enable_testing()
    endif()
    if(FINUFFT_BUILD_DOCS)
        include(cmake/setupSphinx.cmake)
    endif()
endif()

add_subdirectory(src)

# Build CUDA library when requested so its target is available to tests/examples
if(FINUFFT_USE_CUDA)
    add_subdirectory(src/cuda)
endif()

if(BUILD_TESTING AND FINUFFT_BUILD_TESTS AND FINUFFT_USE_CPU)
    add_subdirectory(test)
    add_subdirectory(perftest)
endif()
if(BUILD_TESTING AND FINUFFT_BUILD_TESTS AND FINUFFT_USE_CUDA)
    add_subdirectory(test/cuda)
    add_subdirectory(perftest/cuda)
endif()
if(FINUFFT_BUILD_EXAMPLES AND FINUFFT_USE_CPU)
    add_subdirectory(examples)
endif()
if(FINUFFT_BUILD_EXAMPLES AND FINUFFT_USE_CUDA)
    add_subdirectory(examples/cuda)
endif()
if(FINUFFT_BUILD_FORTRAN)
    enable_language(Fortran)
    add_subdirectory(fortran)
endif()
if(FINUFFT_BUILD_MATLAB)
    add_subdirectory(matlab)
endif()
if(FINUFFT_BUILD_DEVEL)
    add_subdirectory(devel)
endif()
if(FINUFFT_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# gersemi: off
message(STATUS "FINUFFT configuration summary:")
message(STATUS "  CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "  BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}")
message(STATUS "  FINUFFT_USE_CPU: ${FINUFFT_USE_CPU}")
message(STATUS "  FINUFFT_USE_CUDA: ${FINUFFT_USE_CUDA}")
message(STATUS "  FINUFFT_USE_OPENMP: ${FINUFFT_USE_OPENMP}")
message(STATUS "  FINUFFT_POSITION_INDEPENDENT_CODE: ${FINUFFT_POSITION_INDEPENDENT_CODE}")
message(STATUS "  FINUFFT_ENABLE_INSTALL: ${FINUFFT_ENABLE_INSTALL}")
message(STATUS "  FINUFFT_BUILD_EXAMPLES: ${FINUFFT_BUILD_EXAMPLES}")
message(STATUS "  FINUFFT_BUILD_TESTS: ${FINUFFT_BUILD_TESTS}")
message(STATUS "  FINUFFT_BUILD_FORTRAN: ${FINUFFT_BUILD_FORTRAN}")
message(STATUS "  FINUFFT_BUILD_MATLAB: ${FINUFFT_BUILD_MATLAB}")
message(STATUS "  FINUFFT_BUILD_PYTHON: ${FINUFFT_BUILD_PYTHON}")
message(STATUS "  FINUFFT_ENABLE_SANITIZERS: ${FINUFFT_ENABLE_SANITIZERS}")
message(STATUS "  FINUFFT_FFTW_SUFFIX: ${FINUFFT_FFTW_SUFFIX}")
message(STATUS "  FINUFFT_FFTW_LIBRARIES: ${FINUFFT_FFTW_LIBRARIES}")
message(STATUS "  FINUFFT_ARCH_FLAGS: ${FINUFFT_ARCH_FLAGS}")
message(STATUS "  FINUFFT_USE_DUCC0: ${FINUFFT_USE_DUCC0}")
message(STATUS "  CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
# gersemi: on

# -------- Installable package (Config + Targets) --------
if(FINUFFT_ENABLE_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    # Install headers (library targets also install PUBLIC_HEADER below)
    install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

    # Export targets from src/
    install(EXPORT finufftTargets NAMESPACE finufft:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft)

    # Config + version files
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/finufftConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )

    configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/cmake/finufftConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/finufftConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft
    )

    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/finufftConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/finufftConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/finufft
    )

    # Extra data
    install(FILES ${PROJECT_SOURCE_DIR}/LICENSE DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/licenses/finufft)

    if(FINUFFT_USE_CPU)
        install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/examples
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft
            PATTERN "CMakeLists.txt" EXCLUDE
            PATTERN "README" EXCLUDE
            PATTERN "examples/cuda" EXCLUDE
        )
        if(FINUFFT_BUILD_FORTRAN)
            install(
                DIRECTORY ${PROJECT_SOURCE_DIR}/fortran/examples
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/fortran
            )
            install(FILES ${PROJECT_SOURCE_DIR}/include/finufft.fh DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
        endif()
    endif()

    if(FINUFFT_USE_CUDA)
        install(
            DIRECTORY ${PROJECT_SOURCE_DIR}/examples/cuda
            DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/finufft/examples
            PATTERN "README" EXCLUDE
            PATTERN "CMakeLists.txt" EXCLUDE
        )
    endif()
endif()
