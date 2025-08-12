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
option(FINUFFT_STATIC_LINKING "If ON builds the static finufft library, if OFF build a shared finufft library." ON)
option(FINUFFT_POSITION_INDEPENDENT_CODE "Whether to build the finufft library with position independent code (-fPIC). This forced ON when FINUFFT_SHARED_LINKING is ON." ON)
option(FINUFFT_BUILD_DEVEL "Whether to build development executables" OFF)
option(FINUFFT_BUILD_EXAMPLES "Whether to build the FINUFFT examples" OFF)
option(FINUFFT_BUILD_TESTS "Whether to build the FINUFFT tests" OFF)
option(FINUFFT_USE_DUCC0 "Whether to use DUCC0 (instead of FFTW) for CPU FFTs" OFF)
option(FINUFFT_BUILD_DOCS "Whether to build the FINUFFT documentation" OFF)
# if FINUFFT_USE_DUCC0 is ON, the following options are ignored
set(FINUFFT_FFTW_LIBRARIES "DEFAULT" CACHE STRING "Specify a custom FFTW library")
set(FINUFFT_FFTW_SUFFIX "DEFAULT" CACHE STRING "Suffix for FFTW libraries (e.g. OpenMP, Threads etc.) defaults to empty string if OpenMP is disabled, else uses OpenMP. Ignored if DUCC0 is used.")
# if FINUFFT_USE_CPU is OFF, the following options are ignored
set(FINUFFT_ARCH_FLAGS "native" CACHE STRING "Compiler flags for specifying target architecture, defaults to -march=native")
# sphinx tag (don't remove): @cmake_opts_end
cmake_dependent_option(FINUFFT_ENABLE_INSTALL "Disable installation in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_STATIC_LINKING "Disable static libraries in the case of python builds" ON "NOT FINUFFT_BUILD_PYTHON" OFF)
cmake_dependent_option(FINUFFT_SHARED_LINKING "Shared should be the opposite of static linking" ON "NOT FINUFFT_STATIC_LINKING" OFF)
# gersemi: on

# When building shared libraries, we need to build with -fPIC in all cases
if(FINUFFT_SHARED_LINKING)
    set(FINUFFT_POSITION_INDEPENDENT_CODE ON)
endif()

include(cmake/utils.cmake)
include(cmake/finufft-flags.cmake)
include(cmake/finufft-targets.cmake)

set(CPM_DOWNLOAD_VERSION "0.40.5" CACHE STRING "Version of CPM.cmake to use")
set(FFTW_VERSION "3.3.10" CACHE STRING "Version of FFTW to use")
set(XTL_VERSION "0.7.7" CACHE STRING "Version of xtl to use")
set(XSIMD_VERSION "13.2.0" CACHE STRING "Version of xsimd to use")
set(DUCC0_VERSION "ducc0_0_36_0" CACHE STRING "Version of ducc0 to use")
set(CUDA11_CCCL_VERSION "2.8.5" CACHE STRING "Version of FINUFFT-cccl for cuda 11 to use")
set(CUDA12_CCCL_VERSION "3.0.1" CACHE STRING "Version of FINUFFT-cccl for cuda 12 to use")
mark_as_advanced(CPM_DOWNLOAD_VERSION)
mark_as_advanced(FFTW_VERSION)
mark_as_advanced(XTL_VERSION)
mark_as_advanced(XSIMD_VERSION)
mark_as_advanced(DUCC0_VERSION)
mark_as_advanced(CUDA11_CCCL_VERSION)
mark_as_advanced(CUDA12_CCCL_VERSION)

include(cmake/setupCPM.cmake)

if(FINUFFT_USE_CUDA)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        message(
            WARNING
            "FINUFFT WARNING: No CUDA architecture supplied via '-DCMAKE_CUDA_ARCHITECTURES=...', defaulting to 'native' See: https://developer.nvidia.com/cuda-gpus for more details on what architecture to supply."
        )
        detect_cuda_architecture()
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED COMPONENTS cudart cufft)
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

set(INSTALL_TARGETS "")
add_subdirectory(src)

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
message(STATUS "  FINUFFT_USE_CPU: ${FINUFFT_USE_CPU}")
message(STATUS "  FINUFFT_USE_CUDA: ${FINUFFT_USE_CUDA}")
message(STATUS "  FINUFFT_USE_OPENMP: ${FINUFFT_USE_OPENMP}")
message(STATUS "  FINUFFT_STATIC_LINKING: ${FINUFFT_STATIC_LINKING}")
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

if(FINUFFT_ENABLE_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS ${INSTALL_TARGETS} PUBLIC_HEADER)
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
