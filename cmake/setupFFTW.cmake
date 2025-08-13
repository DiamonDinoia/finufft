# Resolve FFTW and expose a stable interface target finufft::fftw.
# Keeps FFTW private to finufft for install, but available for build-tree consumers.

if(TARGET finufft_fftw)
    return()
endif()

# Ensure CPM is available
if(NOT COMMAND CPMAddPackage)
    include(${PROJECT_SOURCE_DIR}/cmake/setupCPM.cmake)
endif()

# Bring in the helper findFFTW module (no install impact)
CPMAddPackage(
    NAME
    findfftw
    GIT_REPOSITORY
    https://github.com/egpbos/findFFTW.git
    GIT_TAG
    master
    EXCLUDE_FROM_ALL
    YES
    GIT_SHALLOW
    YES
)
list(APPEND CMAKE_MODULE_PATH "${findfftw_SOURCE_DIR}")

# Try system FFTW first
find_package(FFTW QUIET)

set(_fftw_targets "")
if(FFTW_FOUND)
    set(_want FFTW::Float FFTW::Double)
    if(FINUFFT_USE_OPENMP)
        list(APPEND _want FFTW::FloatOpenMP FFTW::DoubleOpenMP)
    endif()
    foreach(_t IN LISTS _want)
        if(TARGET "${_t}")
            list(APPEND _fftw_targets "${_t}")
        endif()
    endforeach()
endif()

# Fallback: build static FFTW via CPM
if(NOT _fftw_targets)
    if(FINUFFT_FFTW_SUFFIX STREQUAL "THREADS")
        set(_use_threads ON)
    else()
        set(_use_threads OFF)
    endif()

    CPMAddPackage(
        NAME
        fftw3
        URL
        "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
        URL_HASH
        "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
        EXCLUDE_FROM_ALL
        YES
        OPTIONS
        "ENABLE_FLOAT OFF"
        "BUILD_TESTS OFF"
        "BUILD_SHARED_LIBS OFF"
        "ENABLE_SSE2 ON"
        "ENABLE_AVX ON"
        "ENABLE_AVX2 ON"
        "ENABLE_THREADS ${_use_threads}"
        "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
    )
    CPMAddPackage(
        NAME
        fftw3f
        URL
        "http://www.fftw.org/fftw-${FFTW_VERSION}.tar.gz"
        URL_HASH
        "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
        EXCLUDE_FROM_ALL
        YES
        OPTIONS
        "ENABLE_FLOAT ON"
        "BUILD_TESTS OFF"
        "BUILD_SHARED_LIBS OFF"
        "ENABLE_SSE2 ON"
        "ENABLE_AVX ON"
        "ENABLE_AVX2 ON"
        "ENABLE_THREADS ${_use_threads}"
        "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
    )

    set(_fftw_targets fftw3 fftw3f)
    if(_use_threads)
        list(APPEND _fftw_targets fftw3_threads fftw3f_threads)
    elseif(FINUFFT_USE_OPENMP)
        list(APPEND _fftw_targets fftw3_omp fftw3f_omp)
    endif()

    # Public includes for the built fftw
    if(TARGET fftw3)
        target_include_directories(fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)
    endif()
endif()

if(NOT _fftw_targets)
    message(FATAL_ERROR "FFTW could not be resolved. Install FFTW or enable CPM fallback.")
endif()

# Build-only interface to FFTW (never exported/installed)
add_library(finufft_fftw INTERFACE)
target_link_libraries(finufft_fftw INTERFACE ${_fftw_targets})
add_library(finufft::fftw ALIAS finufft_fftw)

# For status printing
set(FINUFFT_FFTW_LIBRARIES "${_fftw_targets}" CACHE STRING "Resolved FFTW targets" FORCE)

unset(_fftw_targets)
unset(_want)
unset(_use_threads)
