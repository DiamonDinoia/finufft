# Resolve FFTW and expose a stable interface target: finufft::fftw
# - Finds a system FFTW (config or module)
# - Or builds static FFTW with CPM
# - Exposes a build-only INTERFACE target finufft::fftw
# - Updates FINUFFT_FFTW_LIBRARIES cache var to the resolved targets (never "DEFAULT")

# Bring in the FindFFTW module (egpbos/findFFTW)
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

# Helper: push a flag into a list if the target exists
function(_finufft_append_if_target out_list)
    foreach(_t IN LISTS ARGN)
        if(TARGET "${_t}")
            list(APPEND _acc "${_t}")
        endif()
    endforeach()
    set(${out_list} "${_acc}" PARENT_SCOPE)
endfunction()

# 1) Try system FFTW via find_package
set(_fftw_targets_resolved)
find_package(FFTW QUIET) # may define IMPORTED targets like FFTW::Double, FFTW::Float, etc.

if(FFTW_FOUND)
    # Base float/double targets
    list(APPEND _fftw_targets_resolved FFTW::Float FFTW::Double)

    # Pick flavor (OpenMP / Threads / suffix override)
    if(FINUFFT_FFTW_SUFFIX STREQUAL "DEFAULT")
        if(FINUFFT_USE_OPENMP)
            list(APPEND _fftw_targets_resolved FFTW::FloatOpenMP FFTW::DoubleOpenMP)
        endif()
    elseif(FINUFFT_FFTW_SUFFIX STREQUAL "THREADS")
        list(APPEND _fftw_targets_resolved FFTW::FloatThreads FFTW::DoubleThreads)
    else()
        # user override, e.g. "OMP" or "Threads"
        list(APPEND _fftw_targets_resolved "FFTW::Float${FINUFFT_FFTW_SUFFIX}" "FFTW::Double${FINUFFT_FFTW_SUFFIX}")
    endif()

    # Keep only those that actually exist
    _finufft_append_if_target(_fftw_targets_resolved ${_fftw_targets_resolved})
endif()

# 2) If not resolved, build with CPM (static)
if(NOT _fftw_targets_resolved)
    if(FINUFFT_FFTW_SUFFIX STREQUAL "THREADS")
        set(_finufft_use_threads ON)
    else()
        set(_finufft_use_threads OFF)
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
        "ENABLE_THREADS ${_finufft_use_threads}"
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
        "ENABLE_THREADS ${_finufft_use_threads}"
        "ENABLE_OPENMP ${FINUFFT_USE_OPENMP}"
    )

    # Base libs
    set(_fftw_targets_resolved fftw3 fftw3f)

    # Threading flavors
    if(_finufft_use_threads)
        list(APPEND _fftw_targets_resolved fftw3_threads fftw3f_threads)
    elseif(FINUFFT_USE_OPENMP)
        list(APPEND _fftw_targets_resolved fftw3_omp fftw3f_omp)
    endif()

    # Tidy properties
    foreach(_t IN LISTS _fftw_targets_resolved)
        if(TARGET "${_t}")
            set_target_properties(
                ${_t}
                PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" POSITION_INDEPENDENT_CODE TRUE
            )
        endif()
    endforeach()

    # Headers for consumers during build of finufft
    if(TARGET fftw3)
        target_include_directories(fftw3 PUBLIC $<BUILD_INTERFACE:${fftw3_SOURCE_DIR}/api>)
    endif()
endif()

# Final sanity: ensure we have at least the base double/float libs
if(NOT _fftw_targets_resolved)
    message(FATAL_ERROR "FFTW could not be resolved. Install FFTW or use -DFINUFFT_FFTW_LIBRARIES=DOWNLOAD.")
endif()

# Create a stable INTERFACE target to link against in this build
if(NOT TARGET finufft_fftw)
    add_library(finufft_fftw INTERFACE)
    add_library(finufft::fftw ALIAS finufft_fftw)
    target_link_libraries(finufft_fftw INTERFACE ${_fftw_targets_resolved})
endif()

# Update cache var so nobody sees 'DEFAULT' downstream
set(FINUFFT_FFTW_LIBRARIES "${_fftw_targets_resolved}" CACHE STRING "Resolved FFTW targets" FORCE)
unset(_fftw_targets_resolved)
unset(_finufft_use_threads)
