# Build flags and sanitizer helpers for FINUFFT

include(CheckCXXCompilerFlag)

# Probe/choose arch flags for MSVC using a tiny runtime probe
function(finufft_detect_msvc_arch out_var)
    try_run(
        RUN_RESULT_VAR
        COMPILE_RESULT_VAR
        ${CMAKE_BINARY_DIR}
        ${CMAKE_CURRENT_LIST_DIR}/CheckAVX.cpp
        COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
        RUN_OUTPUT_VARIABLE RUN_OUTPUT
    )
    if(RUN_OUTPUT MATCHES "AVX512")
        set(${out_var} "/arch:AVX512" PARENT_SCOPE)
    elseif(RUN_OUTPUT MATCHES "AVX2")
        set(${out_var} "/arch:AVX2" PARENT_SCOPE)
    elseif(RUN_OUTPUT MATCHES "AVX")
        set(${out_var} "/arch:AVX" PARENT_SCOPE)
    elseif(RUN_OUTPUT MATCHES "SSE2")
        set(${out_var} "/arch:SSE2" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

# Compute FINUFFT_ARCH_FLAGS if set to "native"
if(FINUFFT_ARCH_FLAGS STREQUAL "native")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(FINUFFT_ARCH_FLAGS -march=native CACHE STRING "" FORCE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        finufft_detect_msvc_arch(_finufft_arch)
        set(FINUFFT_ARCH_FLAGS "${_finufft_arch}" CACHE STRING "" FORCE)
    else()
        set(FINUFFT_ARCH_FLAGS "" CACHE STRING "" FORCE)
    endif()
endif()

# Helper: add only supported flags to target
function(finufft_target_add_flags_if_supported tgt scope)
    foreach(flag IN LISTS ARGN)
        # Avoid accidental list-splitting and cache the probe result
        string(REPLACE ";" "\\;" flag "${flag}")
        string(MD5 _key "${flag}-${CMAKE_CXX_COMPILER_ID}")
        set(_cache_var "FINUFFT_FLAG_OK_${_key}")
        if(NOT DEFINED ${_cache_var})
            check_cxx_compiler_flag("${flag}" ${_cache_var})
        endif()
        if(${_cache_var})
            target_compile_options(${tgt} ${scope} "${flag}")
        endif()
    endforeach()
endfunction()

# Common warning flags
set(_finufft_warn_gnuc
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unknown-pragmas
    -Wno-deprecated-declarations
)
set(_finufft_warn_msvc /W4)

# Optimization flags
set(_finufft_rel_gnuc
    -O3
    -funroll-loops
    -ffp-contract=fast
    -fno-math-errno
    -fno-signed-zeros
    -fno-trapping-math
    -fassociative-math
    -freciprocal-math
    -fmerge-all-constants
    -ftree-vectorize
    -fcx-limited-range
)
set(_finufft_rel_msvc /Ox /fp:contract /fp:except-)

# Debug flags
set(_finufft_dbg_gnuc -g3)
set(_finufft_dbg_msvc /Zi)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Sanitizer support (only if supported)
function(enable_asan target)
    if(FINUFFT_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            finufft_target_add_flags_if_supported(${target} PRIVATE /fsanitize=address /RTC1)
        else()
            finufft_target_add_flags_if_supported(
                ${target}
                PRIVATE
                -fsanitize=address
                -fsanitize=undefined
                -fsanitize=bounds-strict
            )
            target_link_options(${target} PRIVATE -fsanitize=address -fsanitize=undefined -fsanitize=bounds-strict)
        endif()
    endif()
endfunction()

# Expose helpers/sets as normal vars (no PARENT_SCOPE here)
set(FINUFFT_WARN_GNUC "${_finufft_warn_gnuc}")
set(FINUFFT_WARN_MSVC "${_finufft_warn_msvc}")
set(FINUFFT_REL_GNUC "${_finufft_rel_gnuc}")
set(FINUFFT_REL_MSVC "${_finufft_rel_msvc}")
set(FINUFFT_DBG_GNUC "${_finufft_dbg_gnuc}")
set(FINUFFT_DBG_MSVC "${_finufft_dbg_msvc}")
