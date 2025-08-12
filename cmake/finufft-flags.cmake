# Build flags and sanitizer helpers for FINUFFT

# Map "native" to compiler-specific architecture flags; emulate for MSVC
function(finufft_detect_msvc_arch out_var)
    message(STATUS "Checking for AVX, AVX512 and SSE support")
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
    elseif(RUN_OUTPUT MATCHES "SSE")
        set(${out_var} "/arch:SSE" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${${out_var}}")
endfunction()

if(FINUFFT_ARCH_FLAGS STREQUAL "native")
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(FINUFFT_ARCH_FLAGS -march=native CACHE STRING "" FORCE)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        finufft_detect_msvc_arch(_finufft_arch)
        set(FINUFFT_ARCH_FLAGS "${_finufft_arch}" CACHE STRING "" FORCE)
    else()
        # Other compilers don't support -march=native
        set(FINUFFT_ARCH_FLAGS "" CACHE STRING "" FORCE)
    endif()
endif()

# Common warning flags applied to all targets
set(FINUFFT_WARNING_FLAGS
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall;-Wextra;-Wpedantic;-Wno-unknown-pragmas;-Wno-deprecated-declarations>
)

# Optimization flags
set(FINUFFT_CXX_FLAGS_RELEASE
    $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-funroll-loops;-ffp-contract=fast;-fno-math-errno;-fno-signed-zeros;-fno-trapping-math;-fassociative-math;-freciprocal-math;-fmerge-all-constants;-ftree-vectorize;-fimplicit-constexpr;-fcx-limited-range;-O3>
    $<$<CXX_COMPILER_ID:MSVC>:/Ox;/fp:contract;/fp:except->
)

set(FINUFFT_CXX_FLAGS_DEBUG $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-g3> $<$<CXX_COMPILER_ID:MSVC>:/Zi>)

set(FINUFFT_CXX_FLAGS_RELWITHDEBINFO ${FINUFFT_CXX_FLAGS_RELEASE} ${FINUFFT_CXX_FLAGS_DEBUG})

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Sanitizer support
function(enable_asan target)
    if(FINUFFT_ENABLE_SANITIZERS)
        if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE /fsanitize=address /RTC1)
        else()
            target_compile_options(${target} PRIVATE -fsanitize=address -fsanitize=undefined -fsanitize=bounds-strict)
            target_link_options(${target} PRIVATE -fsanitize=address -fsanitize=undefined -fsanitize=bounds-strict)
        endif()
    endif()
endfunction()
