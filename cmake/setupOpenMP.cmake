# setupOpenMP.cmake
#
# Purpose
#   Produce usable OpenMP imported targets (OpenMP::OpenMP_C and OpenMP::OpenMP_CXX)
#   on all major platforms. Prefer the system-provided OpenMP via CMake's
#   FindOpenMP if available; otherwise (or if requested) fall back to linking
#   against the Intel OpenMP runtime shipped with MATLAB (libiomp5).
#
#   This is useful when:
#     - AppleClang lacks a builtin OpenMP runtime
#     - You must link against the same OpenMP runtime as MATLAB MEX files
#     - FindOpenMP is unavailable/insufficient in your environment
#
# Usage
#   include(setupOpenMP)
#   ensure_openmp_targets(
#     [PREFER_MATLAB]                 # try MATLAB libiomp5 first, then system
#     [FORCE_MATLAB]                  # only use MATLAB libiomp5; do not try system
#     [DISABLE_SYSTEM]                # alias of FORCE_MATLAB (kept for clarity)
#     [MATLAB_ROOT <path>]            # override Matlab_ROOT_DIR (eg. /Applications/MATLAB_R2024a.app)
#     [OUT_VAR <varname>]             # var set to "system" or "matlab" indicating the provider
#   )
#
#   After calling, you can simply do:
#     target_link_libraries(my_target PRIVATE OpenMP::OpenMP_CXX)
#     # (and OpenMP::OpenMP_C for C targets if needed)
#
# Notes
#   - When using the MATLAB fallback, this module creates a real SHARED IMPORTED
#     library for OpenMP::OpenMP_CXX pointing at libiomp5, so dependents inherit
#     an rpath to find the runtime at execution time on UNIX-like systems.
#   - On Windows, IMPORTED_LOCATION is the DLL and, if available, IMPORTED_IMPLIB
#     is set for MSVC's link step.
#   - Compile options are added with language-guarded generator expressions so C
#     and C++ get the right flags.
#

include_guard(GLOBAL)

function(ensure_openmp_targets)
    set(options PREFER_MATLAB FORCE_MATLAB DISABLE_SYSTEM)
    set(oneValueArgs MATLAB_ROOT OUT_VAR)
    cmake_parse_arguments(EOT "${options}" "${oneValueArgs}" "" ${ARGN})

    # Early-out if targets already exist (e.g., user invoked find_package(OpenMP) already)
    if(TARGET OpenMP::OpenMP_CXX AND TARGET OpenMP::OpenMP_C)
        if(EOT_OUT_VAR)
            set(${EOT_OUT_VAR} "system" PARENT_SCOPE)
        endif()
        return()
    endif()

    # Decide whether to try system OpenMP first
    set(_try_system TRUE)
    if(EOT_FORCE_MATLAB OR EOT_DISABLE_SYSTEM)
        set(_try_system FALSE)
    endif()

    if(_try_system)
        # Prefer CMake's built-in FindOpenMP
        find_package(OpenMP QUIET COMPONENTS CXX C)
        if(OpenMP_CXX_FOUND)
            # CMake provides OpenMP::OpenMP_CXX / OpenMP::OpenMP_C targets.
            if(EOT_OUT_VAR)
                set(${EOT_OUT_VAR} "system" PARENT_SCOPE)
            endif()
            return()
        endif()
    endif()

    # --- MATLAB fallback ---
    # Determine MATLAB root
    set(_matlab_root "")
    if(EOT_MATLAB_ROOT)
        set(_matlab_root "${EOT_MATLAB_ROOT}")
    elseif(DEFINED Matlab_ROOT_DIR)
        set(_matlab_root "${Matlab_ROOT_DIR}")
    elseif(EOT_PREFER_MATLAB OR EOT_FORCE_MATLAB OR WIN32 OR APPLE)
        # Try to locate MATLAB automatically when reasonable
        find_package(Matlab QUIET)
        if(Matlab_FOUND)
            set(_matlab_root "${Matlab_ROOT_DIR}")
        endif()
    endif()

    if(NOT _matlab_root)
        if(EOT_FORCE_MATLAB OR EOT_DISABLE_SYSTEM)
            message(FATAL_ERROR "EnsureOpenMPTargets: MATLAB requested but not found. Provide MATLAB_ROOT.")
        else()
            message(
                FATAL_ERROR
                "EnsureOpenMPTargets: Could not configure OpenMP: system OpenMP not found and MATLAB not available."
            )
        endif()
    endif()

    set(_matlab_sys_os "${_matlab_root}/sys/os")

    if(WIN32)
        find_file(matlab_iomp5_dll NAMES libiomp5md.dll HINTS "${_matlab_sys_os}" PATH_SUFFIXES win64)
        find_library(matlab_iomp5_lib NAMES libiomp5md iomp5md iomp5 HINTS "${_matlab_sys_os}/win64")
        if(NOT matlab_iomp5_dll)
            message(
                FATAL_ERROR
                "EnsureOpenMPTargets: Could not find MATLAB's libiomp5md.dll in ${_matlab_sys_os}/win64"
            )
        endif()
    elseif(APPLE)
        # MATLAB ships iomp5 in {maca64,maci64}
        find_library(matlab_iomp5_lib NAMES iomp5 HINTS "${_matlab_sys_os}" PATH_SUFFIXES maca64 maci64)
        if(NOT matlab_iomp5_lib)
            message(
                FATAL_ERROR
                "EnsureOpenMPTargets: Could not find MATLAB's libiomp5.dylib (iomp5) in ${_matlab_sys_os}/maca64 or maci64"
            )
        endif()
    else()
        # Linux / UNIX
        find_library(matlab_iomp5_lib NAMES iomp5 HINTS "${_matlab_sys_os}" PATH_SUFFIXES glnxa64)
        if(NOT matlab_iomp5_lib)
            message(
                FATAL_ERROR
                "EnsureOpenMPTargets: Could not find MATLAB's libiomp5 (typically libiomp5.so) in ${_matlab_sys_os}/glnxa64"
            )
        endif()
    endif()

    # Threads for POSIX platforms
    find_package(Threads QUIET)

    # --- Create imported targets ---
    if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)

        if(WIN32)
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_LOCATION "${matlab_iomp5_dll}")
            if(matlab_iomp5_lib)
                set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_IMPLIB "${matlab_iomp5_lib}")
            endif()
        else()
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_LOCATION "${matlab_iomp5_lib}")
        endif()

        # Link Threads on POSIX if available
        if(NOT WIN32 AND TARGET Threads::Threads)
            target_link_libraries(OpenMP::OpenMP_CXX INTERFACE Threads::Threads)
        endif()

        # --- Compiler flags (language-scoped) ---
        if(MSVC)
            set(_omp_c_flags "$<$<COMPILE_LANGUAGE:C>:/openmp>")
            set(_omp_cxx_flags "$<$<COMPILE_LANGUAGE:CXX>:/openmp>")
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
            # AppleClang requires frontend forwarding
            set(_omp_c_flags "$<$<COMPILE_LANGUAGE:C>:-Xclang;-fopenmp>")
            set(_omp_cxx_flags "$<$<COMPILE_LANGUAGE:CXX>:-Xclang;-fopenmp>")
        else()
            set(_omp_c_flags "$<$<COMPILE_LANGUAGE:C>:-fopenmp>")
            set(_omp_cxx_flags "$<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")
        endif()

        target_compile_options(OpenMP::OpenMP_CXX INTERFACE ${_omp_c_flags} ${_omp_cxx_flags})

        # --- RPATH so dependents can find libiomp5 at runtime on UNIX ---
        if(APPLE OR UNIX)
            if(NOT WIN32)
                get_filename_component(_iomp_dir "${matlab_iomp5_lib}" DIRECTORY)
                # Propagate rpath via interface link options
                target_link_options(OpenMP::OpenMP_CXX INTERFACE "-Wl,-rpath,${_iomp_dir}")
            endif()
        endif()
    endif()

    if(NOT TARGET OpenMP::OpenMP_C)
        add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
        # Reuse the runtime/link/compile settings via CXX target
        target_link_libraries(OpenMP::OpenMP_C INTERFACE OpenMP::OpenMP_CXX)
    endif()

    if(EOT_OUT_VAR)
        set(${EOT_OUT_VAR} "matlab" PARENT_SCOPE)
    endif()
endfunction()
