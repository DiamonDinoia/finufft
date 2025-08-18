# setupOpenMP.cmake
#
# Purpose
#   Produce usable OpenMP imported targets (OpenMP::OpenMP_C and OpenMP::OpenMP_CXX)
#   on all major platforms. Prefer CMake's FindOpenMP (system OpenMP) if available;
#   otherwise (or if requested) fall back to MATLAB's OpenMP runtime (Intel's libiomp5,
#   and very rarely LLVM's libomp on macOS).
#
#   Useful when:
#     - AppleClang lacks a builtin OpenMP runtime/headers
#     - You must link against the same OpenMP runtime as MATLAB MEX files
#     - FindOpenMP is unavailable/insufficient in your environment
#
# Usage
#   include(setupOpenMP)
#   ensure_openmp_targets(
#     [PREFER_MATLAB]                 # try MATLAB runtime first, then system
#     [FORCE_MATLAB]                  # only use MATLAB runtime; do not try system
#     [DISABLE_SYSTEM]                # alias of FORCE_MATLAB
#     [MATLAB_ROOT <path>]            # override Matlab_ROOT_DIR (e.g., /Applications/MATLAB_R2024a.app)
#     [OUT_VAR <varname>]             # var set to "system" or "matlab" indicating the provider
#   )
#
# Notes
#   - For the MATLAB fallback, this module creates a real SHARED IMPORTED library
#     for OpenMP::OpenMP_CXX pointing at the located runtime so dependents inherit
#     an rpath to find it at execution time on UNIX-like systems.
#   - On Windows, IMPORTED_LOCATION is the DLL and IMPORTED_IMPLIB is set to the .lib.
#   - Compile options are added with language-guarded generator expressions.
#

include_guard(GLOBAL)

# Helper: locate system libomp (Homebrew paths on macOS, common defaults elsewhere)
function(_eot_find_system_libomp out_lib out_inc)
    set(_lib_candidates
        "$ENV{HOMEBREW_PREFIX}/opt/libomp/lib"
        "/opt/homebrew/opt/libomp/lib"
        "/usr/local/opt/libomp/lib"
        "/opt/homebrew/lib"
        "/usr/local/lib"
    )
    set(_inc_candidates
        "$ENV{HOMEBREW_PREFIX}/opt/libomp/include"
        "/opt/homebrew/opt/libomp/include"
        "/usr/local/opt/libomp/include"
        "/opt/homebrew/include"
        "/usr/local/include"
    )

    # Library can be named libomp (UNIX) or omp (bare name)
    find_library(_sys_omp_lib NAMES libomp omp HINTS ${_lib_candidates})
    find_path(_sys_omp_inc NAMES omp.h HINTS ${_inc_candidates})

    set(${out_lib} "${_sys_omp_lib}" PARENT_SCOPE)
    set(${out_inc} "${_sys_omp_inc}" PARENT_SCOPE)
endfunction()

# Helper: create imported targets wired to a specific runtime lib and headers.
# Used for UNIX-like (macOS/Linux) paths where a single shared object is enough.
function(_eot_create_imported_targets runtime_lib include_dir provider out_var_name)
    # Threads for POSIX platforms (optional)
    find_package(Threads QUIET)

    if(NOT TARGET OpenMP::OpenMP_CXX)
        add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
        set_target_properties(OpenMP::OpenMP_CXX PROPERTIES IMPORTED_LOCATION "${runtime_lib}")

        # Link Threads on POSIX if available
        if(NOT WIN32 AND TARGET Threads::Threads)
            target_link_libraries(OpenMP::OpenMP_CXX INTERFACE Threads::Threads)
        endif()

        # Compiler flags (language-scoped)
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

        # Provide headers if present (fixes 'omp.h' not found with AppleClang)
        if(NOT "${include_dir}" STREQUAL "" AND EXISTS "${include_dir}/omp.h")
            target_include_directories(OpenMP::OpenMP_CXX INTERFACE "${include_dir}")
        endif()

        # RPATH so dependents can find runtime at execution time on UNIX
        if(UNIX AND NOT WIN32)
            get_filename_component(_rt_dir "${runtime_lib}" DIRECTORY)
            target_link_options(OpenMP::OpenMP_CXX INTERFACE "-Wl,-rpath,${_rt_dir}")
        endif()
    endif()

    if(NOT TARGET OpenMP::OpenMP_C)
        add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
        target_link_libraries(OpenMP::OpenMP_C INTERFACE OpenMP::OpenMP_CXX)
    endif()

    if(NOT "${out_var_name}" STREQUAL "")
        set(${out_var_name} "${provider}" PARENT_SCOPE)
    endif()
endfunction()

function(ensure_openmp_targets)
    set(options PREFER_MATLAB FORCE_MATLAB DISABLE_SYSTEM)
    set(oneValueArgs MATLAB_ROOT OUT_VAR)
    cmake_parse_arguments(EOT "${options}" "${oneValueArgs}" "" ${ARGN})

    # Early-out if targets already exist (e.g., caller ran find_package(OpenMP))
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

    # --- Prefer CMake's built-in FindOpenMP ---
    if(_try_system)
        find_package(OpenMP QUIET COMPONENTS CXX C)
        if(OpenMP_CXX_FOUND)
            # On AppleClang, attach headers from Homebrew libomp if available
            if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
                _eot_find_system_libomp(_syslib _sysinc)
                if(_sysinc)
                    target_include_directories(OpenMP::OpenMP_CXX INTERFACE "${_sysinc}")
                    if(TARGET OpenMP::OpenMP_C)
                        target_include_directories(OpenMP::OpenMP_C INTERFACE "${_sysinc}")
                    endif()
                endif()
            endif()
            if(EOT_OUT_VAR)
                set(${EOT_OUT_VAR} "system" PARENT_SCOPE)
            endif()
            return()
        endif()
    endif()

    # --- Manual "system libomp" path for AppleClang (when FindOpenMP failed) ---
    if(_try_system AND APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        _eot_find_system_libomp(_sys_omp_lib _sys_omp_inc)
        if(_sys_omp_lib)
            _eot_create_imported_targets("${_sys_omp_lib}" "${_sys_omp_inc}" "system" EOT_OUT_VAR)
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
        # Windows: Intel libiomp5md.dll + import .lib
        find_file(matlab_iomp5_dll NAMES libiomp5md.dll HINTS "${_matlab_sys_os}" PATH_SUFFIXES win64)
        find_library(matlab_iomp5_lib NAMES libiomp5md iomp5md iomp5 HINTS "${_matlab_sys_os}/win64")
        if(NOT matlab_iomp5_dll OR NOT matlab_iomp5_lib)
            message(
                FATAL_ERROR
                "EnsureOpenMPTargets: Could not find MATLAB's libiomp5md.dll and/or import lib in ${_matlab_sys_os}/win64"
            )
        endif()

        # Create imported targets (Windows needs DLL + import lib per-config)
        if(NOT TARGET OpenMP::OpenMP_CXX)
            add_library(OpenMP::OpenMP_CXX SHARED IMPORTED)
            set_target_properties(
                OpenMP::OpenMP_CXX
                PROPERTIES
                    IMPORTED_LOCATION "${matlab_iomp5_dll}"
                    IMPORTED_LOCATION_DEBUG "${matlab_iomp5_dll}"
                    IMPORTED_LOCATION_RELEASE "${matlab_iomp5_dll}"
                    IMPORTED_IMPLIB "${matlab_iomp5_lib}"
                    IMPORTED_IMPLIB_DEBUG "${matlab_iomp5_lib}"
                    IMPORTED_IMPLIB_RELEASE "${matlab_iomp5_lib}"
            )
        endif()

        find_package(Threads QUIET)
        if(MSVC)
            target_compile_options(
                OpenMP::OpenMP_CXX
                INTERFACE "$<$<COMPILE_LANGUAGE:C>:/openmp>" "$<$<COMPILE_LANGUAGE:CXX>:/openmp>"
            )
        else()
            target_compile_options(
                OpenMP::OpenMP_CXX
                INTERFACE "$<$<COMPILE_LANGUAGE:C>:-fopenmp>" "$<$<COMPILE_LANGUAGE:CXX>:-fopenmp>"
            )
        endif()
        if(TARGET Threads::Threads)
            target_link_libraries(OpenMP::OpenMP_CXX INTERFACE Threads::Threads)
        endif()

        if(NOT TARGET OpenMP::OpenMP_C)
            add_library(OpenMP::OpenMP_C INTERFACE IMPORTED)
            target_link_libraries(OpenMP::OpenMP_C INTERFACE OpenMP::OpenMP_CXX)
        endif()

        if(EOT_OUT_VAR)
            set(${EOT_OUT_VAR} "matlab" PARENT_SCOPE)
        endif()
        return()
    elseif(APPLE)
        # macOS: MATLAB usually ships Intel iomp5 only on Intel Macs.
        # On Apple Silicon, MATLAB often does not bundle an OpenMP runtime.
        find_library(
            matlab_omp_lib
            NAMES iomp5 omp
            HINTS "${_matlab_sys_os}" "${_matlab_root}"
            PATH_SUFFIXES maca64 maci64 bin/maca64 extern/bin/maca64 sys/os/maca64
        )
        if(NOT matlab_omp_lib)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$")
                if(EOT_FORCE_MATLAB OR EOT_DISABLE_SYSTEM)
                    message(
                        FATAL_ERROR
                        "EnsureOpenMPTargets: MATLAB OpenMP runtime not found on Apple Silicon and system OpenMP disabled."
                    )
                endif()
                # Try manual system libomp as last resort
                _eot_find_system_libomp(_sys_omp_lib _sys_omp_inc)
                if(_sys_omp_lib)
                    _eot_create_imported_targets("${_sys_omp_lib}" "${_sys_omp_inc}" "system" EOT_OUT_VAR)
                    return()
                else()
                    message(
                        FATAL_ERROR
                        "EnsureOpenMPTargets: Neither system OpenMP nor MATLAB OpenMP runtime could be found."
                    )
                endif()
            else()
                message(
                    FATAL_ERROR
                    "EnsureOpenMPTargets: Could not find MATLAB's OpenMP runtime (iomp5/omp) in maci64/maca64."
                )
            endif()
        endif()

        # If we found a MATLAB runtime, still provide headers from system libomp if available.
        _eot_find_system_libomp(_sys_omp_lib _sys_omp_inc)
        _eot_create_imported_targets("${matlab_omp_lib}" "${_sys_omp_inc}" "matlab" EOT_OUT_VAR)
        return()
    else()
        # Linux / UNIX: MATLAB ships Intel libiomp5 in glnxa64
        find_library(matlab_iomp5_lib NAMES iomp5 HINTS "${_matlab_sys_os}" PATH_SUFFIXES glnxa64)
        if(NOT matlab_iomp5_lib)
            message(
                FATAL_ERROR
                "EnsureOpenMPTargets: Could not find MATLAB's libiomp5 (typically libiomp5.so) in ${_matlab_sys_os}/glnxa64"
            )
        endif()
        _eot_create_imported_targets("${matlab_iomp5_lib}" "" "matlab" EOT_OUT_VAR)
        return()
    endif()
endfunction()
