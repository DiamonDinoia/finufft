# MexCUDA.cmake — CMake module to build CUDA MEX via the `mexcuda` executable
#
# Design
# - Creates a pair of custom targets:
#     <name>           : a phony target you can depend on or invoke
#     <name>__build    : actually builds the MEX by running the `mexcuda` executable directly
# - You pass includes/defines/link options/etc. as arguments to the function (no need to call
#   target_* on <name>). This avoids INTERFACE/custom-target "non-compilable" issues.
# - Still supports LINK_TO CMake targets or absolute library files (their linker files are expanded).
# - `CUDA_ARCHITECTURES` can be passed (digits like 75;80;86) or we fall back to CMAKE_CUDA_ARCHITECTURES.
# - `GPU_ARCH` may also be passed (either "sm_80" or just "80"). If both are given, GPU_ARCH wins.
# - Exposes MEX_OUTPUT, MEX_OUTDIR, MEX_OUTPUT_NAME properties on the phony <name> target.
# - Provides make_mex_self_contained(<name>) which creates an extra <name>__fixrpath target that
#   depends on <name> and adjusts rpaths on the built file (works with custom targets).
#
# Usage:
#   include("${CMAKE_SOURCE_DIR}/cmake/MexCUDA.cmake")
#   matlab_add_mexcuda(
#     NAME <target>
#     SRC <.cu files...>
#     [OUTPUT_NAME <basename>]
#     [R2018a]
#     [INCLUDES <dirs...>]
#     [DEFINES <macros...>]
#     [LINK_OPTIONS <opts...>]
#     [MEX_FLAGS <flags...>]
#     [LINK_TO <cmake_targets_or_full_libs...>]
#     [GPU_ARCH <sm_80;sm_75;... | 80;75;...>]
#     [CUDA_ARCHITECTURES <80;75;...>]  # falls back to CMAKE_CUDA_ARCHITECTURES if not set
#   )
#
# Example:
#   matlab_add_mexcuda(
#     NAME        cufinufft_mex
#     SRC         "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu"
#     OUTPUT_NAME "cufinufft"
#     R2018a
#     INCLUDES    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
#                 $<BUILD_INTERFACE:${Matlab_GPU_INCLUDE_DIR}>
#                 $<BUILD_INTERFACE:${Matlab_INCLUDE_DIRS}>
#     DEFINES     R2008OO
#     LINK_TO     cufinufft Matlab::mex Matlab::mx CUDA::cudart
#     LINK_OPTIONS $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:-static-libstdc++>
#                  $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:-static-libgcc>
#     CUDA_ARCHITECTURES 75;80;86
#   )
#   make_mex_self_contained(cufinufft_mex)
#
cmake_minimum_required(VERSION 3.18)

if(COMMAND matlab_add_mexcuda)
    # Prevent redefinition if included twice.
    return()
endif()

# MATLAB root is required to locate the mexcuda executable.
find_package(Matlab REQUIRED) # (No need for MAIN_PROGRAM when invoking mexcuda directly)

# Helper: locate the mexcuda executable under Matlab_ROOT_DIR using GLOB_RECURSE
function(_mexcuda_locate_exe OUT_VAR)
    if(NOT Matlab_ROOT_DIR)
        message(FATAL_ERROR "[mexcuda] Matlab_ROOT_DIR is not set; cannot locate mexcuda.")
    endif()

    # Search typical locations (<matlabroot>/bin and <matlabroot>/bin/<arch>)
    file(
        GLOB_RECURSE _CANDIDATES
        LIST_DIRECTORIES FALSE
        "${Matlab_ROOT_DIR}/bin/mexcuda*"
        "${Matlab_ROOT_DIR}/bin/*/mexcuda*"
    )

    # Filter to platform-appropriate executable names
    if(WIN32)
        list(FILTER _CANDIDATES INCLUDE REGEX ".*[/\\]mexcuda(\\.bat|\\.exe)$")
    else()
        list(FILTER _CANDIDATES INCLUDE REGEX ".*[/\\]mexcuda$")
    endif()

    if(NOT _CANDIDATES)
        message(FATAL_ERROR "[mexcuda] Could not find 'mexcuda' under Matlab_ROOT_DIR='${Matlab_ROOT_DIR}'.")
    endif()

    # Prefer an architecture subdir (e.g., glnxa64/win64/maci64) if present
    set(_BEST "")
    foreach(_c IN LISTS _CANDIDATES)
        if(_c MATCHES "/bin/[^/\\]+/mexcuda(\\.bat|\\.exe)?$")
            set(_BEST "${_c}")
            break()
        endif()
    endforeach()
    if(NOT _BEST)
        list(GET _CANDIDATES 0 _BEST)
    endif()

    set(${OUT_VAR} "${_BEST}" PARENT_SCOPE)
endfunction()

# Add a CUDA MEX build driven by mexcuda (invoked directly)
function(matlab_add_mexcuda)
    set(_opts R2018a)
    set(_one NAME OUTPUT_NAME)
    set(_multi
        SRC
        INCLUDES
        DEFINES
        LINK_OPTIONS
        MEX_FLAGS
        LINK_TO
        GPU_ARCH
        CUDA_ARCHITECTURES
    )
    cmake_parse_arguments(MAM "${_opts}" "${_one}" "${_multi}" ${ARGN})

    if(NOT MAM_NAME)
        message(FATAL_ERROR "matlab_add_mexcuda: NAME is required")
    endif()
    if(NOT MAM_SRC)
        message(FATAL_ERROR "matlab_add_mexcuda: at least one SRC is required")
    endif()

    set(_target ${MAM_NAME})
    set(_output_name ${MAM_OUTPUT_NAME})
    if(NOT _output_name)
        set(_output_name ${_target})
    endif()

    set(_outdir "${CMAKE_CURRENT_BINARY_DIR}/mex")
    file(MAKE_DIRECTORY "${_outdir}")

    # Absolute sources
    set(_abs_srcs)
    foreach(_s IN LISTS MAM_SRC)
        get_filename_component(_abs "${_s}" ABSOLUTE)
        list(APPEND _abs_srcs "${_abs}")
    endforeach()

    # Resolve LINK_TO: CMake targets → their linker files; absolute files passed through
    set(_link_inputs)
    set(_dep_targets)
    foreach(_lib IN LISTS MAM_LINK_TO)
        if(TARGET ${_lib})
            list(APPEND _link_inputs "$<SHELL_PATH:$<TARGET_LINKER_FILE:${_lib}>>")
            list(APPEND _dep_targets ${_lib})
        elseif(IS_ABSOLUTE "${_lib}")
            list(APPEND _link_inputs "$<SHELL_PATH:${_lib}>")
        else()
            message(
                WARNING
                "matlab_add_mexcuda(${_target}): LINK_TO entry '${_lib}' is neither a CMake target nor an absolute path; ignored."
            )
        endif()
    endforeach()

    # Includes/defines/link options (allow generator expressions)
    set(_inc_flags)
    foreach(_inc IN LISTS MAM_INCLUDES)
        list(APPEND _inc_flags "-I$<SHELL_PATH:${_inc}>")
    endforeach()

    set(_def_flags)
    foreach(_d IN LISTS MAM_DEFINES)
        list(APPEND _def_flags "-D${_d}")
    endforeach()

    set(_linkopt_flags ${MAM_LINK_OPTIONS})
    set(_mexflags ${MAM_MEX_FLAGS})

    # GPU arch flags
    set(_gpu_flags)
    if(MAM_GPU_ARCH)
        foreach(_g IN LISTS MAM_GPU_ARCH)
            if(_g MATCHES "^sm_[0-9]+")
                list(APPEND _gpu_flags "-gpu=${_g}")
            else()
                list(APPEND _gpu_flags "-gpu=sm_${_g}")
            endif()
        endforeach()
    else()
        set(_archs ${MAM_CUDA_ARCHITECTURES})
        if(NOT _archs AND CMAKE_CUDA_ARCHITECTURES)
            set(_archs ${CMAKE_CUDA_ARCHITECTURES})
        endif()
        foreach(_g IN LISTS _archs)
            if(_g MATCHES "^([0-9]+)$")
                list(APPEND _gpu_flags "-gpu=sm_${_g}")
            endif()
        endforeach()
    endif()

    # Locate mexcuda executable
    _mexcuda_locate_exe(_mexcuda_exe)

    # Determine expected output (if FindMatlab provided the extension)
    set(_mex_output "")
    if(DEFINED Matlab_MEX_EXTENSION)
        set(_mex_output "${_outdir}/${_output_name}.${Matlab_MEX_EXTENSION}")
    else()
        message(STATUS "[mexcuda] Matlab_MEX_EXTENSION not provided; continuing with stamp-only dependencies.")
    endif()

    # Use a stamp file as the declared OUTPUT to avoid duplicate-output issues in Ninja
    set(_mex_stamp "${_outdir}/${_output_name}.mexbuilt")

    # Build command: invoke mexcuda directly
    set(_cmd_args
        -silent
        -outdir
        "$<SHELL_PATH:${_outdir}>"
        -output
        "${_output_name}"
    )
    if(MAM_R2018a)
        list(APPEND _cmd_args -R2018a)
    endif()
    list(
        APPEND
        _cmd_args
        ${_gpu_flags}
        ${_def_flags}
        ${_inc_flags}
        ${_linkopt_flags}
        ${_mexflags}
        ${_link_inputs}
        ${_abs_srcs}
    )

    if(_mex_output)
        add_custom_command(
            OUTPUT "${_mex_stamp}"
            COMMAND "${_mexcuda_exe}" ${_cmd_args}
            COMMAND ${CMAKE_COMMAND} -E touch "${_mex_stamp}"
            BYPRODUCTS "${_mex_output}"
            DEPENDS ${_abs_srcs} ${_dep_targets}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "[mexcuda] Building MEX ${_output_name}"
            VERBATIM
        )
    else()
        add_custom_command(
            OUTPUT "${_mex_stamp}"
            COMMAND "${_mexcuda_exe}" ${_cmd_args}
            COMMAND ${CMAKE_COMMAND} -E touch "${_mex_stamp}"
            DEPENDS ${_abs_srcs} ${_dep_targets}
            WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
            COMMENT "[mexcuda] Building MEX ${_output_name}"
            VERBATIM
        )
    endif()

    add_custom_target(${_target} DEPENDS "${_mex_stamp}")
    add_custom_target(${_target}__build ALL DEPENDS "${_mex_stamp}")

    set_target_properties(
        ${_target}
        PROPERTIES MEX_OUTPUT "${_mex_output}" MEX_OUTDIR "${_outdir}" MEX_OUTPUT_NAME "${_output_name}"
    )
endfunction()
