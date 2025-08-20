# MexCUDA.cmake — CMake module to build CUDA MEX via MATLAB's `mexcuda`
#
# Design
# - Creates a pair of custom targets:
#     <name>           : a phony target you can depend on or invoke
#     <name>__build    : actually builds the MEX by running `matlab -batch` with `mexcuda`
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

# MATLAB launcher for -batch
find_package(Matlab REQUIRED COMPONENTS MAIN_PROGRAM)

# Helper: get mex extension once at configure time.
function(_mexcuda_get_mexext OUT_VAR)
    execute_process(
        COMMAND "${Matlab_MAIN_PROGRAM}" -batch "try, fprintf('%s', mexext); catch, exit(1); end"
        OUTPUT_VARIABLE _MEX_EXT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    set(${OUT_VAR} "${_MEX_EXT}" PARENT_SCOPE)
endfunction()

# Add a CUDA MEX build driven by mexcuda
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
            list(APPEND _link_inputs "'$<SHELL_PATH:$<TARGET_LINKER_FILE:${_lib}>>'")
            list(APPEND _dep_targets ${_lib})
        elseif(IS_ABSOLUTE "${_lib}")
            list(APPEND _link_inputs "'$<SHELL_PATH:${_lib}>'")
        else()
            message(
                WARNING
                "matlab_add_mexcuda(${_target}): LINK_TO entry '${_lib}' is neither a CMake target nor an absolute path; ignored."
            )
        endif()
    endforeach()
    string(JOIN " " _link_inputs_str ${_link_inputs})

    # Includes/defines/link options (allow generator expressions; quote paths)
    set(_inc_flags)
    foreach(_inc IN LISTS MAM_INCLUDES)
        list(APPEND _inc_flags "-I'$<SHELL_PATH:${_inc}>'")
    endforeach()
    string(JOIN " " _inc_flags_str ${_inc_flags})

    set(_def_flags)
    foreach(_d IN LISTS MAM_DEFINES)
        list(APPEND _def_flags "-D${_d}")
    endforeach()
    string(JOIN " " _def_flags_str ${_def_flags})

    string(JOIN " " _linkopt_flags_str ${MAM_LINK_OPTIONS})
    string(JOIN " " _mexflags_str ${MAM_MEX_FLAGS})

    # GPU arch flags
    set(_gpu_flags_str "")
    if(MAM_GPU_ARCH)
        foreach(_g IN LISTS MAM_GPU_ARCH)
            if(_g MATCHES "^sm_\\d+")
                string(APPEND _gpu_flags_str " -gpu=${_g}")
            else()
                string(APPEND _gpu_flags_str " -gpu=sm_${_g}")
            endif()
        endforeach()
    else()
        set(_archs ${MAM_CUDA_ARCHITECTURES})
        if(NOT _archs AND CMAKE_CUDA_ARCHITECTURES)
            set(_archs ${CMAKE_CUDA_ARCHITECTURES})
        endif()
        foreach(_g IN LISTS _archs)
            if(_g MATCHES "^([0-9]+)$")
                string(APPEND _gpu_flags_str " -gpu=sm_${_g}")
            endif()
        endforeach()
    endif()

    # MATLAB script generation
    set(_script_dir "${CMAKE_CURRENT_BINARY_DIR}/${_target}.mexbuild")
    file(MAKE_DIRECTORY "${_script_dir}")
    set(_script "${_script_dir}/build_${_target}.m")

    # Build source list (single-quoted absolute paths)
    set(_src_quoted)
    foreach(_s IN LISTS _abs_srcs)
        string(REPLACE "'" "''" _s_q "${_s}")
        list(APPEND _src_quoted "'${_s_q}'")
    endforeach()
    string(JOIN " " _srcs_str ${_src_quoted})

    set(_api_flag "")
    if(MAM_R2018a)
        set(_api_flag " -R2018a")
    endif()

    file(
        GENERATE OUTPUT
        "${_script}"
        CONTENT
            "try\n  mexcmd = ['mexcuda -silent -outdir ''${_outdir}'' -output ''${_output_name}''${_api_flag} ${_gpu_flags_str} ${_def_flags_str} ${_inc_flags_str} ${_linkopt_flags_str} ${_mexflags_str} ${_link_inputs_str} ${_srcs_str}'];\n  % disp(mexcmd);\n  eval(mexcmd);\ncatch e\n  disp(getReport(e));\n  exit(1);\nend\n"
    )

    _mexcuda_get_mexext(_mexext)
    if(_mexext)
        set(_mex_output "${_outdir}/${_output_name}.${_mexext}")
    else()
        set(_mex_output "${_outdir}/${_output_name}")
    endif()

    add_custom_command(
        OUTPUT "${_mex_output}"
        COMMAND "${Matlab_MAIN_PROGRAM}" -batch "run('${_script}')"
        BYPRODUCTS "${_mex_output}"
        DEPENDS ${_abs_srcs} ${_dep_targets}
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "[mexcuda] Building MEX ${_output_name}"
        VERBATIM
    )

    add_custom_target(${_target} DEPENDS "${_mex_output}")
    add_custom_target(${_target}__build ALL DEPENDS "${_mex_output}")

    set_target_properties(
        ${_target}
        PROPERTIES MEX_OUTPUT "${_mex_output}" MEX_OUTDIR "${_outdir}" MEX_OUTPUT_NAME "${_output_name}"
    )
endfunction()

# Add an rpath-fix target that runs after the MEX is built
function(make_mex_self_contained tgt)
    get_target_property(_mex_file ${tgt} MEX_OUTPUT)
    if(NOT _mex_file)
        message(
            WARNING
            "make_mex_self_contained(${tgt}): target has no MEX_OUTPUT property; did you create it with matlab_add_mexcuda?"
        )
        return()
    endif()

    if(APPLE)
        find_program(INSTALL_NAME_TOOL install_name_tool)
        if(INSTALL_NAME_TOOL)
            add_custom_target(
                ${tgt}__fixrpath
                ALL
                COMMAND ${INSTALL_NAME_TOOL} -add_rpath "@loader_path" "${_mex_file}"
                DEPENDS ${tgt}
                COMMENT "[mexcuda] Adding @loader_path rpath to ${_mex_file}"
            )
        else()
            message(STATUS "install_name_tool not found; skipping rpath tweak for ${tgt}")
        endif()
    elseif(UNIX)
        find_program(PATCHELF patchelf)
        if(PATCHELF)
            add_custom_target(
                ${tgt}__fixrpath
                ALL
                COMMAND ${PATCHELF} --set-rpath "\$ORIGIN" "${_mex_file}"
                DEPENDS ${tgt}
                COMMENT "[mexcuda] Setting RUNPATH=$ORIGIN on ${_mex_file}"
            )
        else()
            message(STATUS "patchelf not found; skipping rpath tweak for ${tgt}")
        endif()
    else()
        # Windows: nothing to do
    endif()
endfunction()
