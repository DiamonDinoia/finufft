# MexCUDA.cmake — CMake module to build CUDA MEX via MATLAB's `mexcuda`
#
# Highlights
# - Creates a normal-looking CMake target but builds via: `matlab -batch mexcuda ...`.
# - Respects target properties you set with CMake commands:
#     * INCLUDE_DIRECTORIES  (from target_include_directories)
#     * COMPILE_DEFINITIONS  (from target_compile_definitions)
#     * CUDA_ARCHITECTURES   (from set_property/target_compile_features etc.)
#       → mapped to `-gpu=sm_XX` flags; explicit GPU_ARCH argument overrides this.
# - Lets you add libraries/targets to the MEX link via LINK_TO (recommended).
# - Provides `make_mex_self_contained()` to set rpaths for portable loading.
#
# Usage:
#   include("${CMAKE_SOURCE_DIR}/cmake/MexCUDA.cmake")
#
#   matlab_add_mexcuda(
#       NAME <target>
#       SRC <.cu files...>
#       [OUTPUT_NAME <basename>]   # defaults to NAME
#       [R2018a]                   # adds -R2018a (unified MEX API)
#       [GPU_ARCH <sm_80;sm_75;...>]
#       [LINK_TO <cmake_targets_or_full_lib_paths...>]
#       [MEX_FLAGS <extra mexcuda flags...>]   # e.g. -largeArrayDims -v -O
#   )
#
# Example:
#   matlab_add_mexcuda(
#     NAME        cufinufft_mex
#     SRC         "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu"
#     OUTPUT_NAME "cufinufft"
#     R2018a
#     LINK_TO     cufinufft Matlab::mex Matlab::mx CUDA::cudart
#   )
#   target_compile_definitions(cufinufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
#   target_include_directories(cufinufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
#   target_include_directories(cufinufft_mex PRIVATE $<BUILD_INTERFACE:${Matlab_GPU_INCLUDE_DIR}>)
#   target_include_directories(cufinufft_mex PRIVATE $<BUILD_INTERFACE:${Matlab_INCLUDE_DIRS}>)
#   make_mex_self_contained(cufinufft_mex)
#
# Requirements:
# - CMake >= 3.18 (generator expressions & CUDA_ARCHITECTURES property)
# - MATLAB reachable via `find_package(Matlab COMPONENTS MAIN_PROGRAM)`
#
cmake_minimum_required(VERSION 3.18)

if(COMMAND matlab_add_mexcuda)
    # Prevent redefinition if included twice.
    return()
endif()

# Find MATLAB launcher
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

# Create a MEX build target that invokes `mexcuda` under MATLAB -batch.
function(matlab_add_mexcuda)
    set(_opts R2018a)
    set(_one NAME OUTPUT_NAME)
    set(_multi SRC LINK_TO GPU_ARCH MEX_FLAGS)
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

    # Absolute paths to sources
    set(_abs_srcs)
    foreach(_s IN LISTS MAM_SRC)
        get_filename_component(_abs "${_s}" ABSOLUTE)
        list(APPEND _abs_srcs "${_abs}")
    endforeach()

    # Build list of link inputs from LINK_TO: use linker file of CMake targets, or pass absolute paths directly.
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
                "matlab_add_mexcuda(${_target}): LINK_TO entry '${_lib}' is neither a CMake target nor an absolute path; ignored. Prefer CMake targets (cufinufft, Matlab::mx, CUDA::cudart) or absolute library files."
            )
        endif()
    endforeach()

    # Generator-expressions for includes and defines captured from target properties
    set(_inc_expr
        "$<$<BOOL:$<TARGET_PROPERTY:${_target},INCLUDE_DIRECTORIES>>: -I$<JOIN:$<TARGET_PROPERTY:${_target},INCLUDE_DIRECTORIES>,\" -I\">>"
    )
    set(_def_expr
        "$<$<BOOL:$<TARGET_PROPERTY:${_target},COMPILE_DEFINITIONS>>: -D$<JOIN:$<TARGET_PROPERTY:${_target},COMPILE_DEFINITIONS>,\" -D\">>"
    )
    # Also forward target_link_options via LINK_OPTIONS property
    set(_linkopt_expr
        "$<$<BOOL:$<TARGET_PROPERTY:${_target},LINK_OPTIONS>>: $<JOIN:$<TARGET_PROPERTY:${_target},LINK_OPTIONS>,\" \">>"
    )

    # CUDA architectures — 2 ways: — 2 ways:
    # 1) Explicit GPU_ARCH passed to the function → produce literal -gpu flags now.
    # 2) Otherwise, read the target's CUDA_ARCHITECTURES property at generate time and
    #    translate in MATLAB (strips '-real'/'-virtual', keeps only digits).
    set(_gpu_flags_str "")
    foreach(_g IN LISTS MAM_GPU_ARCH)
        string(APPEND _gpu_flags_str " -gpu=sm_${_g}")
    endforeach()

    # Raw arch list from target property; evaluated by file(GENERATE) at generate-time.
    set(_cuda_arch_expr "$<JOIN:$<TARGET_PROPERTY:${_target},CUDA_ARCHITECTURES>,;>")

    # Extra mex flags
    string(JOIN " " _mexflags ${MAM_MEX_FLAGS})

    # Compose list strings for MATLAB script
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

    # Link inputs string from generator-expr list
    if(_link_inputs)
        string(JOIN " " _link_inputs_str ${_link_inputs})
    else()
        set(_link_inputs_str "")
    endif()

    # MATLAB API flag
    set(_api_flag "")
    if(MAM_R2018a)
        set(_api_flag " -R2018a")
    endif()

    # Generate the .m script (we build a 'mexargs' string and eval it)
    file(
        GENERATE OUTPUT
        "${_script}"
        CONTENT
            "try
  mexargs = '';
  mexargs = [mexargs ' -silent -outdir ''${_outdir}'' -output ''${_output_name}''${_api_flag}'];
  % Forwarded compile defs & include dirs from target properties
  mexargs = [mexargs ' ${_def_expr} ${_inc_expr} ${_linkopt_expr} ${_mexflags}'];
  % GPU archs: explicit overrides property-derived
  gpuExplicit = '${_gpu_flags_str}';
  mexargs = [mexargs, gpuExplicit];
  if strlength(strtrim(gpuExplicit)) == 0
    archesRaw = '${_cuda_arch_expr}';
    if ~isempty(archesRaw) && ~strcmpi(archesRaw,'native') && ~strcmpi(archesRaw,'all')
      items = strsplit(archesRaw, ';');
      for i = 1:numel(items)
        tok = regexp(items{i}, '(\\d+)', 'tokens', 'once');
        if ~isempty(tok)
          mexargs = [mexargs ' -gpu=sm_' tok{1}];
        end
      end
    end
  end
  % Build full command line
  mexcmd = ['mexcuda' mexargs ' ${_link_inputs_str} ${_srcs_str}'];
  % Uncomment to debug:
  % disp(mexcmd);
  eval(mexcmd);
catch e
  disp(getReport(e));
  exit(1);
end
"
    )

    # Determine MEX extension now for BYPRODUCTS; continue even if unknown.
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

    add_custom_target(${_target} ALL DEPENDS "${_mex_output}")

    set_target_properties(
        ${_target}
        PROPERTIES MEX_OUTPUT "${_mex_output}" MEX_OUTDIR "${_outdir}" MEX_OUTPUT_NAME "${_output_name}"
    )
endfunction()

# Convenience: set rpath so the MEX can find adjacent shared libs without LD_LIBRARY_PATH.
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
            add_custom_command(
                TARGET ${tgt}
                POST_BUILD
                COMMAND ${INSTALL_NAME_TOOL} -add_rpath "@loader_path" "${_mex_file}"
                COMMENT "[mexcuda] Adding @loader_path rpath to ${_mex_file}"
            )
        else()
            message(STATUS "install_name_tool not found; skipping rpath tweak for ${tgt}")
        endif()
    elseif(UNIX)
        find_program(PATCHELF patchelf)
        if(PATCHELF)
            add_custom_command(
                TARGET ${tgt}
                POST_BUILD
                COMMAND ${PATCHELF} --set-rpath "\$ORIGIN" "${_mex_file}"
                COMMENT "[mexcuda] Setting RUNPATH=$ORIGIN on ${_mex_file}"
            )
        else()
            message(STATUS "patchelf not found; skipping rpath tweak for ${tgt}")
        endif()
    else()
        # Windows: nothing to do; DLL lookup includes module directory by default
    endif()
endfunction()
