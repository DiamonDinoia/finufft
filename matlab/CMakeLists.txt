find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER MX_LIBRARY)
set(MEX_EXT "${Matlab_MEX_EXTENSION}")
set(FINUFFT_MEX_DEFS R2008OO)

# Helper: prefer MATLAB's bundled OpenMP if available
# - Enables OpenMP compile flags
# - Links against libiomp5 from the MATLAB installation
function(use_matlab_openmp tgt)
    if(NOT FINUFFT_USE_OPENMP)
        return()
    endif()
    # Derive Matlab root if not set by FindMatlab
    if(NOT Matlab_ROOT_DIR)
        # Guess from include path: <MATLAB>/extern/include
        list(GET Matlab_INCLUDE_DIRS 0 _mat_inc)
        get_filename_component(_extern_dir "${_mat_inc}" DIRECTORY) # .../extern
        get_filename_component(Matlab_ROOT_DIR "${_extern_dir}" DIRECTORY) # <MATLAB>
    endif()

    if(WIN32)
        set(_MAT_BIN_SUBDIR "win64")
        set(_MAT_OMP_NAME "libiomp5md.lib")
    elseif(APPLE)
        set(_MAT_BIN_SUBDIR "maci64")
        set(_MAT_OMP_NAME "libiomp5.dylib")
    else()
        set(_MAT_BIN_SUBDIR "glnxa64")
        set(_MAT_OMP_NAME "libiomp5.so")
    endif()

    set(_MAT_OMP_PATH "${Matlab_ROOT_DIR}/bin/${_MAT_BIN_SUBDIR}/${_MAT_OMP_NAME}")

    if(EXISTS "${_MAT_OMP_PATH}")
        get_filename_component(_MAT_OMP_DIR "${_MAT_OMP_PATH}" DIRECTORY)
        # Link to MATLAB's OpenMP runtime
        target_link_libraries(${tgt} "${_MAT_OMP_PATH}")
        target_link_libraries(${tgt} PRIVATE "${_MAT_OMP_PATH}")

        message(STATUS "Using MATLAB-supplied OpenMP for target '${tgt}': ${_MAT_OMP_PATH}")
    else()
        message(STATUS "MATLAB OpenMP not found at ${_MAT_OMP_PATH}; skipping MATLAB OpenMP setup for '${tgt}'")
    endif()
endfunction()

function(make_mex_self_contained tgt)
    if(UNIX AND NOT APPLE)
        # Avoid MATLAB's older libstdc++ at runtime
        target_link_options(${tgt} PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endfunction()

# --- CPU MEX ---
if(FINUFFT_USE_CPU)
    matlab_add_mex(
        NAME finufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/finufft.cpp"
        OUTPUT_NAME finufft
        R2018a
        LINK_TO finufft
    )
    target_compile_definitions(finufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(finufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    use_matlab_openmp(finufft_mex)
    make_mex_self_contained(finufft_mex)
endif()

# --- CUDA MEX ---
if(FINUFFT_USE_CUDA)
    add_library(cufinufft_mex MODULE "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu")
    set_target_properties(
        cufinufft_mex
        PROPERTIES
            PREFIX ""
            SUFFIX ".${MEX_EXT}"
            OUTPUT_NAME "cufinufft"
            POSITION_INDEPENDENT_CODE ON
            CUDA_SEPARABLE_COMPILATION ON
            LINKER_LANGUAGE CUDA
    )
    target_compile_definitions(cufinufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(cufinufft_mex PRIVATE ${Matlab_INCLUDE_DIRS})
    target_link_libraries(cufinufft_mex cufinufft Matlab::mex Matlab::mx)
    use_matlab_openmp(cufinufft_mex)
    make_mex_self_contained(cufinufft_mex)
endif()

# MATLAB assets (.m and .docsrc files)
file(GLOB MATLAB_MFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.m" "${CMAKE_CURRENT_SOURCE_DIR}/*.docsrc")

# Copy .m files next to built MEX(es) for in-tree testing
if(TARGET finufft_mex)
    add_custom_command(
        TARGET finufft_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} $<TARGET_FILE_DIR:finufft_mex>
    )
endif()
if(TARGET cufinufft_mex)
    add_custom_command(
        TARGET cufinufft_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} $<TARGET_FILE_DIR:cufinufft_mex>
    )
endif()

# RPATH for installed artifacts
foreach(tgt IN ITEMS finufft_mex cufinufft_mex)
    if(TARGET ${tgt})
        if(APPLE)
            set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH ON)
        elseif(UNIX AND NOT APPLE)
            set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "\$ORIGIN")
        endif()
    endif()
endforeach()

# ---------------- Install & CPack ----------------
if(FINUFFT_MATLAB_INSTALL)
    # Install MATLAB assets
    install(FILES ${MATLAB_MFILES} DESTINATION matlab)
    # Install class folder (if present)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/@gpuArray")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/@gpuArray" DESTINATION matlab)
    endif()

    # Install the MEX targets into matlab/
    if(TARGET finufft_mex)
        install(TARGETS finufft_mex RUNTIME DESTINATION matlab LIBRARY DESTINATION matlab ARCHIVE DESTINATION matlab)
    endif()
    if(TARGET cufinufft_mex)
        install(TARGETS cufinufft_mex RUNTIME DESTINATION matlab LIBRARY DESTINATION matlab ARCHIVE DESTINATION matlab)
    endif()

    # CPack: produce a clean archive with ONLY top-level "matlab/"
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
    if(WIN32)
        set(CPACK_GENERATOR "ZIP")
        set(CPACK_PACKAGE_FILE_NAME "finufft-matlab-mex-windows")
    elseif(APPLE)
        set(CPACK_GENERATOR "TGZ")
        set(CPACK_PACKAGE_FILE_NAME "finufft-matlab-mex-macos-${CMAKE_OSX_DEPLOYMENT_TARGET}")
    else()
        set(CPACK_GENERATOR "TGZ")
        set(CPACK_PACKAGE_FILE_NAME "finufft-matlab-mex-linux")
    endif()

    include(CPack)
endif()
