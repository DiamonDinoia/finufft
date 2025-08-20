cmake_minimum_required(VERSION 3.18)

# MATLAB: we also need MAIN_PROGRAM for `matlab -batch`
find_package(Matlab REQUIRED COMPONENTS MAIN_PROGRAM MEX_COMPILER MX_LIBRARY)

# Bring in mexcuda helper (builds via `matlab -batch mexcuda`)
include("${CMAKE_SOURCE_DIR}/cmake/MexCUDA.cmake")

# MEX extension and common defines
set(MEX_EXT "${Matlab_MEX_EXTENSION}")
set(FINUFFT_MEX_DEFS R2008OO)

# ------------------------------------------------------------
# --- CPU MEX (uses CMake's 'matlab_add_mex' → mex) ---
if(FINUFFT_USE_CPU)
    matlab_add_mex(
        NAME finufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/finufft.cpp"
        OUTPUT_NAME "finufft"
        R2018a
        LINK_TO finufft
    )

    target_compile_definitions(finufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(finufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
    if(UNIX AND NOT APPLE)
        target_link_options(finufft_mex PRIVATE -static-libstdc++ -static-libgcc)
    endif()

    # RPATH via CMake (no patchelf)
    if(APPLE)
        set_target_properties(
            finufft_mex
            PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path" INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    elseif(UNIX)
        set_target_properties(
            finufft_mex
            PROPERTIES BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN" INSTALL_RPATH_USE_LINK_PATH TRUE
        )
    endif()
endif()

# ------------------------------------------------------------
# --- CUDA MEX (uses mexcuda via matlab_add_mexcuda) ---
if(FINUFFT_USE_CUDA)
    find_path(
        Matlab_GPU_INCLUDE_DIR
        NAMES gpu/mxGPUArray.h
        HINTS
            "${Matlab_ROOT_DIR}/extern/include"
            "${Matlab_ROOT_DIR}/toolbox/parallel/gpu/extern/include"
            "${Matlab_ROOT_DIR}/toolbox/distcomp/gpu/extern/include"
    )
    if(NOT Matlab_GPU_INCLUDE_DIR)
        message(
            FATAL_ERROR
            "MATLAB GPU header mxGPUArray.h not found. "
            "Ensure Parallel Computing Toolbox is installed, or add its include path."
        )
    endif()

    # If you prefer, you can set your own list; module falls back to CMAKE_CUDA_ARCHITECTURES.
    # set(FINUFFT_CUDA_ARCHS "75;80;86" CACHE STRING "CUDA SM versions for cufinufft_mex")
    matlab_add_mexcuda(
        NAME cufinufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu"
        OUTPUT_NAME "cufinufft"
        R2018a
        INCLUDES
            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
            $<BUILD_INTERFACE:${Matlab_GPU_INCLUDE_DIR}>
            $<BUILD_INTERFACE:${Matlab_INCLUDE_DIRS}>
        DEFINES ${FINUFFT_MEX_DEFS}
        LINK_TO cufinufft Matlab::mex Matlab::mx CUDA::cudart
        LINK_OPTIONS
            # Static libstdc++/libgcc on Linux (keep yours)
            $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:-static-libstdc++>
            $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:-static-libgcc>
            # RPATH via linker (no patchelf)
            $<$<AND:$<BOOL:UNIX>,$<NOT:$<BOOL:APPLE>>>:-Wl,-rpath,\$ORIGIN>
            $<$<BOOL:APPLE>:-Wl,-rpath,@loader_path>
        CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    )
endif()

# ------------------------------------------------------------
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
    get_target_property(CUFINUFFT_MEX_PATH cufinufft_mex MEX_OUTPUT)
    if(CUFINUFFT_MEX_PATH)
        get_filename_component(CUFINUFFT_MEX_DIR "${CUFINUFFT_MEX_PATH}" DIRECTORY)
        add_custom_target(
            cufinufft_mex_copy
            ALL
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} "${CUFINUFFT_MEX_DIR}"
            DEPENDS cufinufft_mex
            COMMENT "Copy MATLAB .m files next to cufinufft MEX"
        )
    endif()
endif()

# ------------------------------------------------------------
# RPATH for installed artifacts (CPU target only; CUDA handled by make_mex_self_contained)
foreach(tgt IN ITEMS finufft_mex)
    if(TARGET ${tgt})
        if(APPLE)
            set_target_properties(
                ${tgt}
                PROPERTIES BUILD_WITH_INSTALL_RPATH OFF INSTALL_RPATH "" INSTALL_RPATH_USE_LINK_PATH TRUE
            )
        elseif(UNIX AND NOT APPLE)
            set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "\$ORIGIN" INSTALL_RPATH_USE_LINK_PATH TRUE)
        endif()
    endif()
endforeach()

# ------------------------------------------------------------
# Install & CPack (optional)
if(FINUFFT_MATLAB_INSTALL)
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        set(CPACK_STRIP_FILES TRUE)
    endif()

    # Install MATLAB assets
    if(MATLAB_MFILES)
        install(FILES ${MATLAB_MFILES} DESTINATION matlab)
    endif()

    # Install class folder (if present)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/@gpuArray")
        install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/@gpuArray" DESTINATION matlab)
    endif()

    # Install the CPU MEX target as before
    if(TARGET finufft_mex)
        install(TARGETS finufft_mex RUNTIME DESTINATION matlab LIBRARY DESTINATION matlab ARCHIVE DESTINATION matlab)
    endif()

    # Install the CUDA MEX built via matlab_add_mexcuda using the emitted file path
    if(TARGET cufinufft_mex)
        get_target_property(CUFINUFFT_MEX_PATH cufinufft_mex MEX_OUTPUT)
        if(CUFINUFFT_MEX_PATH)
            install(FILES "${CUFINUFFT_MEX_PATH}" DESTINATION matlab)
        endif()
    endif()

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
