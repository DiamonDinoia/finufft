# Ensure MATLAB is on PATH if provided by setup-matlab
find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER MX_LIBRARY)

# MEX extension and common defines
set(MEX_EXT "${Matlab_MEX_EXTENSION}")
set(FINUFFT_MEX_DEFS R2008OO)

# Make MEX self-contained where we can
function(make_mex_self_contained tgt)
    if(UNIX AND NOT APPLE)
        # Avoid MATLAB's older libstdc++ at runtime
        target_link_options(${tgt} PRIVATE -static-libstdc++ -static-libgcc)
    endif()
endfunction()

# ------------------------------------------------------------
# --- CPU MEX via CMake's Matlab API ---
if(FINUFFT_USE_CPU)
    # Everything that needs to be linked beyond Matlab::mex
    set(link_to_cpu finufft Matlab::mx Matlab::mex)

    matlab_add_mex(
        NAME finufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/finufft.cpp"
        OUTPUT_NAME finufft
        LINK_TO ${link_to_cpu}
        # R2017b enforces large array dims while keeping the separate complex API
        R2017b
    )

    target_compile_definitions(finufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(finufft_mex PRIVATE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)

    # Ensure mexFunction is exported on Windows (typically not needed with matlab_add_mex, but keep if required)
    if(WIN32)
        target_link_options(finufft_mex PRIVATE "/EXPORT:mexFunction")
    endif()

    make_mex_self_contained(finufft_mex)
endif()

# --- CUDA MEX via CMake's Matlab API ---
if(FINUFFT_USE_CUDA)
    # Locate MATLAB GPU header (mxGPUArray.h)
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

    # Search recursively from ${Matlab_ROOT_DIR} for the MWGPU library
    if(WIN32)
        file(GLOB_RECURSE _hits LIST_DIRECTORIES false "${Matlab_ROOT_DIR}/**/mwgpu.lib")
    elseif(APPLE)
        file(GLOB_RECURSE _hits LIST_DIRECTORIES false "${Matlab_ROOT_DIR}/**/libmwgpu.dylib")
    else() # Linux / UNIX
        file(
            GLOB_RECURSE _hits
            LIST_DIRECTORIES false
            "${Matlab_ROOT_DIR}/**/libmwgpu.so"
            "${Matlab_ROOT_DIR}/**/libmwgpu.so.*"
        )
    endif()

    list(REMOVE_DUPLICATES _hits)
    if(_hits)
        list(GET _hits 0 Matlab_MWGPU_LIBRARY)
        set(Matlab_MWGPU_LIBRARY "${Matlab_MWGPU_LIBRARY}" CACHE FILEPATH "MATLAB MWGPU library")
        message(STATUS "Found MWGPU: ${Matlab_MWGPU_LIBRARY}")
    else()
        message(FATAL_ERROR "Could not find MWGPU under ${Matlab_ROOT_DIR}")
    endif()
    # GPU libraries to link (Matlab::mex is added by matlab_add_mex automatically)
    set(link_to_cuda
        cufinufft
        CUDA::cudart
        Matlab::mex
        Matlab::mx
        ${Matlab_MWGPU_LIBRARY}
    )

    matlab_add_mex(
        NAME cufinufft_mex
        SRC "${CMAKE_CURRENT_SOURCE_DIR}/cufinufft.cu"
        OUTPUT_NAME cufinufft
        LINK_TO ${link_to_cuda}
        # R2017b enforces large array dims while keeping the separate complex API
        R2017b
    )

    target_compile_definitions(cufinufft_mex PRIVATE ${FINUFFT_MEX_DEFS})
    target_include_directories(cufinufft_mex PRIVATE ${Matlab_GPU_INCLUDE_DIR})

    # On Windows: export symbol if needed (usually unnecessary)
    if(WIN32)
        target_link_options(cufinufft_mex PRIVATE "/EXPORT:mexFunction")
    endif()

    make_mex_self_contained(cufinufft_mex)
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
    add_custom_command(
        TARGET cufinufft_mex
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MATLAB_MFILES} $<TARGET_FILE_DIR:cufinufft_mex>
    )
endif()

# ------------------------------------------------------------
# RPATH for installed artifacts
foreach(tgt IN ITEMS finufft_mex cufinufft_mex)
    if(TARGET ${tgt})
        if(APPLE)
            # Let the loader find deps next to the MEX at runtime
            set_target_properties(
                ${tgt}
                PROPERTIES BUILD_WITH_INSTALL_RPATH ON BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path"
            )
        elseif(UNIX)
            # Same idea on Linux
            set_target_properties(
                ${tgt}
                PROPERTIES BUILD_WITH_INSTALL_RPATH ON BUILD_RPATH "\$ORIGIN" INSTALL_RPATH "\$ORIGIN"
            )
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
