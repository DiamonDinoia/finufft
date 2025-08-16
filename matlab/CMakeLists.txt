# matlab/CMakeLists.txt

find_package(Matlab REQUIRED COMPONENTS MEX_COMPILER MX_LIBRARY)
set(MEX_EXT "${Matlab_MEX_EXTENSION}")
set(FINUFFT_MEX_DEFS R2008OO)

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
    target_link_libraries(cufinufft_mex PRIVATE cufinufft Matlab::mex Matlab::mx)
endif()

# MATLAB assets
file(GLOB MATLAB_MFILES "${CMAKE_CURRENT_SOURCE_DIR}/*.m" "${CMAKE_CURRENT_SOURCE_DIR}/*.docsrc")

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

foreach(tgt IN ITEMS finufft_mex cufinufft_mex)
    if(TARGET ${tgt})
        if(APPLE)
            set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "@loader_path" BUILD_WITH_INSTALL_RPATH ON)
        elseif(UNIX AND NOT APPLE)
            set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "\$ORIGIN")
        endif()
    endif()
endforeach()

if(FINUFFT_MATLAB_INSTALL)
    set(CPACK_GENERATOR "TGZ;ZIP")
    set(CPACK_PACKAGE_NAME "finufft-matlab")
    include(CPack)

    install(FILES ${MATLAB_MFILES} DESTINATION matlab)
    install(DIRECTORY "@gpuArray" DESTINATION matlab)

    if(TARGET finufft_mex)
        install(
            TARGETS finufft_mex
            LIBRARY
                DESTINATION
                    matlab # .mexa64/.mexmaci64
            RUNTIME
                DESTINATION
                    matlab # .mexw64 on Windows
        )
    endif()
    if(TARGET cufinufft_mex)
        install(TARGETS cufinufft_mex LIBRARY DESTINATION matlab RUNTIME DESTINATION matlab)
    endif()
    # Convenience meta-target
    if(TARGET finufft_mex OR TARGET cufinufft_mex)
        add_custom_target(
            matlab
            DEPENDS $<$<TARGET_EXISTS:finufft_mex>:finufft_mex> $<$<TARGET_EXISTS:cufinufft_mex>:cufinufft_mex>
        )
    endif()
endif()
