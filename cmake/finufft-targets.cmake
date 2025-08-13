# Helper functions for configuring FINUFFT targets

function(finufft_link_test target)
    if(FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
    endif()

    # Prefer the umbrella target (brings finufft + deps for build-tree consumers)
    if(TARGET finufft::all)
        target_link_libraries(${target} PRIVATE finufft::all)
    else()
        target_link_libraries(${target} PRIVATE finufft)
    endif()

    enable_asan(${target})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE ${FINUFFT_WARNING_FLAGS})
    set_target_properties(
        ${target}
        PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
    )
endfunction()

function(set_finufft_options target)
    target_compile_features(${target} PUBLIC cxx_std_17)

    target_compile_options(${target} PRIVATE ${FINUFFT_WARNING_FLAGS})
    target_compile_options(
        ${target}
        PRIVATE
            $<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>
            $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>
            $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>
            $<$<CONFIG:Debug>:${FINUFFT_CXX_FLAGS_DEBUG}>
    )

    # Public includes for build-tree, install-time includes are relative
    target_include_directories(
        ${target}
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        INTERFACE $<INSTALL_INTERFACE:include>
    )

    set_target_properties(
        ${target}
        PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
            INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
    )

    enable_asan(${target})

    if(FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
    endif()

    # NOTE:
    # * Do NOT link FFTW here (keeps it out of the install interface).
    # * xsimd is header-only; we include it privately from src/CMakeLists.
    # * OpenMP linkage choice is handled in src/CMakeLists.
endfunction()
