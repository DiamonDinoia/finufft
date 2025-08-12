# Helper functions for configuring FINUFFT targets

function(finufft_link_test target)
    target_link_libraries(${target} PRIVATE finufft::finufft)
    enable_asan(${target})
    target_compile_features(${target} PRIVATE cxx_std_17)

    # Tests should compile with the same backend selection as the library
    if(FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PRIVATE FINUFFT_USE_DUCC0)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_WARN_GNUC})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_WARN_MSVC})
    endif()

    set_target_properties(
        ${target}
        PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
    )
endfunction()

function(set_finufft_options target)
    target_compile_features(${target} PUBLIC cxx_std_17)

    # Default hidden visibility on ELF (minimize exported symbols)
    set_target_properties(${target} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)

    # Make backend selection visible to dependents (so headers/source agree)
    if(FINUFFT_USE_DUCC0)
        target_compile_definitions(${target} PUBLIC FINUFFT_USE_DUCC0)
    endif()

    # Warnings
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_WARN_GNUC})
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_WARN_MSVC})
    endif()

    # Arch/optimization per config (PRIVATE)
    if(CMAKE_BUILD_TYPE MATCHES "^[Rr]elease|RelWithDebInfo$")
        if(FINUFFT_ARCH_FLAGS)
            finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_ARCH_FLAGS})
        endif()
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_REL_GNUC})
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_REL_MSVC})
        endif()
    elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_DBG_GNUC})
        elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
            finufft_target_add_flags_if_supported(${target} PRIVATE ${FINUFFT_DBG_MSVC})
        endif()
    endif()

    # Public includes (export-safe)
    target_include_directories(
        ${target}
        PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    )

    set_target_properties(
        ${target}
        PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
            INTERPROCEDURAL_OPTIMIZATION ${FINUFFT_INTERPROCEDURAL_OPTIMIZATION}
    )

    enable_asan(${target})
endfunction()
