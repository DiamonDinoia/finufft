CPMAddPackage(
    NAME
    ducc0
    GIT_REPOSITORY
    https://gitlab.mpcdf.mpg.de/mtr/ducc.git
    GIT_TAG
    ${DUCC0_VERSION}
    DOWNLOAD_ONLY
    YES
)

if(ducc0_ADDED)
    add_library(
        ducc0
        STATIC
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/string_utils.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/threading.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/infra/mav.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/math/gridding_kernel.cc
        ${ducc0_SOURCE_DIR}/src/ducc0/math/gl_integrator.cc
    )
    target_include_directories(ducc0 PUBLIC ${ducc0_SOURCE_DIR}/src/)
    target_compile_features(ducc0 PRIVATE cxx_std_17)
    target_compile_options(
        ducc0
        PRIVATE
            $<$<CONFIG:Release,RelWithDebInfo>:${FINUFFT_ARCH_FLAGS}>
            $<$<CONFIG:Release>:${FINUFFT_CXX_FLAGS_RELEASE}>
            $<$<CONFIG:RelWithDebInfo>:${FINUFFT_CXX_FLAGS_RELWITHDEBINFO}>
            $<$<CONFIG:Debug>:${FINUFFT_CXX_FLAGS_DEBUG}>
            $<$<CXX_COMPILER_ID:GNU,Clang>:-ffast-math>
            $<$<CXX_COMPILER_ID:MSVC>:/fp:fast>
    )
    set_target_properties(
        ducc0
        PROPERTIES
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
            POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
    )
    if(NOT OpenMP_CXX_FOUND)
        find_package(Threads REQUIRED)
        target_link_libraries(ducc0 PRIVATE Threads::Threads)
    endif()
    enable_asan(ducc0)
endif()
