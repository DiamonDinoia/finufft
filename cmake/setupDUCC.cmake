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

add_library(
    ducc0
    STATIC
    ${ducc0_SOURCE_DIR}/src/ducc0/infra/string_utils.cc
    ${ducc0_SOURCE_DIR}/src/ducc0/infra/threading.cc
    ${ducc0_SOURCE_DIR}/src/ducc0/infra/mav.cc
    ${ducc0_SOURCE_DIR}/src/ducc0/math/gridding_kernel.cc
    ${ducc0_SOURCE_DIR}/src/ducc0/math/gl_integrator.cc
)
add_library(ducc0::ducc0 ALIAS ducc0)

# DUCC0 needs its headers to compile (PRIVATE),
# and finufft needs them while building (BUILD_INTERFACE on INTERFACE).
target_include_directories(ducc0 PRIVATE ${ducc0_SOURCE_DIR}/src INTERFACE $<BUILD_INTERFACE:${ducc0_SOURCE_DIR}/src>)
# (No INSTALL_INTERFACE on purpose to avoid leaking source paths.)

target_compile_features(ducc0 PRIVATE cxx_std_17)

# Basic fast-math; only if supported
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    finufft_target_add_flags_if_supported(ducc0 PRIVATE -ffast-math)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    finufft_target_add_flags_if_supported(ducc0 PRIVATE /fp:fast)
endif()

set_target_properties(
    ducc0
    PROPERTIES
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>"
        POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
)

# Threading for DUCC0 itself
if(FINUFFT_USE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries(ducc0 PRIVATE OpenMP::OpenMP_CXX)
else()
    find_package(Threads REQUIRED)
    target_link_libraries(ducc0 PRIVATE Threads::Threads)
endif()

enable_asan(ducc0)

# Install & export ducc0 so finufft’s export remains self-contained.
# (INTERFACE include dir is build-only, so nothing leaks into install.)
if(FINUFFT_ENABLE_INSTALL)
    install(
        TARGETS ducc0
        EXPORT finufftTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()
