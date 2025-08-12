set(PRECISION_INDEPENDENT_SRC precision_independent.cu ../utils.cpp)

set(PRECISION_DEPENDENT_SRC
    spreadinterp.cpp
    1d/cufinufft1d.cu
    1d/spread1d_wrapper.cu
    1d/interp1d_wrapper.cu
    2d/cufinufft2d.cu
    2d/spread2d_wrapper.cu
    2d/interp2d_wrapper.cu
    3d/spread3d_wrapper.cu
    3d/interp3d_wrapper.cu
    3d/cufinufft3d.cu
    memtransfer_wrapper.cu
    deconvolve_wrapper.cu
    cufinufft.cu
    common.cu
)

# Link scope for CUDA lib: PUBLIC for static builds (propagate), PRIVATE for shared (minimize exports)
if(BUILD_SHARED_LIBS)
    set(_cufinufft_link_scope PRIVATE)
else()
    set(_cufinufft_link_scope PUBLIC)
endif()

add_library(cufinufft ${PRECISION_INDEPENDENT_SRC} ${PRECISION_DEPENDENT_SRC})
add_library(cufinufft::cufinufft ALIAS cufinufft)

# Public includes: only export install/include (export-safe)
target_include_directories(
    cufinufft
    PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
    PRIVATE
        ${PROJECT_SOURCE_DIR}/contrib
        $<TARGET_PROPERTY:CUDA::cudart,INTERFACE_INCLUDE_DIRECTORIES>
        $<TARGET_PROPERTY:CUDA::cufft,INTERFACE_INCLUDE_DIRECTORIES>
)

# Minimize exported symbols on ELF
set_target_properties(
    cufinufft
    PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
        WINDOWS_EXPORT_ALL_SYMBOLS ${FINUFFT_WINDOWS_EXPORT_ALL}
        INTERPROCEDURAL_OPTIMIZATION OFF
        POSITION_INDEPENDENT_CODE ${FINUFFT_POSITION_INDEPENDENT_CODE}
        LINKER_LANGUAGE CUDA
)

target_compile_features(cufinufft PRIVATE cxx_std_17 cuda_std_17)

if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(cufinufft PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --extra-device-vectorization>)
endif()

# CUDA libs: scope depends on shared/static
if(WIN32 OR FINUFFT_BUILD_TESTS OR CIBUILDWHEEL)
    target_link_libraries(cufinufft ${_cufinufft_link_scope} CUDA::cudart CUDA::cufft)
else()
    target_link_libraries(cufinufft ${_cufinufft_link_scope} CUDA::cudart_static CUDA::cufft_static)
endif()

target_link_libraries(cufinufft PRIVATE CCCL::CCCL)

# Public headers for CUDA part
file(GLOB CUFINUFFT_PUBLIC_HEADERS "${CMAKE_SOURCE_DIR}/include/cufinufft*.h")
set_target_properties(cufinufft PROPERTIES PUBLIC_HEADER "${CUFINUFFT_PUBLIC_HEADERS}")

if(FINUFFT_ENABLE_INSTALL)
    install(
        TARGETS cufinufft
        EXPORT finufftTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()
