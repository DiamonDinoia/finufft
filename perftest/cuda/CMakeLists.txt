add_executable(cuperftest cuperftest.cu)
target_include_directories(cuperftest PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
target_link_libraries(cuperftest PRIVATE cufinufft CUDA::cufft CUDA::cudart $<BUILD_INTERFACE:finufft_common>)
target_compile_features(cuperftest PRIVATE cxx_std_17)
set_target_properties(
    cuperftest
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)

add_executable(cuperftest_methods cuperftest_methods.cu)
target_include_directories(cuperftest_methods PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
target_link_libraries(cuperftest_methods PRIVATE cufinufft CUDA::cufft CUDA::cudart $<BUILD_INTERFACE:finufft_common>)
target_compile_features(cuperftest_methods PRIVATE cxx_std_17)
set_target_properties(
    cuperftest_methods
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)
