add_executable(cuperftest cuperftest.cu)
target_link_libraries(cuperftest cufinufft)
target_compile_features(cuperftest PRIVATE cxx_std_17)
set_target_properties(
    cuperftest
    PROPERTIES
        LINKER_LANGUAGE CUDA
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
)
