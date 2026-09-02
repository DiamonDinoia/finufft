file(GLOB example_src "*.cpp")

foreach(srcfile ${example_src})
    string(REPLACE ".cpp" "" executable ${srcfile})
    get_filename_component(executable ${executable} NAME)
    add_executable(${executable} ${srcfile})
    target_link_libraries(${executable} PRIVATE cufinufft CUDA::cufft CUDA::cudart $<BUILD_INTERFACE:finufft_common>)
    target_compile_features(${executable} PRIVATE cxx_std_17)
    add_test(NAME example_${executable} COMMAND ${executable})
endforeach()

# The C example needs C99 complex, which MSVC lacks. HAS_C_COMPLEX is probed once in
# cmake/toolchain.cmake.
if(HAS_C_COMPLEX)
    find_library(MATH_LIBRARY m)
    add_executable(cuda_simple1d1c simple1d1c.c)
    target_link_libraries(cuda_simple1d1c PRIVATE cufinufft CUDA::cufft CUDA::cudart)
    if(MATH_LIBRARY)
        target_link_libraries(cuda_simple1d1c PRIVATE ${MATH_LIBRARY})
    endif()
    add_test(NAME example_cuda_simple1d1c COMMAND cuda_simple1d1c)
endif()
