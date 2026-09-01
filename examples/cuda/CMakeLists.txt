file(GLOB example_src "*.cpp")

foreach(srcfile ${example_src})
    string(REPLACE ".cpp" "" executable ${srcfile})
    get_filename_component(executable ${executable} NAME)
    add_executable(${executable} ${srcfile})
    target_include_directories(${executable} PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
    target_link_libraries(${executable} PRIVATE cufinufft CUDA::cufft CUDA::cudart $<BUILD_INTERFACE:finufft_common>)
    target_compile_features(${executable} PRIVATE cxx_std_17)
    add_test(NAME example_${executable} COMMAND ${executable})
endforeach()

# Windows does not have complex support in the c compiler we need to compile in c++ mode
# https://en.cppreference.com/w/c/numeric/complex.html
# when that happens we need to skip c examples as they will not work
include(CheckCSourceCompiles)
# gersemi: off
check_c_source_compiles("
        #include <complex.h>
        int main() {
            double complex z = 1.0 + 2.0*I;
            double x = creal(z);
            return 0;
        }
    " HAS_C_COMPLEX)
# gersemi: on

if(HAS_C_COMPLEX)
    find_library(MATH_LIBRARY m)
    add_executable(cuda_simple1d1c simple1d1c.c)
    target_include_directories(cuda_simple1d1c PUBLIC ${CUFINUFFT_INCLUDE_DIRS})
    target_link_libraries(cuda_simple1d1c PRIVATE cufinufft CUDA::cufft CUDA::cudart)
    if(MATH_LIBRARY)
        target_link_libraries(cuda_simple1d1c PRIVATE ${MATH_LIBRARY})
    endif()
    add_test(NAME example_cuda_simple1d1c COMMAND cuda_simple1d1c)
endif()
