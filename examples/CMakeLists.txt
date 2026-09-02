set(EXAMPLES
    guru1d1
    guru2d1_adjoint
    guru1d1f
    guru2d1
    gurumany1d1
    many1d1
    simple1d1
    simple1d1f
    simple2d1
    simulplans1d1
    spreadinterponly1d
)
set(EXAMPLES_OPENMP threadsafe1d1 threadsafe2d2f)
set(EXAMPLES_C guru1d1c simple1d1c simple1d1cf)

find_library(MATH_LIBRARY m)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_compile_features(${EXAMPLE} PRIVATE cxx_std_14)
    target_link_libraries(${EXAMPLE} PRIVATE finufft)
    if(PROJECT_IS_TOP_LEVEL)
        enable_asan(${EXAMPLE})
    endif()
    add_test(NAME example_${EXAMPLE} COMMAND ${EXAMPLE})
endforeach()

# The C examples need C99 complex, which MSVC lacks. HAS_C_COMPLEX is probed once in
# cmake/toolchain.cmake. Minimal not working example: https://marco.godbolt.org/z/cf3YroEcv

if(HAS_C_COMPLEX)
    foreach(EXAMPLE ${EXAMPLES_C})
        add_executable(${EXAMPLE} ${EXAMPLE}.c)
        target_link_libraries(${EXAMPLE} PRIVATE finufft)
        if(PROJECT_IS_TOP_LEVEL)
            enable_asan(${EXAMPLE})
        endif()
        if(MATH_LIBRARY)
            target_link_libraries(${EXAMPLE} PRIVATE ${MATH_LIBRARY})
        endif()
                add_test(NAME example_${EXAMPLE} COMMAND ${EXAMPLE})
    endforeach()
endif()

if(FINUFFT_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    foreach(EXAMPLE ${EXAMPLES_OPENMP})
        add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
        target_link_libraries(${EXAMPLE} PRIVATE finufft OpenMP::OpenMP_CXX)
        target_compile_features(${EXAMPLE} PRIVATE cxx_std_11)
        if(PROJECT_IS_TOP_LEVEL)
            enable_asan(${EXAMPLE})
        endif()
        add_test(NAME example_${EXAMPLE} COMMAND ${EXAMPLE})
    endforeach()
endif()
