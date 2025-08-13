set(EXAMPLES
    guru1d1
    guru2d1_adjoint
    guru1d1f
    guru2d1
    many1d1
    simple1d1
    simple1d1f
    simulplans1d1
)
set(EXAMPLES_OPENMP threadsafe1d1 threadsafe2d2f)
set(EXAMPLES_C guru1d1c simple1d1c simple1d1cf)

foreach(EXAMPLE ${EXAMPLES})
    add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
    target_link_libraries(${EXAMPLE} PRIVATE finufft::finufft)
    if(CMAKE_PROJECT_NAME STREQUAL "FINUFFT")
        enable_asan(${EXAMPLE})
    endif()
endforeach()

foreach(EXAMPLE ${EXAMPLES_C})
    add_executable(${EXAMPLE} ${EXAMPLE}.c)
    target_link_libraries(${EXAMPLE} PRIVATE finufft::finufft)
    if(CMAKE_PROJECT_NAME STREQUAL "FINUFFT")
        enable_asan(${EXAMPLE})
    endif()
endforeach()

# OpenMP-using examples must opt-in to OpenMP to get compile flags
if(FINUFFT_USE_OPENMP)
    find_package(OpenMP REQUIRED COMPONENTS CXX)
    foreach(EXAMPLE ${EXAMPLES_OPENMP})
        add_executable(${EXAMPLE} ${EXAMPLE}.cpp)
        target_link_libraries(${EXAMPLE} PRIVATE finufft::finufft OpenMP::OpenMP_CXX)
        if(CMAKE_PROJECT_NAME STREQUAL "FINUFFT")
            enable_asan(${EXAMPLE})
        endif()
    endforeach()
endif()
