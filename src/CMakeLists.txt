# CMakeLists.txt for building FINUFFT and related components

# Build the CPU FINUFFT library
if(FINUFFT_USE_CPU)
    # CPU-specific dependencies
    if((APPLE) AND (CMAKE_CXX_COMPILER_ID STREQUAL "GNU"))
        add_link_options("-ld_classic")
    endif()
    set(FINUFFT_FFTW_LIBRARIES)
    include(${PROJECT_SOURCE_DIR}/cmake/setupXSIMD.cmake)
    if(FINUFFT_USE_DUCC0)
        include(${PROJECT_SOURCE_DIR}/cmake/setupDUCC.cmake)
        set(FINUFFT_FFTLIBS ducc0)
    else()
        include(${PROJECT_SOURCE_DIR}/cmake/setupFFTW.cmake)
        set(FINUFFT_FFTLIBS ${FINUFFT_FFTW_LIBRARIES})
    endif()
    if(FINUFFT_USE_OPENMP)
        find_package(OpenMP COMPONENTS C CXX REQUIRED)
    endif()
    set(FINUFFT_FFTW_LIBRARIES "${FINUFFT_FFTW_LIBRARIES}" PARENT_SCOPE)
    set(FINUFFT_FFTLIBS "${FINUFFT_FFTLIBS}" PARENT_SCOPE)

    set(FINUFFT_SOURCES
        spreadinterp.cpp
        fft.cpp
        finufft_core.cpp
        c_interface.cpp
        finufft_utils.cpp
        utils.cpp
    )
    if(FINUFFT_BUILD_FORTRAN)
        list(APPEND FINUFFT_SOURCES ${PROJECT_SOURCE_DIR}/fortran/finufftfort.cpp)
    endif()
    if(NOT FINUFFT_STATIC_LINKING)
        add_library(finufft SHARED ${FINUFFT_SOURCES})
    else()
        add_library(finufft STATIC ${FINUFFT_SOURCES})
    endif()
    set_finufft_options(finufft)
    if(WIN32 AND FINUFFT_SHARED_LINKING)
        target_compile_definitions(finufft PRIVATE dll_EXPORTS FINUFFT_DLL)
    endif()
    find_library(MATH_LIBRARY m)
    if(MATH_LIBRARY)
        target_link_libraries(finufft PUBLIC ${MATH_LIBRARY})
    endif()
    if(FINUFFT_ENABLE_INSTALL)
        file(GLOB FINUFFT_PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/finufft*.h")
        set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")
    endif()
    list(APPEND INSTALL_TARGETS finufft)
endif()

# Build the CUDA accelerated library
if(FINUFFT_USE_CUDA)
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "compute_")
        message(
            FATAL_ERROR
            "CMAKE_CUDA_ARCHITECTURES must be a list of integers like 70;75;86, not strings like compute_89"
        )
    endif()
    include(${PROJECT_SOURCE_DIR}/cmake/setupCCCL.cmake)
    add_subdirectory(cuda)
endif()

# Propagate install targets to parent
set(INSTALL_TARGETS "${INSTALL_TARGETS}" PARENT_SCOPE)
