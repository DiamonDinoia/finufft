# CPU FINUFFT library
if(FINUFFT_USE_CPU)
    if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_link_options("-ld_classic")
    endif()

    include(${PROJECT_SOURCE_DIR}/cmake/setupXSIMD.cmake)

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

    add_library(finufft ${FINUFFT_SOURCES})
    # Backend: DUCC0 vendored as OBJECT files, or FFTW via findFFTW
    if(FINUFFT_USE_DUCC0)
        include(${PROJECT_SOURCE_DIR}/cmake/setupDUCC.cmake)
        # Add DUCC0 objects into finufft
        target_link_libraries(finufft PRIVATE ducc0::ducc0)
    else()
        include(${PROJECT_SOURCE_DIR}/cmake/setupFFTW.cmake)
        target_link_libraries(finufft PRIVATE ${FINUFFT_REQUIRED_FFTW_TARGETS})
    endif()

    add_library(finufft::finufft ALIAS finufft)
    set_finufft_options(finufft)

    if(WIN32 AND BUILD_SHARED_LIBS)
        target_compile_definitions(finufft PRIVATE dll_EXPORTS FINUFFT_DLL)
    endif()

    # Link libm if present
    find_library(MATH_LIBRARY m)
    if(MATH_LIBRARY)
        # For static builds, we still keep libm PRIVATE — it’s ubiquitous.
        target_link_libraries(finufft PRIVATE ${MATH_LIBRARY})
    endif()

    # xsimd headers only
    if(TARGET xsimd::xsimd)
        target_include_directories(finufft PRIVATE $<TARGET_PROPERTY:xsimd::xsimd,INTERFACE_INCLUDE_DIRECTORIES>)
    elseif(TARGET xsimd)
        target_include_directories(finufft PRIVATE $<TARGET_PROPERTY:xsimd,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    # OpenMP on finufft (PRIVATE for shared; PUBLIC if you want to propagate for static — optional)
    if(FINUFFT_USE_OPENMP)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        if(BUILD_SHARED_LIBS)
            target_link_libraries(finufft PRIVATE OpenMP::OpenMP_CXX)
        else()
            # Static build: propagate OpenMP to dependents to avoid missing -fopenmp at final link
            target_link_libraries(finufft PUBLIC OpenMP::OpenMP_CXX)
        endif()
    endif()

    # FFT backend: when using FFTW path, keep targets PRIVATE (users don’t need to see them)
    if(NOT FINUFFT_USE_DUCC0)
        target_link_libraries(finufft PRIVATE ${FINUFFT_REQUIRED_FFTW_TARGETS})
    endif()

    # Public headers
    file(GLOB FINUFFT_PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/finufft*.h")
    set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")

    # Install only finufft
    if(FINUFFT_ENABLE_INSTALL)
        install(
            TARGETS finufft
            EXPORT finufftTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
            PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        )
    endif()
endif()

# CUDA library
if(FINUFFT_USE_CUDA)
    if(CMAKE_CUDA_ARCHITECTURES MATCHES "compute_")
        message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be integers like 70;75;86, not compute_XX")
    endif()
    include(${PROJECT_SOURCE_DIR}/cmake/setupCCCL.cmake)
    add_subdirectory(cuda)
endif()
