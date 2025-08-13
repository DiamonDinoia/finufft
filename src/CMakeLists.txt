# --- src/CMakeLists.txt (CPU part) ---

if(FINUFFT_USE_CPU)
    if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        add_link_options("-ld_classic")
    endif()

    if(FINUFFT_USE_OPENMP)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
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
    add_library(finufft::finufft ALIAS finufft)
    set_finufft_options(finufft)

    if(FINUFFT_USE_DUCC0)
        include(${PROJECT_SOURCE_DIR}/cmake/setupDUCC.cmake)
        target_link_libraries(finufft PRIVATE ducc0::ducc0)
    else()
        include(${PROJECT_SOURCE_DIR}/cmake/setupFFTW.cmake)
        # Use FFTW while building finufft, and expose link-only dep to consumers
        target_link_libraries(finufft PRIVATE finufft::fftw)
        target_link_libraries(finufft INTERFACE $<LINK_ONLY:finufft::fftw>)
    endif()

    if(FINUFFT_USE_OPENMP)
        # Compile finufft with OpenMP; expose link-only dep to consumers so
        # they don't inherit OpenMP compile flags unless they opt-in.
        target_link_libraries(finufft PRIVATE OpenMP::OpenMP_CXX)
        target_link_libraries(finufft INTERFACE $<LINK_ONLY:OpenMP::OpenMP_CXX>)
    endif()

    # xsimd is header-only
    if(TARGET xsimd::xsimd)
        target_include_directories(finufft PRIVATE $<TARGET_PROPERTY:xsimd::xsimd,INTERFACE_INCLUDE_DIRECTORIES>)
    elseif(TARGET xsimd)
        target_include_directories(finufft PRIVATE $<TARGET_PROPERTY:xsimd,INTERFACE_INCLUDE_DIRECTORIES>)
    endif()

    if(WIN32 AND BUILD_SHARED_LIBS)
        target_compile_definitions(finufft PRIVATE dll_EXPORTS FINUFFT_DLL)
    endif()

    find_library(MATH_LIBRARY m)
    if(MATH_LIBRARY)
        target_link_libraries(finufft PRIVATE ${MATH_LIBRARY})
    endif()

    # Build-tree umbrella for tests/examples (not installed)
    add_library(finufft_all INTERFACE)
    target_link_libraries(finufft_all INTERFACE finufft)
    if(NOT FINUFFT_USE_DUCC0)
        target_link_libraries(finufft_all INTERFACE finufft::fftw)
    else()
        target_link_libraries(finufft_all INTERFACE ducc0::ducc0)
    endif()
    add_library(finufft::all ALIAS finufft_all)

    # Install only finufft
    file(GLOB FINUFFT_PUBLIC_HEADERS "${PROJECT_SOURCE_DIR}/include/finufft*.h")
    set_target_properties(finufft PROPERTIES PUBLIC_HEADER "${FINUFFT_PUBLIC_HEADERS}")

    if(FINUFFT_ENABLE_INSTALL)
        if(TARGET finufft_fftw)
            install(TARGETS finufft_fftw EXPORT finufftTargets)
        endif()

        include(GNUInstallDirs)
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
