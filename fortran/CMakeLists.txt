add_library(
    directft
    OBJECT
    directft/dirft1d.f
    directft/dirft1df.f
    directft/dirft2d.f
    directft/dirft2df.f
    directft/dirft3d.f
    directft/dirft3df.f
)

# FFT backend for Fortran examples follows core choice
if(FINUFFT_USE_DUCC0)
    set(_backend_targets ducc0::ducc0)
else()
    set(_backend_targets ${FINUFFT_REQUIRED_FFTW_TARGETS})
endif()

set(FORTRAN_EXAMPLES
    guru1d1
    nufft1d_demo
    nufft2d_demo
    nufft2dmany_demo
    nufft3d_demo
    simple1d1
)

foreach(EXAMPLE ${FORTRAN_EXAMPLES})
    add_executable(fort_${EXAMPLE} examples/${EXAMPLE}.f)
    add_executable(fort_${EXAMPLE}f examples/${EXAMPLE}f.f)
    target_link_libraries(fort_${EXAMPLE} PRIVATE directft finufft::finufft ${_backend_targets})
    target_link_libraries(fort_${EXAMPLE}f PRIVATE directft finufft::finufft ${_backend_targets})
endforeach()
