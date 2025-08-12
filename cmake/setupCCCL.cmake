string(REPLACE "." ";" _cuda_ver_list ${CMAKE_CUDA_COMPILER_VERSION})
list(GET _cuda_ver_list 0 _cuda_major)
message(STATUS "CUDA ${_cuda_major} detected")

if(_cuda_major LESS 12)
    set(_cccl_tag "v${CUDA11_CCCL_VERSION}")
else()
    set(_cccl_tag "v${CUDA12_CCCL_VERSION}")
endif()

CPMAddPackage(
    NAME
    CCCL
    GITHUB_REPOSITORY
    NVIDIA/cccl
    GIT_TAG
    ${_cccl_tag}
)
