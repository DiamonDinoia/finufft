CPMAddPackage(
    NAME
    sphinx_cmake
    GITHUB_REPOSITORY
    k0ekk0ek/cmake-sphinx
    GIT_TAG
    e13c40a
    DOWNLOAD_ONLY
    YES
)
list(APPEND CMAKE_MODULE_PATH ${sphinx_cmake_SOURCE_DIR}/cmake/Modules)

find_package(Sphinx REQUIRED)
sphinx_add_docs(finufft_sphinx BUILDER html SOURCE_DIRECTORY ${FINUFFT_SOURCE_DIR}/docs)
