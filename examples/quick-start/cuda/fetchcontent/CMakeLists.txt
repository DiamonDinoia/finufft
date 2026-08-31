# The GPU twin of ../../fetchcontent: FetchContent instead of an installed
# find_package, linking finufft::cufinufft.
#
# CI redirects the clone at the checkout with FetchContent's own documented
# override, -DFETCHCONTENT_SOURCE_DIR_FINUFFT=<path>.
cmake_minimum_required(VERSION 3.25)
# CXX only: FINUFFT runs enable_language(CUDA) itself when FINUFFT_USE_CUDA is
# ON, so the consumer never has to.
project(cufinufft_consume_fetchcontent LANGUAGES CXX)

set(FINUFFT_GIT_REPOSITORY "https://github.com/flatironinstitute/finufft.git" CACHE STRING "")
set(FINUFFT_GIT_TAG "master" CACHE STRING "FINUFFT ref to fetch; set a release tag like v2.5.1 to pin")
set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(finufft GIT_REPOSITORY ${FINUFFT_GIT_REPOSITORY} GIT_TAG ${FINUFFT_GIT_TAG})
FetchContent_MakeAvailable(finufft)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::cufinufft)
target_compile_features(app PRIVATE cxx_std_17)
