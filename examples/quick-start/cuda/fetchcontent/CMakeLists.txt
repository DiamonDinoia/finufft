cmake_minimum_required(VERSION 3.25)
project(cufinufft_consume_fetchcontent LANGUAGES CXX)

set(FINUFFT_GIT_REPOSITORY "https://github.com/flatironinstitute/finufft.git" CACHE STRING "")
set(FINUFFT_GIT_TAG "master" CACHE STRING "FINUFFT ref to fetch; set a release tag like v2.5.1 to pin")
set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(finufft GIT_REPOSITORY ${FINUFFT_GIT_REPOSITORY} GIT_TAG ${FINUFFT_GIT_TAG})
FetchContent_MakeAvailable(finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::cufinufft)
target_compile_features(app PRIVATE cxx_std_17)
