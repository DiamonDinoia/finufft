# Second quick-start variant: FetchContent, which CMake ships. It clones FINUFFT
# at configure time and builds it as part of this project.
#
#   cmake -S . -B build && cmake --build build && ./build/app
#
# CI configures this project unchanged and redirects the clone at the checkout
# with FetchContent's own documented override,
# -DFETCHCONTENT_SOURCE_DIR_FINUFFT=<path>, so nothing below is CI-specific.
cmake_minimum_required(VERSION 3.25)
project(finufft_consume_fetchcontent LANGUAGES C CXX)

# sphinx tag (don't remove): @quickstart_fetchcontent_start
set(FINUFFT_GIT_REPOSITORY "https://github.com/flatironinstitute/finufft.git" CACHE STRING "")
set(FINUFFT_GIT_TAG "master" CACHE STRING "FINUFFT ref to fetch; set a release tag like v2.5.1 to pin")

include(FetchContent)
FetchContent_Declare(finufft GIT_REPOSITORY ${FINUFFT_GIT_REPOSITORY} GIT_TAG ${FINUFFT_GIT_TAG})
FetchContent_MakeAvailable(finufft)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
# sphinx tag (don't remove): @quickstart_fetchcontent_end
target_compile_features(app PRIVATE cxx_std_17)

# On Windows the finufft DLL must sit next to the executable at runtime. A
# subproject build leaves it under _deps, where the loader does not look, and the
# app then exits 127.
if(WIN32)
    get_target_property(_finufft_type finufft TYPE)
    if(_finufft_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(
            TARGET app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:finufft> $<TARGET_FILE_DIR:app>
        )
    endif()
endif()
