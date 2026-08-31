# Third consume variant: CPM, which docs/install.rst recommends first. CPM wraps
# FetchContent, but not identically: it adds EXCLUDE_FROM_ALL and SYSTEM, so the
# FINUFFT targets are not part of `all` and its headers arrive as system
# includes. That is the combination a CPM user gets and the sibling
# fetchcontent/ project does not exercise.
cmake_minimum_required(VERSION 3.25)
project(finufft_consume_cpm LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "" CACHE PATH "Local FINUFFT checkout to consume")
set(CPM_DOWNLOAD_VERSION "0.43.1" CACHE STRING "Version of CPM.cmake to use")

# The bootstrap docs/install.rst points at, taken from the checkout rather than
# copied, so this project and FINUFFT itself cannot drift to two CPM versions.
include(${FINUFFT_SOURCE_DIR}/cmake/setupCPM.cmake)

# CPM_<name>_SOURCE is CPM's own local override: the recipe below is the
# published one, and CI redirects it at the checkout instead of the release.
set(CPM_finufft_SOURCE "${FINUFFT_SOURCE_DIR}")
CPMAddPackage(
    NAME
    finufft
    GIT_REPOSITORY
    https://github.com/flatironinstitute/finufft.git
    GIT_TAG
    2.5.0
    GIT_SHALLOW
    YES
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)

# On Windows the finufft DLL must sit next to the executable at runtime, exactly
# as in the two consumers beside this one.
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
