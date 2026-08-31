# Third quick-start variant: CPM, which docs/install.rst recommends first. CPM
# wraps FetchContent but not identically: EXCLUDE_FROM_ALL keeps the FINUFFT
# targets out of `all` and SYSTEM turns its headers into system includes.
#
#   cmake -S . -B build && cmake --build build && ./build/app
#
# CI redirects the clone at the checkout with CPM's own documented override,
# -DCPM_finufft_SOURCE=<path>, so nothing in the tagged block below is
# CI-specific. FINUFFT_SOURCE_DIR only says where to find the CPM bootstrap;
# drop it and add the four lines from the CPM wiki instead.
cmake_minimum_required(VERSION 3.25)
project(finufft_consume_cpm LANGUAGES CXX)

# Defaults to the checkout this file lives in, so the recipe runs with no
# arguments; CI passes the same path explicitly.
set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Checkout that supplies cmake/setupCPM.cmake")
set(CPM_DOWNLOAD_VERSION "0.43.1" CACHE STRING "Version of CPM.cmake to use")

# The bootstrap docs/install.rst points at, taken from the checkout rather than
# copied, so this project and FINUFFT itself cannot drift to two CPM versions.
include(${FINUFFT_SOURCE_DIR}/cmake/setupCPM.cmake)

# sphinx tag (don't remove): @quickstart_cpm_start
set(FINUFFT_GIT_REPOSITORY "https://github.com/flatironinstitute/finufft.git" CACHE STRING "")
set(FINUFFT_GIT_TAG "master" CACHE STRING "FINUFFT ref to fetch; set a release tag like v2.5.1 to pin")

CPMAddPackage(
    NAME
    finufft
    GIT_REPOSITORY
    ${FINUFFT_GIT_REPOSITORY}
    GIT_TAG
    ${FINUFFT_GIT_TAG}
    GIT_SHALLOW
    YES
    EXCLUDE_FROM_ALL
    YES
    SYSTEM
)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
# sphinx tag (don't remove): @quickstart_cpm_end
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
