cmake_minimum_required(VERSION 3.25)
project(finufft_consume_cpm LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Checkout that supplies cmake/setupCPM.cmake")
set(CPM_DOWNLOAD_VERSION "0.43.1" CACHE STRING "Version of CPM.cmake to use")
include(${FINUFFT_SOURCE_DIR}/cmake/setupCPM.cmake)

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

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)

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
