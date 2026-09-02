cmake_minimum_required(VERSION 3.25)
project(finufft_consume_cpm LANGUAGES CXX)

set(CPM_DOWNLOAD_VERSION 0.43.1)
if(CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif()
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    file(
        DOWNLOAD https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
    )
endif()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage("gh:flatironinstitute/finufft#master")

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
add_custom_command(
    TARGET app
    POST_BUILD
    # app pads the copy list so a destination argument survives where
    # TARGET_RUNTIME_DLLS is empty, which is every platform without DLLs.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:app> $<TARGET_FILE:app> $<TARGET_FILE_DIR:app>
    COMMAND_EXPAND_LISTS
)
