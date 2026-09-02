cmake_minimum_required(VERSION 3.25)
project(finufft_consume_subdirectory LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/finufft" CACHE PATH "FINUFFT checkout, e.g. a git submodule")
add_subdirectory(${FINUFFT_SOURCE_DIR} finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
add_custom_command(
    TARGET app
    POST_BUILD
    # copy_if_different needs a source, and TARGET_RUNTIME_DLLS is empty wherever there
    # are no DLLs. Listing app keeps the argument list valid; a self-copy does nothing.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:app> $<TARGET_FILE:app> $<TARGET_FILE_DIR:app>
    COMMAND_EXPAND_LISTS
)
