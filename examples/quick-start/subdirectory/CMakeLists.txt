cmake_minimum_required(VERSION 3.25)
project(finufft_consume_subdirectory LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/finufft" CACHE PATH "FINUFFT checkout, e.g. a git submodule")
add_subdirectory(${FINUFFT_SOURCE_DIR} finufft)

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
