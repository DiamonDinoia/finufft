cmake_minimum_required(VERSION 3.26)
project(finufft_consume_subdirectory LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/finufft" CACHE PATH "FINUFFT checkout, e.g. a git submodule")
add_subdirectory(${FINUFFT_SOURCE_DIR} finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
add_custom_command(
    TARGET app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:app> $<TARGET_RUNTIME_DLLS:app>
    COMMAND_EXPAND_LISTS
)
