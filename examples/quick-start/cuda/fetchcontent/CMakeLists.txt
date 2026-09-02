cmake_minimum_required(VERSION 3.25)
project(cufinufft_consume_fetchcontent LANGUAGES CXX)

set(FINUFFT_USE_CUDA ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(finufft GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git GIT_TAG master)
FetchContent_MakeAvailable(finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::cufinufft)
add_custom_command(
    TARGET app
    POST_BUILD
    # copy_if_different needs a source, and TARGET_RUNTIME_DLLS is empty wherever there
    # are no DLLs. Listing app keeps the argument list valid; a self-copy does nothing.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:app> $<TARGET_FILE:app> $<TARGET_FILE_DIR:app>
    COMMAND_EXPAND_LISTS
)
