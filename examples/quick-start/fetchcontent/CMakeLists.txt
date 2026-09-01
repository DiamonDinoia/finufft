cmake_minimum_required(VERSION 3.26)
project(finufft_consume_fetchcontent LANGUAGES CXX)

include(FetchContent)
FetchContent_Declare(finufft GIT_REPOSITORY https://github.com/flatironinstitute/finufft.git GIT_TAG master)
FetchContent_MakeAvailable(finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
add_custom_command(
    TARGET app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:app> $<TARGET_RUNTIME_DLLS:app>
    COMMAND_EXPAND_LISTS
)
