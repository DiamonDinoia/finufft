cmake_minimum_required(VERSION 3.26)
# CXX is enabled because finufft is a C++ library: its config file looks for the C++
# OpenMP runtime, and a static libfinufft has to be linked with the C++ driver.
project(finufft_consume_fortran LANGUAGES Fortran CXX)

find_package(finufft REQUIRED)

add_executable(app main.f)
target_link_libraries(app PRIVATE finufft::finufft)
add_custom_command(
    TARGET app
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy -t $<TARGET_FILE_DIR:app> $<TARGET_RUNTIME_DLLS:app>
    COMMAND_EXPAND_LISTS
)
