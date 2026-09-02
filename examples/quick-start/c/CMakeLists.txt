cmake_minimum_required(VERSION 3.25)
# CXX is enabled because a static libfinufft links with the C++ driver.
project(finufft_consume_c LANGUAGES C CXX)

find_package(finufft REQUIRED)

add_executable(app main.c)
target_link_libraries(app PRIVATE finufft::finufft)
# a C program calling cexp() and cabs() links libm itself
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(app PRIVATE ${MATH_LIBRARY})
endif()
add_custom_command(
    TARGET app
    POST_BUILD
    # copy_if_different needs a source, and TARGET_RUNTIME_DLLS is empty wherever there
    # are no DLLs. Listing app keeps the argument list valid; a self-copy does nothing.
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_RUNTIME_DLLS:app> $<TARGET_FILE:app> $<TARGET_FILE_DIR:app>
    COMMAND_EXPAND_LISTS
)
