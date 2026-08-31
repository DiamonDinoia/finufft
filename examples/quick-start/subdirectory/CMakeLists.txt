# Fourth consume variant: a plain add_subdirectory, which is what a git submodule
# gives you. CPM hands CMake two arguments the submodule user does not,
# EXCLUDE_FROM_ALL and SYSTEM, so the FINUFFT targets stay out of `all` and its
# headers become system includes. This route passes neither, and neither does the
# published FetchContent recipe beside it: in both the FINUFFT targets join `all`
# and its install rules join the parent project's.
cmake_minimum_required(VERSION 3.25)
project(finufft_consume_subdirectory LANGUAGES CXX)

# Defaults to the checkout this file lives in, so the recipe runs with no
# arguments; a submodule user names the submodule instead, and CI passes the
# same path explicitly.
set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Local FINUFFT checkout to consume")

# The binary directory has to be named because the source lies outside this
# project's tree; a submodule sits inside it and needs no second argument. What
# is under test is the bare add_subdirectory, and that is identical either way.
add_subdirectory(${FINUFFT_SOURCE_DIR} finufft)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)

# On Windows the finufft DLL must sit next to the executable at runtime, exactly
# as in the three consumers beside this one.
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
