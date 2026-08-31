# Quick start: use an installed FINUFFT from another project.
#
#   cmake -S . -B build -DCMAKE_PREFIX_PATH=<install prefix>
#   cmake --build build && ./build/app
#
# (or -Dfinufft_DIR=<install prefix>/lib/cmake/finufft). The directories beside
# this one build the same program the other four ways: fetchcontent/, cpm/,
# subdirectory/ and makefile/.
#
# tools/ci/install-test.sh runs this project on every pull request, so the recipe
# cannot rot. It is also the regression guard for missing transitive headers, a
# missing find_dependency() in the installed config, and the Windows DLL runtime
# path.
cmake_minimum_required(VERSION 3.19)
# CXX only, deliberately: a C++-only consumer is the common case and the strict
# one. Enabling C here would hide a find_dependency() in finufftConfig.cmake that
# asks for a C component the consumer never enables.
project(finufft_consume LANGUAGES CXX)

# sphinx tag (don't remove): @quickstart_find_package_start
find_package(finufft REQUIRED)

add_executable(app ../main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
# sphinx tag (don't remove): @quickstart_find_package_end
target_compile_features(app PRIVATE cxx_std_17)

# On Windows the finufft DLL must sit next to the executable at runtime.
if(WIN32 AND TARGET finufft::finufft)
    get_target_property(_finufft_type finufft::finufft TYPE)
    if(_finufft_type STREQUAL "SHARED_LIBRARY")
        add_custom_command(
            TARGET app
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:finufft::finufft> $<TARGET_FILE_DIR:app>
        )
    endif()
endif()
