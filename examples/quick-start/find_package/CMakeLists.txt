cmake_minimum_required(VERSION 3.19)
project(finufft_consume LANGUAGES CXX)

find_package(finufft REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)

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
