cmake_minimum_required(VERSION 3.25)
project(finufft_consume_subdirectory LANGUAGES CXX)

set(FINUFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." CACHE PATH "Local FINUFFT checkout to consume")
add_subdirectory(${FINUFFT_SOURCE_DIR} finufft)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::finufft)
target_compile_features(app PRIVATE cxx_std_17)

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
