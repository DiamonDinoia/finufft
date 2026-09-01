cmake_minimum_required(VERSION 3.19)
project(cufinufft_consume LANGUAGES CXX)

find_package(finufft REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE finufft::cufinufft)
target_compile_features(app PRIVATE cxx_std_17)
