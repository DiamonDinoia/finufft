CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.8.3
        OPTIONS
        "BENCHMARK_ENABLE_TESTING Off"
        "HAVE_STD_REGEX On"
        "RUN_HAVE_STD_REGEX On"
)

if (benchmark_ADDED)
    # patch benchmark target
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 17)
endif()

add_library(finufft_benchlib INTERFACE)
target_include_directories(finufft_benchlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(finufft_benchlib INTERFACE finufft_static)
set_target_properties(finufft_benchlib PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_20)
target_compile_options(finufft_benchlib INTERFACE -fno-omit-frame-pointer -fno-inline)
add_executable(finufft_benchmarks benchmakrs.cpp)
target_compile_definitions(finufft_benchmarks PRIVATE DATA_FILE="${CMAKE_CURRENT_BINARY_DIR}/input.csv")
target_link_libraries(finufft_benchmarks PRIVATE finufft_benchlib benchmark::benchmark)
