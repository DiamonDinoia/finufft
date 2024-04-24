CPMAddPackage(
        NAME Tabulate
        GITHUB_REPOSITORY p-ranav/tabulate
        VERSION 1.5
        DOWNLOAD_ONLY Yes
)

if (Tabulate_ADDED)
    add_library(tabulate INTERFACE)
    target_include_directories(tabulate INTERFACE ${Tabulate_SOURCE_DIR}/include)
endif ()

CPMAddPackage(
        NAME Indicators
        GITHUB_REPOSITORY p-ranav/indicators
        VERSION 2.3
        DOWNLOAD_ONLY Yes
)

if (Indicators_ADDED)
    add_library(indicators INTERFACE)
    target_include_directories(indicators INTERFACE ${Indicators_SOURCE_DIR}/include)
endif ()

if (benchmark_ADDED)
    set_target_properties(benchmark PROPERTIES CXX_STANDARD 20)
endif()

add_library(finufft_benchlib INTERFACE)
target_include_directories(finufft_benchlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(finufft_benchlib INTERFACE finufft_static tabulate indicators)
set_target_properties(finufft_benchlib PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_20)
add_executable(finufft_benchmarks benchmakrs.cpp)
target_compile_definitions(finufft_benchmarks PRIVATE DATA_FILE="${CMAKE_CURRENT_SOURCE_DIR}/input.csv")
target_link_libraries(finufft_benchmarks PRIVATE finufft_benchlib benchmark::benchmark)
