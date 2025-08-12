CPMAddPackage(
    NAME
    benchmark
    GITHUB_REPOSITORY
    google/benchmark
    VERSION
    1.8.3
    OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
)

add_executable(foldrescale foldrescale.cpp)
target_link_libraries(foldrescale PRIVATE finufft::finufft xsimd benchmark::benchmark)
target_compile_features(foldrescale PRIVATE cxx_std_17)
enable_asan(foldrescale)

add_executable(padding padding.cpp)
target_compile_features(padding PRIVATE cxx_std_17)
target_link_libraries(padding PRIVATE finufft::finufft xsimd)
# Reuse FINUFFT_ARCH_FLAGS instead of hardcoding
if(FINUFFT_ARCH_FLAGS)
    finufft_target_add_flags_if_supported(padding PRIVATE ${FINUFFT_ARCH_FLAGS})
endif()
enable_asan(padding)
