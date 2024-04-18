
#include "benchmarks.h"
#include "parse_csv.h"
#include <benchmark/benchmark.h>
#include <iostream>

// std visit to call the correct benchmark
constexpr auto RUNS = 5;

int main(int argc, char** argv) {
    std::minstd_rand gen(42);
    const auto data = parseCSV(DATA_FILE);
    for (const auto &row : data) {
        benchmark::RegisterBenchmark((std::stringstream{} << row).str(), [&gen, &row](benchmark::State &state) {
            state.PauseTiming();
            auto benchmark = convertToFinufftBenchmark(gen, row);
            state.ResumeTiming();
            std::visit(
                [](auto &&arg) {
                    using T = std::decay_t<decltype(arg)>;
                    if constexpr (std::is_same_v<T, finufft_benchmark<1>>) {
                        benchmark::DoNotOptimize(arg.benchmark_finufft1d1());
                        benchmark::DoNotOptimize(arg.benchmark_finufft1d2());
                    } else if constexpr (std::is_same_v<T,
                                                        finufft_benchmark<2>>) {
                        benchmark::DoNotOptimize(arg.benchmark_finufft2d1());
                        benchmark::DoNotOptimize(arg.benchmark_finufft2d2());
                    } else if constexpr (std::is_same_v<T,
                                                        finufft_benchmark<3>>) {
                        benchmark::DoNotOptimize(arg.benchmark_finufft3d1());
                        benchmark::DoNotOptimize(arg.benchmark_finufft3d2());
                    }
                },
                benchmark);
        })->Unit(benchmark::kMillisecond);
    }
    benchmark::Initialize(&argc, argv); // Initialize the benchmark
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}