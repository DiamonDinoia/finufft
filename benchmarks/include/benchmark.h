
#ifndef FINUFFT_BENCHMARK_H
#define FINUFFT_BENCHMARK_H

#include "benchmark.h"
#include <algorithm>
#include <any>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <parse_csv.h>
#include <ranges>
#include <tabulate/table.hpp>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __attribute__((always_inline))
#elif defined(_MSC_VER) && !defined(__clang__)
#define BENCHMARK_ALWAYS_INLINE __forceinline
#define __func__ __FUNCTION__
#else
#define BENCHMARK_ALWAYS_INLINE
#endif

template <class Tp>
constexpr inline BENCHMARK_ALWAYS_INLINE
    typename std::enable_if<std::is_trivially_copyable<Tp>::value &&
                            (sizeof(Tp) <= sizeof(Tp *))>::type
    DoNotOptimize(Tp &value) {
    asm volatile("" : "+m,r"(value) : : "memory");
}

template <class Tp>
inline BENCHMARK_ALWAYS_INLINE
    typename std::enable_if<!std::is_trivially_copyable<Tp>::value ||
                            (sizeof(Tp) > sizeof(Tp *))>::type
    DoNotOptimize(Tp &value) {
    asm volatile("" : "+m"(value) : : "memory");
}

template <class Tp>
inline BENCHMARK_ALWAYS_INLINE
    typename std::enable_if<std::is_trivially_copyable<Tp>::value &&
                            (sizeof(Tp) <= sizeof(Tp *))>::type
    DoNotOptimize(Tp &&value) {
    asm volatile("" : "+m,r"(value) : : "memory");
}

template <class Tp>
inline BENCHMARK_ALWAYS_INLINE
    typename std::enable_if<!std::is_trivially_copyable<Tp>::value ||
                            (sizeof(Tp) > sizeof(Tp *))>::type
    DoNotOptimize(Tp &&value) {
    asm volatile("" : "+m"(value) : : "memory");
}

inline BENCHMARK_ALWAYS_INLINE void ClobberMemory() {
    std::atomic_signal_fence(std::memory_order_acq_rel);
}

template <std::uint64_t exact_iterations = 0, std::uint64_t min_iterations = 5,
          std::uint64_t max_iterations = 10000>
std::pair<double, double> benchmarkFunction(auto func) {
    using Clock =
        std::conditional_t<std::chrono::high_resolution_clock::is_steady,
                           std::chrono::high_resolution_clock,
                           std::chrono::steady_clock>;
    using duration_t = std::chrono::duration<double>;

    std::vector<duration_t> times;

    const auto measure = [&func, &times] {
        ClobberMemory();
        const auto start = Clock::now();
        DoNotOptimize(func());
        const auto end = Clock::now();
        ClobberMemory();
        times.emplace_back(end - start);
    };

    const auto metrics = [&times]() -> std::pair<double, double> {
        std::vector<double> seconds;
        seconds.reserve(times.size());
        std::transform(times.begin(), times.end(), std::back_inserter(seconds),
                       [](const duration_t &x) { return x.count(); });

        const double mean = std::reduce(seconds.begin(), seconds.end()) /
                            static_cast<double>(seconds.size());

        std::vector<double> diff_squares;
        diff_squares.reserve(seconds.size());
        std::ranges::transform(seconds, std::back_inserter(diff_squares),
                               [mean](const double time) {
                                   return (time - mean) * (time - mean);
                               });

        const double stddev =
            std::sqrt(std::reduce(diff_squares.begin(), diff_squares.end()) /
                      static_cast<double>(diff_squares.size()));
        const double standard_error =
            stddev / std::sqrt(static_cast<double>(diff_squares.size()));
        const double relative_error = standard_error * 100 / mean;

        return {mean, relative_error};
    };

    constexpr auto iterations =
        exact_iterations == 0 ? min_iterations : exact_iterations;
    times.reserve(iterations);
    for (std::uint64_t i = 0; i < iterations; ++i) {
        measure();
    }

    for (std::uint64_t i = iterations; i < max_iterations; ++i) {
        const auto [mean, error] = metrics();
        if (mean * i > 10 || error < 1.0) {
            return {mean, error};
        }
        measure();
    }
    return metrics();
}

class BenchmarkReporter {
  public:
    void ReportRun(const CSVRow &row, std::int32_t type, auto func) {
        if (table.size() == 0) {
            table.add_row({
                "type",
                "ntr",
                "precision",
                "M",
                "N1",
                "N2",
                "N3",
                "eps",
                "upsampfac",
                "sort",
                "kerval",
                "kerpad",
                "max_sp_size",
                "threads",
                "NU/s",
                "Mean (s)",
                "Error (%)",
            });
        }

        const auto [mean, error] = benchmarkFunction(std::move(func));

        auto parse_eps = [&row]() {
            std::stringstream ss;
            ss << std::scientific << std::setprecision(0) << row.eps;
            return ss.str();
        };

        auto parse_throughput = [&row, &mean]() {
            std::stringstream ss;
            ss << std::scientific << std::setprecision(0)
               << static_cast<double>(row.M) / mean;
            return ss.str();
        };

        auto parse_mean = [&mean]() {
            std::stringstream ss;
            ss << std::scientific << std::setprecision(4) << mean;
            return ss.str();
        };

        auto parse_upsampfac = [&row]() {
            std::stringstream ss;
            ss << std::fixed << std::setprecision(2) << row.upsampfac;
            return ss.str();
        };

        auto parse_precision = [&row]() {
            std::stringstream ss;
            ss << row.precision;
            return ss.str();
        };

        table.add_row(
            {std::to_string(type), std::to_string(row.ntr), parse_precision(),
             std::to_string(row.M), std::to_string(row.N1),
             std::to_string(row.N2.value_or(0)),
             std::to_string(row.N3.value_or(0)), parse_eps(), parse_upsampfac(),
             std::to_string(row.sort), std::to_string(row.kerval),
             std::to_string(row.kerpad), std::to_string(row.max_sp_size),
             std::to_string(row.threads), parse_throughput(), parse_mean(),
             std::to_string(error) + "%"});
    }

    void print(std::ostream &os = std::cout) {
        if (table.size() == 0) {
            return;
        }
        for (auto i = 0; i < 2; i++) {
            table.column(table[0].size() - 2 - i)
                .format()
                .color(tabulate::Color::blue)
                .font_style({tabulate::FontStyle::bold});
        }
        table.column(table[0].size() - 1)
            .format()
            .color(tabulate::Color::red)
            .font_style({tabulate::FontStyle::bold});
        table.print(os);
    }

  private:
    tabulate::Table table;
};

#endif // FINUFFT_BENCHMARK_H
