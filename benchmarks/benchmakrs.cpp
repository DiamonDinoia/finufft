#include "finufft_benchmarks.h"
#include "parse_csv.h"
#include <indicators/progress_bar.hpp>
#include <ranges>
#include "benchmark.h"
// std visit to call the correct benchmark
constexpr auto RUNS = 5;

int main(int argc, char **argv) {

    using namespace indicators;

    std::minstd_rand gen(42);
    const auto data = parseCSV(DATA_FILE);

    ProgressBar bar{
        option::BarWidth{50},
        option::Start{"["},
        option::Fill{"="},
        option::Lead{">"},
        option::Remainder{" "},
        option::End{"]"},
        option::PostfixText{"Running benchmarks"},
        option::ForegroundColor{Color::blue},
        option::ShowElapsedTime{true},
        option::ShowRemainingTime{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}};
    bar.set_option(option::ShowPercentage(true));
    bar.set_progress(0);
    const auto step = 100.0 / static_cast<double>(data.size() - 1);
    BenchmarkReporter reporter;
    for (const auto &row : data) {
        {
            std::stringstream ss;
            ss << row;
            bar.set_option(option::PostfixText{ss.str()});
            bar.set_progress(bar.current());
        }
        auto benchmark_case = convertToFinufftBenchmark(gen, row);
        std::visit(
            [&](auto &&arg) {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (T::Dimensions == 1) {
                    reporter.ReportRun(
                        row, 1, [&arg] { return arg.benchmark_finufft1d1(); });
                    reporter.ReportRun(
                        row, 2, [&arg] { return arg.benchmark_finufft1d2(); });
                } else if constexpr (T::Dimensions == 2) {
                    reporter.ReportRun(
                        row, 1, [&arg] { return arg.benchmark_finufft2d1(); });
                    reporter.ReportRun(
                        row, 2, [&arg] { return arg.benchmark_finufft2d2(); });
                } else if constexpr (T::Dimensions == 3) {
                    reporter.ReportRun(
                        row, 1, [&arg] { return arg.benchmark_finufft3d1(); });
                    reporter.ReportRun(
                        row, 2, [&arg] { return arg.benchmark_finufft3d2(); });
                }
            },
            benchmark_case);
        bar.set_progress(
            static_cast<size_t>(static_cast<double>(bar.current()) + step));
    }
    bar.set_progress(
        static_cast<size_t>(static_cast<double>(bar.current()) + step));
    bar.mark_as_completed();
    reporter.print();
    return 0;
}