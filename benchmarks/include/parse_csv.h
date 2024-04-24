
#ifndef FINUFFT_PARSE_CSV_H
#define FINUFFT_PARSE_CSV_H

#include "finufft_benchmarks.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

struct CSVRow {
    enum class Precision {
        Single, // Single precision
        Double  // Double precision
    };
    Precision precision;
    std::int64_t ntr;
    std::int64_t M;
    std::int64_t N1;
    std::optional<std::int64_t> N2;
    std::optional<std::int64_t> N3;
    double eps = 10e-14;
    std::int32_t sort;
    std::int32_t kerval;
    std::int32_t kerpad;
    std::int32_t max_sp_size;
    std::int32_t threads;
    double upsampfac;

    friend std::ostream &operator<<(std::ostream &os, const CSVRow &row);
};

std::ostream &operator<<(std::ostream &os, const CSVRow::Precision &precision) {
    switch (precision) {
    case CSVRow::Precision::Single:
        os << "single";
        break;
    case CSVRow::Precision::Double:
        os << "double";
        break;
    }
    return os;
}

std::ostream &operator<<(std::ostream &os, const CSVRow &row) {
    os << "precision: " << row.precision << ", ntr: " << row.ntr
       << ", M: " << row.M << ", eps: " << row.eps << ", N1: " << row.N1
       << ", N2: " << row.N2.value_or(0) << ", N3: " << row.N3.value_or(0)
       << ", sort: " << row.sort << ", kerval: " << row.kerval
       << ", kerpad: " << row.kerpad << ", max_sp_size: " << row.max_sp_size
       << ", upsampfac: " << row.upsampfac << ", threads: " << row.threads;
    return os;
}

std::vector<CSVRow> parseCSV(const std::string &filename) {
    finufft_opts opts;
    finufft_default_opts(&opts);

    std::vector<CSVRow> rows;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {

        auto read_as_double = [](const std::string &field) -> std::int64_t {
            if (field.empty()) {
                return 0;
            }
            return static_cast<std::int64_t>(std::stod(field));
        };

        auto parse_precision =
            [](const std::string &field) -> CSVRow::Precision {
            if (field == "single") {
                return CSVRow::Precision::Single;
            } else if (field == "double") {
                return CSVRow::Precision::Double;
            } else {
                throw std::invalid_argument("Invalid precision: " + field);
            }
        };

        auto parse_eps = [](const std::string &field,
                            const CSVRow::Precision precision) -> double {
            if (field.empty()) {
                if (precision == CSVRow::Precision::Single) {
                    return 1e-6;
                } else {
                    return 1e-14;
                }
            }
            return std::stod(field);
        };

        std::istringstream ss(line);
        CSVRow row;
        std::string field;

        std::getline(ss, field, ',');
        row.precision = parse_precision(field);

        std::getline(ss, field, ',');
        row.ntr = std::stoi(field);

        std::getline(ss, field, ',');
        row.M = read_as_double(field);

        std::getline(ss, field, ',');
        row.eps = parse_eps(field, row.precision);

        std::getline(ss, field, ',');
        row.N1 = read_as_double(field);

        std::getline(ss, field, ',');
        row.N2 = field.empty()
                     ? std::nullopt
                     : std::optional<std::int64_t>(read_as_double(field));

        std::getline(ss, field, ',');
        row.N3 = field.empty()
                     ? std::nullopt
                     : std::optional<std::int64_t>(read_as_double(field));

        std::getline(ss, field, ',');
        row.sort = field.empty() ? opts.spread_sort : std::stoi(field);

        std::getline(ss, field, ',');
        row.kerval = field.empty() ? opts.spread_kerevalmeth : std::stoi(field);

        std::getline(ss, field, ',');
        row.kerpad = field.empty() ? opts.spread_kerpad : std::stoi(field);

        std::getline(ss, field, ',');
        row.upsampfac = field.empty() ? opts.upsampfac : std::stod(field);

        std::getline(ss, field, ',');
        row.max_sp_size =
            field.empty() ? opts.spread_max_sp_size : std::stoi(field);

        std::getline(ss, field, ',');
        row.threads = field.empty() ? opts.nthreads : std::stoi(field);
        rows.push_back(row);
    }

    return rows;
}

finufft_opts parse_opts(const CSVRow &row) {
    finufft_opts opts;
    finufft_default_opts(&opts);

    // Set optional fields in opts if they are present in the CSVRow
    opts.spread_sort = row.sort;
    opts.spread_kerevalmeth = row.kerval;
    opts.spread_kerpad = row.kerpad;
    opts.upsampfac = row.upsampfac;
    opts.spread_max_sp_size = row.max_sp_size;
    opts.nthreads = row.threads;
    return opts;
}
using FinufftBenchmarkVariant =
    std::variant<finufft_benchmark<1, float>, finufft_benchmark<2, float>,
                 finufft_benchmark<3, float>, finufft_benchmark<1, double>,
                 finufft_benchmark<2, double>, finufft_benchmark<3, double>>;

FinufftBenchmarkVariant convertToFinufftBenchmark(std::minstd_rand &rng,
                                                  const CSVRow &row) {
    finufft_opts opts = parse_opts(row);
    //    opts.nthreads = 1;
    // Create an array for the dimensions
    const auto dim0 = row.N1;
    const auto dim1 = row.N2.value_or(0);
    const auto dim2 = row.N3.value_or(0);
    double eps = row.eps;

    if (row.N2 && row.N3) {
        if (row.precision == CSVRow::Precision::Single) {
            // Create the finufft_benchmark object for 3 dimensions
            return finufft_benchmark<3, float>(rng, {dim0, dim1, dim2}, row.M,
                                               eps, opts);
        } else {
            // Create the finufft_benchmark object for 3 dimensions
            return finufft_benchmark<3, double>(rng, {dim0, dim1, dim2}, row.M,
                                                eps, opts);
        }
    } else if (row.N2) {
        if (row.precision == CSVRow::Precision::Single) {
            // Create the finufft_benchmark object for 2 dimensions
            return finufft_benchmark<2, float>(rng, {dim0, dim1}, row.M, eps,
                                               opts);
        } else {
            // Create the finufft_benchmark object for 2 dimensions
            return finufft_benchmark<2, double>(rng, {dim0, dim1}, row.M, eps,
                                                opts);
        }
    } else {
        if (row.precision == CSVRow::Precision::Single) {
            // Create the finufft_benchmark object for 1 dimension
            return finufft_benchmark<1, float>(rng, {dim0}, row.M, eps, opts);
        } else {
            // Create the finufft_benchmark object for 1 dimension
            return finufft_benchmark<1, double>(rng, {dim0}, row.M, eps, opts);
        }
    }
}

#endif // FINUFFT_PARSE_CSV_H
