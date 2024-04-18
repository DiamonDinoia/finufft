
#ifndef FINUFFT_PARSE_CSV_H
#define FINUFFT_PARSE_CSV_H

#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

struct CSVRow {
    std::int64_t type;
    std::int64_t ntr;
    std::int64_t M;
    std::optional<double> eps;
    std::int64_t N1;
    std::optional<std::int64_t> N2;
    std::optional<std::int64_t> N3;
    std::optional<std::int32_t> sort;
    std::optional<std::int32_t> kerval;
    std::optional<std::int32_t> kerpad;
    std::optional<std::int32_t> max_sp_size;
    std::optional<double> upsampfac;

    friend std::ostream &operator<<(std::ostream &os, const CSVRow &row) {
        os << "type: " << row.type << ", ntr: " << row.ntr << ", M: " << row.M
           << ", eps: " << row.eps.value_or(0) << ", N1: " << row.N1
           << ", N2: " << row.N2.value_or(0) << ", N3: " << row.N3.value_or(0)
           << ", sort: " << row.sort.value_or(0) << ", kerval: "
           << row.kerval.value_or(0) << ", kerpad: " << row.kerpad.value_or(0)
           << ", max_sp_size: " << row.max_sp_size.value_or(0)
           << ", upsampfac: " << row.upsampfac.value_or(0);
        return os;
    }
};

std::vector<CSVRow> parseCSV(const std::string &filename) {
    std::vector<CSVRow> rows;
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::string line;

    // Skip the header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream ss(line);
        CSVRow row;
        std::string field;

        std::getline(ss, field, ',');
        row.type = std::stoi(field);

        std::getline(ss, field, ',');
        row.ntr = std::stoi(field);

        std::getline(ss, field, ',');
        row.M = std::stol(field);

        std::getline(ss, field, ',');
        row.eps = field.empty() ? std::nullopt
                                : std::optional<double>(std::stod(field));

        std::getline(ss, field, ',');
        row.N1 = std::stol(field);

        std::getline(ss, field, ',');
        row.N2 = field.empty() ? std::nullopt
                               : std::optional<std::int64_t>(std::stol(field));

        std::getline(ss, field, ',');
        row.N3 = field.empty() ? std::nullopt
                               : std::optional<std::int64_t>(std::stol(field));

        std::getline(ss, field, ',');
        row.sort = field.empty()
                       ? std::nullopt
                       : std::optional<std::int32_t>(std::stoi(field));

        std::getline(ss, field, ',');
        row.kerval = field.empty()
                         ? std::nullopt
                         : std::optional<std::int32_t>(std::stoi(field));

        std::getline(ss, field, ',');
        row.kerpad = field.empty()
                         ? std::nullopt
                         : std::optional<std::int32_t>(std::stoi(field));

        std::getline(ss, field, ',');
        row.upsampfac = field.empty() ? std::nullopt
                                      : std::optional<double>(std::stod(field));

        std::getline(ss, field, ',');
        row.max_sp_size = field.empty()
                              ? std::nullopt
                              : std::optional<std::int32_t>(std::stod(field));

        rows.push_back(row);
    }

    return rows;
}

finufft_opts parse_opts(const CSVRow &row) {
    finufft_opts opts;
    finufft_default_opts(&opts);

    // Set optional fields in opts if they are present in the CSVRow
    if (row.sort)
        opts.spread_sort = *row.sort;
    if (row.kerval)
        opts.spread_kerevalmeth = *row.kerval;
    if (row.kerpad)
        opts.spread_kerpad = *row.kerpad;
    if (row.upsampfac)
        opts.upsampfac = *row.upsampfac;
    if (row.max_sp_size)
        opts.spread_max_sp_size = *row.max_sp_size;
    return opts;
}
using FinufftBenchmarkVariant =
    std::variant<finufft_benchmark<1>, finufft_benchmark<2>,
                 finufft_benchmark<3>>;

FinufftBenchmarkVariant convertToFinufftBenchmark(std::minstd_rand& rng,
                                                  const CSVRow &row) {
    finufft_opts opts = parse_opts(row);
    opts.nthreads = 1;
    // Create an array for the dimensions
    const auto dim0 = row.N1;
    const auto dim1 = row.N2.value_or(0);
    const auto dim2 = row.N3.value_or(0);

    double eps = row.eps.value_or(1e-6);
    if (row.N2 && row.N3) {
        // Create the finufft_benchmark object for 3 dimensions
        return finufft_benchmark<3>(rng, {dim0, dim1, dim2}, row.M, eps,
                                    opts);
    } else if (row.N2) {
        // Create the finufft_benchmark object for 2 dimensions
        return finufft_benchmark<2>(rng, {dim0, dim1}, row.M, eps, opts);
    } else {
        // Create the finufft_benchmark object for 1 dimension
        return finufft_benchmark<1>(rng, {dim0}, row.M, eps, opts);
    }
}

#endif // FINUFFT_PARSE_CSV_H
