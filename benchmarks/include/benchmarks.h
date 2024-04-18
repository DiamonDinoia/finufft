#ifndef FINUFFT_BENCHMARKS_H
#define FINUFFT_BENCHMARKS_H

#include "finufft.h"
#include <array>
#include <cmath>
#include <complex>
#include <random>

namespace {
template <std::size_t Dim>
concept OneDimensional = Dim == 1;

template <std::size_t Dim>
concept TwoDimensional = Dim == 2;

template <std::size_t Dim>
concept ThreeDimensional = Dim == 3;

struct EMPTY {};
} // namespace
/**
 * @brief Constructor for the finufft_benchmark class.
 *
 * @param seed Seed for the random number generator.
 * @param dims Dimensions for the transform.
 * @param M Parameter M.
 * @param eps Error tolerance.
 * @param opts Options for the transform.
 */

template <std::size_t Dim>
class finufft_benchmark {
  public:
    finufft_benchmark(std::minstd_rand& rng, std::array<std::int64_t, Dim> dims,
                      std::int64_t M, double eps, finufft_opts opts)
        : gen(rng), dims(std::move(dims)), M(M),
          N(std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>())),
          eps(eps), opts(opts) {
        generate_points();
        generate_c();
        F.resize(N, 0);
    }
    /**
     * @brief Performs the finufft1d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */

    std::vector<std::complex<double>> &benchmark_finufft1d1()
        requires OneDimensional<Dim>
    {
        const auto ier = finufft1d1(M, x.data(), c.data(), +1, eps, dims[0],
                                    F.data(), &opts);
        if (ier != 0) {
            throw std::runtime_error("finufft1d1 failed");
        }
        return F;
    }

    /**
     * @brief Performs the finufft1d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<double>> &benchmark_finufft1d2()
        requires OneDimensional<Dim>
    {
        const auto ier = finufft1d2(M, x.data(), c.data(), +1, eps, dims[0],
                                    F.data(), &opts);
        if (ier != 0) {
            throw std::runtime_error("finufft1d2 failed");
        }
        return F;
    }

    /**
     * @brief Performs the finufft2d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<double>> &benchmark_finufft2d1()
        requires TwoDimensional<Dim>
    {
        const auto ier = finufft2d1(M, x.data(), y.data(), c.data(), +1, eps,
                                    dims[0], dims[1], F.data(), &opts);
        if (ier != 0) {
            throw std::runtime_error("finufft1d2 failed");
        }
        return F;
    }

    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<double>> &benchmark_finufft2d2()
        requires TwoDimensional<Dim>
    {
        const auto ier = finufft2d2(M, x.data(), y.data(), c.data(), +1, eps,
                                    dims[0], dims[1], F.data(), &opts);
        if (ier != 0) {
            throw std::runtime_error("finufft1d2 failed");
        }
        return F;
    }
    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<double>> &benchmark_finufft3d1()
        requires ThreeDimensional<Dim>
    {
        const auto ier =
            finufft3d1(M, x.data(), y.data(), z.data(), c.data(), +1, eps,
                       dims[0], dims[1], dims[2], F.data(), &opts);
        if (ier != 0) {
            throw std::runtime_error("finufft1d2 failed");
        }
        return F;
    }

    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<double>> &benchmark_finufft3d2()
        requires ThreeDimensional<Dim>
    {
        const auto ier =
            finufft3d2(M, x.data(), y.data(), z.data(), c.data(), +1, eps,
                       dims[0], dims[1], dims[2], F.data(), &opts);
        if (ier != 0) {
            throw std::runtime_error("finufft1d2 failed");
        }
        return F;
    }

  private:
    static_assert(Dim > 0 && Dim <= 3);

    std::uniform_real_distribution<double> pi_dist{-M_PI, M_PI};
    std::uniform_real_distribution<double> one_one_dist{-1, 1};
    const std::complex<double> I{0, 1};
    std::minstd_rand& gen;
    std::int64_t M;
    std::int64_t N;
    std::array<std::int64_t, Dim> dims;
    double eps;
    finufft_opts opts;

    // this is always present
    std::vector<double> x{};
    // only needed for 2D transforms
    [[no_unique_address]]
    typename std::conditional<(Dim > 1), std::vector<double>, EMPTY>::type y{};
    // only needed for 3D transforms
    [[no_unique_address]]
    typename std::conditional<(Dim > 2), std::vector<double>, EMPTY>::type z{};
    std::vector<std::complex<double>> c{};
    std::vector<std::complex<double>> F{};

    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    void generate_points() {
        x.resize(M);
        for (std::uint64_t i = 0; i < M; ++i) {
            x[i] = pi_dist(gen);
        }
        if constexpr (Dim > 1) {
            y.resize(M);
            for (std::uint64_t i = 0; i < M; ++i) {
                y[i] = pi_dist(gen);
            }
        }
        if constexpr (Dim > 2) {
            z.resize(M);
            for (std::uint64_t i = 0; i < M; ++i) {
                z[i] = pi_dist(gen);
            }
        }
    }

    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    void generate_c() {
        c.resize(M);
        for (std::uint64_t i = 0; i < M; ++i) {
            c[i] = one_one_dist(gen) + I * one_one_dist(gen);
        }
    }
};

#endif // FINUFFT_BENCHMARKS_H
