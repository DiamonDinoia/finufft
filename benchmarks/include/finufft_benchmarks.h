#ifndef FINUFFT_FINUFFT_BENCHMARKS_H
#define FINUFFT_FINUFFT_BENCHMARKS_H

#include "finufft.h"
#include <array>
#include <cmath>
#include <complex>
#include <random>
#include <source_location>

template <std::size_t Dim>
concept OneDimensional = Dim == 1;

template <std::size_t Dim>
concept TwoDimensional = Dim == 2;

template <std::size_t Dim>
concept ThreeDimensional = Dim == 3;

template <typename T>
concept SingleType = std::is_same_v<T, float>;

template <typename T>
concept DoubleType = std::is_same_v<T, double>;

struct EMPTY {};

/**
 * @brief Constructor for the finufft_benchmark class.
 *
 * @param seed Seed for the random number generator.
 * @param dims Dimensions for the transform.
 * @param M Parameter number of not uniform points.
 * @param eps Error tolerance.
 * @param opts Options for the transform.
 */
template <std::size_t Dim, typename RealType>
class finufft_benchmark {
  public:
    static constexpr auto Dimensions = Dim;

    finufft_benchmark(std::minstd_rand &rng, std::array<std::int64_t, Dim> dims,
                      std::int64_t M, RealType eps, finufft_opts opts)
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
     * @return std::vector<std::complex<single>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft1d1()
        requires SingleType<RealType> && OneDimensional<Dim>
    {
       
        safe_call(__func__, finufftf1d1, M, x.data(), c.data(), +1, eps,
                  dims[0], F.data(), &opts);
        return F;
    }

    /**
     * @brief Performs the finufft1d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft1d1()
        requires DoubleType<RealType> && OneDimensional<Dim>
    {
       
        safe_call(__func__, finufft1d1, M, x.data(), c.data(), +1, eps, dims[0],
                  F.data(), &opts);
        return F;
    }
    /**
     * @brief Performs the finufft1d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft1d2()
        requires SingleType<RealType> && OneDimensional<Dim>

    {
       
        safe_call(__func__, finufftf1d2, M, x.data(), c.data(), +1, eps,
                  dims[0], F.data(), &opts);
        return F;
    }

    /**
     * @brief Performs the finufft1d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft1d2()
        requires DoubleType<RealType> && OneDimensional<Dim>
    {
       
        safe_call(__func__, finufft1d2, M, x.data(), c.data(), +1, eps, dims[0],
                  F.data(), &opts);
        return F;
    }
    /**
     * @brief Performs the finufft2d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft2d1()
        requires SingleType<RealType> && TwoDimensional<Dim>
    {
       
        safe_call(__func__, finufftf2d1, M, x.data(), y.data(), c.data(), +1,
                  eps, dims[0], dims[1], F.data(), &opts);
        return F;
    }

    /**
     * @brief Performs the finufft2d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft2d1()
        requires DoubleType<RealType> && TwoDimensional<Dim>
    {
       
        safe_call(__func__, finufft2d1, M, x.data(), y.data(), c.data(), +1,
                  eps, dims[0], dims[1], F.data(), &opts);
        return F;
    }
    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft2d2()
        requires SingleType<RealType> && TwoDimensional<Dim>
    {
       
        safe_call(__func__, finufftf2d2, M, x.data(), y.data(), c.data(), +1,
                  eps, dims[0], dims[1], F.data(), &opts);
        return F;
    }

    /**
     * @brief Performs the finufft2d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft2d2()
        requires DoubleType<RealType> && TwoDimensional<Dim>
    {
       
        safe_call(__func__, finufft2d2, M, x.data(), y.data(), c.data(), +1,
                  eps, dims[0], dims[1], F.data(), &opts);
        return F;
    }
    /**
     * @brief Performs the finufft3d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft3d1()
        requires SingleType<RealType> && ThreeDimensional<Dim>
    {
       
        safe_call(__func__, finufftf3d1, M, x.data(), y.data(), z.data(),
                  c.data(), +1, eps, dims[0], dims[1], dims[2], F.data(),
                  &opts);
        return F;
    }
    /**
     * @brief Performs the finufft3d1 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft3d1()
        requires DoubleType<RealType> && ThreeDimensional<Dim>
    {
       
        safe_call(__func__, finufft3d1, M, x.data(), y.data(), z.data(),
                  c.data(), +1, eps, dims[0], dims[1], dims[2], F.data(),
                  &opts);
        return F;
    }

    /**
     * @brief Performs the finufft3d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft3d2()
        requires SingleType<RealType> && ThreeDimensional<Dim>
    {
        safe_call(__func__, finufftf3d2, M, x.data(), y.data(), z.data(),
                  c.data(), +1, eps, dims[0], dims[1], dims[2], F.data(),
                  &opts);
        return F;
    }

    /**
     * @brief Performs the finufft3d2 transform and returns the result.
     *
     * @return std::vector<std::complex<double>>& Result of the transform.
     * @throws std::runtime_error if the transform fails.
     */
    std::vector<std::complex<RealType>> &benchmark_finufft3d2()
        requires DoubleType<RealType> && ThreeDimensional<Dim>
    {
        safe_call(__func__, finufft3d2, M, x.data(), y.data(), z.data(),
                  c.data(), +1, eps, dims[0], dims[1], dims[2], F.data(),
                  &opts);
        return F;
    }

  private:
    static_assert(Dim > 0 && Dim <= 3);
    std::uniform_real_distribution<RealType> pi_dist{-M_PI, M_PI};
    std::uniform_real_distribution<RealType> one_one_dist{-1, 1};
    const std::complex<RealType> I{0, 1};
    std::minstd_rand &gen;
    std::int64_t M;
    std::int64_t N;
    std::array<std::int64_t, Dim> dims;
    RealType eps;
    finufft_opts opts;

    // this is always present
    std::vector<RealType> x{};
    // only needed for 2D transforms
    [[no_unique_address]]
    typename std::conditional<(Dim > 1), std::vector<RealType>, EMPTY>::type
        y{};
    // only needed for 3D transforms
    [[no_unique_address]]
    typename std::conditional<(Dim > 2), std::vector<RealType>, EMPTY>::type
        z{};
    std::vector<std::complex<RealType>> c{};
    std::vector<std::complex<RealType>> F{};

    static_assert(std::is_same<RealType, double>::value ||
                      std::is_same<RealType, float>::value,
                  "RealType must be either double or float");
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

    /**
     * @brief Performs the finufft transform .
     * @throws std::runtime_error if the transform fails.
     */
    template <class Callable, class... Args>
    inline void safe_call(std::string_view name, Callable &&func,
                          Args... args) {
        const auto ier =
            std::forward<Callable>(func)(std::forward<Args>(args)...);
        if (ier != 0) {
            throw std::runtime_error(name.data());
        }
    }
};

#endif // FINUFFT_FINUFFT_BENCHMARKS_H
