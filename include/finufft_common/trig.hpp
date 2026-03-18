#pragma once

#include <array>
#include <cmath>
#include <complex>
#include <tuple>
#include <type_traits>

#include <finufft_common/constants.h>
#include <finufft_common/defines.h>

namespace finufft::common::math {

// Default precision: 15 digits for double (6 terms), 7 for float (4 terms).
template<typename T>
inline constexpr int default_approx_digits = std::is_same_v<T, float> ? 7 : 15;

namespace detail {

struct consts {
  static constexpr double pi_over_2     = PI / 2.0;
  static constexpr double neg_pi_over_2 = -pi_over_2;
  static constexpr double inv_pi_over_2 = 1.0 / pi_over_2;
};

// Polynomial coefficients for sin(t) ~ t + sp*t^3 and cos(t) ~ 1 + cp*t^2
// on [-pi/4, pi/4], evaluated via Horner scheme on t^2.
// Ordered from highest to lowest degree term.
inline constexpr std::array<double, 6> sin_coeffs = {
    0x1.5e585f68f956ep-33, -0x1.ae5f687b275b3p-26, 0x1.71de33799ebc6p-19,
    -0x1.a01a019367fdp-13, 0x1.1111111104f1dp-7,   -0x1.555555555541bp-3,
};

inline constexpr std::array<double, 6> cos_coeffs = {
    0x1.1b88ad1c62723p-29,  -0x1.27df3a1e26a95p-22, 0x1.a019f7fcecefp-16,
    -0x1.6c16c163eaf27p-10, 0x1.555555554ef27p-5,   -0x1.fffffffffff91p-2,
};

template<int TolDigits, typename T> struct approx_coefficients {
  static constexpr std::size_t nterms() {
    if (TolDigits <= 4) return 2;
    if (TolDigits <= 6) return 3;
    if (TolDigits <= 8) return 4;
    if (TolDigits <= 12) return 5;
    return 6;
  }

  template<std::size_t M, std::size_t... I>
  static constexpr auto cast_tail_impl(const std::array<double, M> &src,
                                       std::index_sequence<I...>) {
    return std::array<T, sizeof...(I)>{static_cast<T>(src[M - sizeof...(I) + I])...};
  }

  template<std::size_t M>
  static constexpr auto cast_tail(const std::array<double, M> &src) {
    return cast_tail_impl(src, std::make_index_sequence<nterms()>{});
  }

  static constexpr auto sin_poly = cast_tail(sin_coeffs);
  static constexpr auto cos_poly = cast_tail(cos_coeffs);
};

template<typename C, std::size_t N, typename T, std::size_t... I>
FINUFFT_ALWAYS_INLINE T horner_impl(const std::array<C, N> &coeffs, T x,
                                    std::index_sequence<I...>) {
  T acc = T(coeffs[0]);
  ((acc = std::fma(acc, x, T(coeffs[I + 1]))), ...);
  return acc;
}

template<typename C, std::size_t N, typename T>
FINUFFT_ALWAYS_INLINE T eval_horner(const std::array<C, N> &coeffs, T x) {
  static_assert(N > 0);
  return horner_impl(coeffs, x, std::make_index_sequence<N - 1>{});
}

template<int TolDigits, typename T>
FINUFFT_ALWAYS_INLINE std::pair<T, T> evaluate_reduced(T t) {
  constexpr auto &si = approx_coefficients<TolDigits, T>::sin_poly;
  constexpr auto &ct = approx_coefficients<TolDigits, T>::cos_poly;
  const T t2         = t * t;
  const T t3         = t2 * t;
  return {std::fma(eval_horner(si, t2), t3, t), std::fma(eval_horner(ct, t2), t2, T(1))};
}

template<int TolDigits, typename T>
FINUFFT_ALWAYS_INLINE std::tuple<T, T> sincos_impl(T angle) {
  const auto qi =
      static_cast<long long>(std::nearbyint(angle * T(consts::inv_pi_over_2)));
  T x1;
  if constexpr (std::is_same_v<T, float>) {
    constexpr T pio2_1 = 1.5703125f;
    constexpr T pio2_2 = 4.837512969970703125e-4f;
    constexpr T pio2_3 = 7.549789954891882e-8f;
    x1                 = std::fma(T(qi), -pio2_1, angle);
    x1                 = std::fma(T(qi), -pio2_2, x1);
    x1                 = std::fma(T(qi), -pio2_3, x1);
  } else {
    x1 = std::fma(T(qi), T(consts::neg_pi_over_2), angle);
  }
  const auto [s1, c1] = evaluate_reduced<TolDigits>(x1);
  const T s2          = (qi & 1) == 0 ? s1 : c1;
  const T c2          = (qi & 1) == 0 ? c1 : -s1;
  return {(qi & 2) == 0 ? s2 : -s2, (qi & 2) == 0 ? c2 : -c2};
}

} // namespace detail

// sincos: returns {sin(angle), cos(angle)} via polynomial approximation.
template<typename T, int TolDigits = default_approx_digits<T>>
FINUFFT_FLATTEN FINUFFT_ALWAYS_INLINE std::tuple<T, T> sincos(T angle) {
  return detail::sincos_impl<TolDigits>(angle);
}

// cis: returns magnitude * exp(i * angle).
template<typename T, int TolDigits = default_approx_digits<T>>
FINUFFT_FLATTEN FINUFFT_ALWAYS_INLINE std::complex<T> cis(T angle, T magnitude = T(1)) {
  if constexpr (std::is_same_v<T, float> && TolDigits == default_approx_digits<float>) {
    return std::polar(magnitude, angle);
  } else {
    const auto [s, c] = detail::sincos_impl<TolDigits>(angle);
    return {magnitude * c, magnitude * s};
  }
}

// polar: returns magnitude * exp(i * angle), argument order matches std::polar.
template<typename T, int TolDigits = default_approx_digits<T>>
FINUFFT_FLATTEN FINUFFT_ALWAYS_INLINE std::complex<T> polar(T magnitude, T angle) {
  return cis<T, TolDigits>(angle, magnitude);
}

} // namespace finufft::common::math
