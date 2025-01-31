#ifndef FINUFFT_HPP
#define FINUFFT_HPP

#define FINUFFT_USE_NAMESPACE
#include "finufft/finufft_core.h"

#include <finufft_opts.h>
#include <iostream>

namespace finufft {

template<typename T> class plan {
public:
  plan(int type, std::initializer_list<internal::BIGINT> n_modes, int iflag, T tol,
       int ntrans = 1, finufft_opts opts = get_default_opts())
      : ier{0}, opts{opts}, n_modes_array{init_nmdes_array(n_modes)},
        finufft_plan{
            type, n_modes.size(), n_modes_array.data(), iflag, ntrans, tol, &this->opts,
            ier} {
    if (ier != 0) {
      std::cerr << "Error in plan()" << std::endl;
    }
  }

private:
  int ier;
  finufft_opts opts;
  std::array<internal::BIGINT, 3> n_modes_array; // Initialize to zero
  internal::FINUFFT_PLAN_T<T> finufft_plan;

  static auto get_default_opts() noexcept {
    finufft_opts opts{};
    internal::finufft_default_opts_t(&opts);
    return opts;
  }

  static constexpr auto init_nmdes_array(
      std::initializer_list<internal::BIGINT> n_modes) noexcept {
    std::array<internal::BIGINT, 3> n_modes_array{};
    // static_assert(n_modes.size() > 0 && n_modes.size() <= 3,
    // "n_modes must have between 1 and 3 elements");
    std::copy(n_modes.begin(), n_modes.end(), n_modes_array.begin());
    return n_modes_array;
  }
};
} // namespace finufft

#undef FINUFFT_ALWAYS_INLINE
#undef FINUFFT_NEVER_INLINE
#undef FINUFFT_RESTRICT
#undef FINUFFT_UNREACHABLE
#undef FINUFFT_UNLIKELY
#undef FINUFFT_LIKELY
#undef FINUFFT_USE_NAMESPACE

#endif // FINUFFT_HPP
