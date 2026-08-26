#ifndef FINUFFT_HPP
#define FINUFFT_HPP

// Modern C++ interface to the FINUFFT CPU library (requires C++20).
// This interface is EXPERIMENTAL: names and semantics may change in future
// releases. The C API in finufft.h is the stability contract.
// Wraps the public C guru API (finufft.h) in an RAII class template:
//   - finufft::plan<float> and finufft::plan<double> own the plan handle.
//   - Errors throw finufft::error, which carries the FINUFFT_ERR_* code.
//   - Point and data arrays pass as std::span. setpts is variadic: one span
//     per coordinate dimension, and for type 3 the next dim spans pass the
//     target frequencies, so the argument list itself fixes the layout.
// All C symbols stay in the global namespace; this header adds only the
// finufft namespace.

#if !defined(__cplusplus) || __cplusplus < 202002L
#error "finufft.hpp requires C++20 or later"
#endif
#include <version>
#if !defined(__cpp_lib_span) || __cpp_lib_span < 202002L
#error "finufft.hpp requires std::span (C++20)"
#endif

#include <finufft.h>

#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <version>

namespace finufft {

// Exception type thrown by the C++ interface. The code matches the
// FINUFFT_ERR_* constants from finufft_errors.h; see docs/errors.rst.
class error : public std::runtime_error {
public:
  error(int code, const std::string &context)
      : std::runtime_error(context + ": finufft error " + std::to_string(code)),
        code_(code) {}
  [[nodiscard]] int code() const noexcept { return code_; }

private:
  int code_;
};

// Returns the library default options, matching finufft_default_opts().
template<typename T> [[nodiscard]] finufft_opts default_opts() noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "finufft::default_opts<T>: T must be float or double");
  finufft_opts o;
  if constexpr (std::is_same_v<T, float>) {
    ::finufftf_default_opts(&o);
  } else {
    ::finufft_default_opts(&o);
  }
  return o;
}

namespace detail {

// Per-precision bindings to the C API. The primary template is undefined, so
// misuse of another T fails at compile time.
template<typename T> struct c_api;

template<> struct c_api<float> {
  using plan_s                          = ::finufftf_plan_s;
  static constexpr auto makeplan        = &::finufftf_makeplan;
  static constexpr auto setpts          = &::finufftf_setpts;
  static constexpr auto execute         = &::finufftf_execute;
  static constexpr auto execute_adjoint = &::finufftf_execute_adjoint;
  static constexpr auto destroy         = &::finufftf_destroy;
};

template<> struct c_api<double> {
  using plan_s                          = ::finufft_plan_s;
  static constexpr auto makeplan        = &::finufft_makeplan;
  static constexpr auto setpts          = &::finufft_setpts;
  static constexpr auto execute         = &::finufft_execute;
  static constexpr auto execute_adjoint = &::finufft_execute_adjoint;
  static constexpr auto destroy         = &::finufft_destroy;
};

template<typename T> struct plan_deleter {
  void operator()(typename c_api<T>::plan_s *p) const noexcept {
    if (p) c_api<T>::destroy(p);
  }
};

// Narrows a span size to the library's BIGINT (int64_t) with overflow check.
template<typename S> std::int64_t checked_count(const S &s, const char *what) {
  if (s.size() > static_cast<std::size_t>(INT64_MAX))
    throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                std::string(what) + ": array size exceeds int64 range");
  return static_cast<std::int64_t>(s.size());
}

} // namespace detail

// RAII NUFFT plan. Construct with the transform type (1, 2 or 3), the mode
// counts (one entry per dimension; the entry count selects the dimension),
// the sign of the exponential, the tolerance, and optionally the number of
// transforms and the options. Then call setpts() once, and execute() any
// number of times.
//
//   finufft::plan p(1, {64, 64}, +1, 1e-9);          // 2d type 1, double
//   p.setpts(x, y);                                   // x, y: same-length spans
//   p.execute(coeffs, modes);                         // sizes checked
template<typename T> class plan {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "finufft::plan<T>: T must be float or double");
  using api = detail::c_api<T>;

public:
  plan(int type, std::initializer_list<std::int64_t> n_modes, int iflag, T tol,
       int ntrans = 1, const finufft_opts &opts = default_opts<T>()) {
    init(type, static_cast<int>(n_modes.size()), n_modes.begin(), iflag, ntrans, tol,
         &opts);
  }

  // Same as above, for mode counts held in a container (vector, array, ...).
  plan(int type, std::span<const std::int64_t> n_modes, int iflag, T tol, int ntrans = 1,
       const finufft_opts &opts = default_opts<T>()) {
    init(type, static_cast<int>(n_modes.size()), n_modes.data(), iflag, ntrans, tol,
         &opts);
  }

  // Types 1 and 2: pass dim source-point spans (x; x,y; or x,y,z).
  // Type 3: pass dim source-point spans followed by dim target-frequency
  // spans (x,s; x,y,s,t; or x,y,z,s,t,u). nj and nk come from the span
  // sizes. Coord spans accept any contiguous range of T (vector, array, ...).
  template<typename... S>
    requires(std::is_convertible_v<S, std::span<const T>> && ...)
  void setpts(const S &...pts) {
    const std::array<std::span<const T>, sizeof...(S)> a{std::span<const T>(pts)...};
    const std::size_t want = (type_ == 3) ? 2 * std::size_t(dim_) : std::size_t(dim_);
    if (a.size() != want)
      throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                  "finufft::plan::setpts: type " + std::to_string(type_) + " in " +
                      std::to_string(dim_) + "d expects " + std::to_string(want) +
                      " array(s), got " + std::to_string(a.size()));
    const auto sz = [&a](std::size_t i) -> std::size_t {
      return i < a.size() ? a[i].size() : 0; // clamps help -Warray-bounds analysis
    };
    const auto at = [&a](std::size_t i) -> const T * {
      return i < a.size() ? a[i].data() : nullptr;
    };
    for (std::size_t i = 1; i < std::size_t(dim_); ++i)
      if (sz(i) != sz(0))
        throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                    "finufft::plan::setpts: source point arrays differ in size");
    const T *x = at(0);
    const T *y = dim_ > 1 ? at(1) : nullptr;
    const T *z = dim_ > 2 ? at(2) : nullptr;
    const T *s = nullptr, *t = nullptr, *u = nullptr;
    std::int64_t nk = 0;
    if (type_ == 3) {
      s = at(std::size_t(dim_));
      if (dim_ > 1) t = at(std::size_t(dim_) + 1);
      if (dim_ > 2) u = at(std::size_t(dim_) + 2);
      for (std::size_t i = 1; i < std::size_t(dim_); ++i)
        if (sz(std::size_t(dim_) + i) != sz(std::size_t(dim_)))
          throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                      "finufft::plan::setpts: target point arrays differ in size");
      nk = detail::checked_count(a[std::size_t(dim_)], "finufft::plan::setpts");
    }
    do_setpts(detail::checked_count(a[0], "finufft::plan::setpts"), x, y, z, nk, s, t, u);
  }

  // execute: weights is cj (one value per NU point per transform), result is
  // fk (one per mode per transform, or per target for type 3). Buffer sizes
  // are checked against the plan before the library call.
  void execute(std::span<std::complex<T>> weights, std::span<std::complex<T>> result) {
    check_io("finufft::plan::execute", weights, result);
    check(api::execute(handle_.get(), weights.data(), result.data()),
          "finufft::plan::execute");
  }

  // execute with the exponential sign flipped (no replan).
  void execute_adjoint(std::span<std::complex<T>> weights,
                       std::span<std::complex<T>> result) {
    check_io("finufft::plan::execute_adjoint", weights, result);
    check(api::execute_adjoint(handle_.get(), weights.data(), result.data()),
          "finufft::plan::execute_adjoint");
  }

private:
  std::unique_ptr<typename api::plan_s, detail::plan_deleter<T>> handle_{};
  int type_ = 0, dim_ = 0, ntrans_ = 0;
  std::int64_t nmodes_prod_ = 0, nj_ = 0, nk_ = 0;
  bool has_pts_ = false;

  void init(int type, int dim, const std::int64_t *n_modes, int iflag, int ntrans, T tol,
            const finufft_opts *opts) {
    typename api::plan_s *raw = nullptr;
    const int ier = api::makeplan(type, dim, n_modes, iflag, ntrans, tol, &raw, opts);
    if (ier) throw error(ier, "finufft::plan::plan");
    handle_.reset(raw);
    type_        = type;
    dim_         = dim;
    ntrans_      = ntrans;
    nmodes_prod_ = 1;
    for (int i = 0; i < dim; ++i) nmodes_prod_ *= n_modes[i];
  }

  static void check(int ier, const char *what) {
    if (ier) throw error(ier, what);
  }

  void do_setpts(std::int64_t nj, const T *x, const T *y, const T *z, std::int64_t nk,
                 const T *s, const T *t, const T *u) {
    ensure_valid("finufft::plan::setpts");
    check(api::setpts(handle_.get(), nj, x, y, z, nk, s, t, u), "finufft::plan::setpts");
    nj_      = nj;
    nk_      = nk;
    has_pts_ = true;
  }

  void ensure_valid(const char *what) const {
    if (!handle_) throw error(FINUFFT_ERR_PLAN_NOTVALID, what);
  }

  void check_io(const char *what, std::span<std::complex<T>> weights,
                std::span<std::complex<T>> result) const {
    ensure_valid(what);
    if (!has_pts_)
      throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                  std::string(what) + ": setpts has not been called on this plan");
    const std::int64_t want_w = nj_ * ntrans_;
    const std::int64_t want_r = (type_ == 3 ? nk_ : nmodes_prod_) * ntrans_;
    if (detail::checked_count(weights, what) != want_w)
      throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                  std::string(what) + ": weights size does not match nj*ntrans");
    if (detail::checked_count(result, what) != want_r)
      throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                  std::string(what) + ": result size does not match the plan's "
                                      "mode/target count times ntrans");
  }
};

// Deduction guides let the tolerance literal pick the precision:
//   finufft::plan p(1, {64}, +1, 1e-6);   // plan<double>
//   finufft::plan p(1, {64}, +1, 1e-6f);  // plan<float>
template<typename T> plan(int, std::initializer_list<std::int64_t>, int, T) -> plan<T>;
template<typename T> plan(int, std::span<const std::int64_t>, int, T) -> plan<T>;

} // namespace finufft

#endif // FINUFFT_HPP
