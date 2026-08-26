#ifndef CUFINUFFT_HPP
#define CUFINUFFT_HPP

// Modern C++ interface to the cuFINUFFT (GPU) library (requires C++20).
// This interface is EXPERIMENTAL: names and semantics may change in future
// releases. The C API in cufinufft.h is the stability contract.
// Mirrors finufft.hpp so that CPU and GPU call sites share one shape:
//   - cufinufft::plan<float> and cufinufft::plan<double> own the plan handle.
//   - Errors throw cufinufft::error, which carries the FINUFFT_ERR_* code.
//   - Point and data arrays pass as std::span. setpts is variadic: one span
//     per coordinate dimension, and for type 3 the next dim spans pass the
//     target frequencies.
// All spans and device pointers in this header refer to device memory;
// construct the spans from the device pointer and count (std::span(p, n)).
// Differences from the CPU header: there is no execute_adjoint (the GPU C API
// does not expose it) and the type 3 target count is capped to int.
// All C symbols stay in the global namespace; this header adds only the
// cufinufft namespace.

#if !defined(__cplusplus) || __cplusplus < 202002L
#error "cufinufft.hpp requires C++20 or later"
#endif
#include <version>
#if !defined(__cpp_lib_span) || __cpp_lib_span < 202002L
#error "cufinufft.hpp requires std::span (C++20)"
#endif

#include <cufinufft.h>

#include <array>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <version>

namespace cufinufft {

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

// Returns the library default options, matching cufinufft_default_opts().
// cufinufft_opts does not depend on T, but templating this keeps the shape of
// finufft::default_opts<T>() so that code generic over CPU/GPU still compiles.
template<typename T> [[nodiscard]] cufinufft_opts default_opts() noexcept {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "cufinufft::default_opts<T>: T must be float or double");
  cufinufft_opts o;
  ::cufinufft_default_opts(&o);
  return o;
}

namespace detail {

// Per-precision bindings to the C API. The primary template is undefined, so
// misuse of another T fails at compile time.
template<typename T> struct c_api;

template<> struct c_api<float> {
  using plan_s                   = ::cufinufft_fplan_s;
  using cux                      = cuFloatComplex;
  static constexpr auto makeplan = &::cufinufftf_makeplan;
  static constexpr auto setpts   = &::cufinufftf_setpts;
  static constexpr auto execute  = &::cufinufftf_execute;
  static constexpr auto destroy  = &::cufinufftf_destroy;
};

template<> struct c_api<double> {
  using plan_s                   = ::cufinufft_plan_s;
  using cux                      = cuDoubleComplex;
  static constexpr auto makeplan = &::cufinufft_makeplan;
  static constexpr auto setpts   = &::cufinufft_setpts;
  static constexpr auto execute  = &::cufinufft_execute;
  static constexpr auto destroy  = &::cufinufft_destroy;
};

template<typename T> struct plan_deleter {
  void operator()(typename c_api<T>::plan_s *p) const noexcept {
    if (p) c_api<T>::destroy(p);
  }
};

// Narrows a span size to int64_t with overflow check.
template<typename S> std::int64_t checked_count(const S &s, const char *what) {
  if (s.size() > static_cast<std::size_t>(INT64_MAX))
    throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                std::string(what) + ": array size exceeds int64 range");
  return static_cast<std::int64_t>(s.size());
}

// The CUDA complex types share std::complex layout (two consecutive reals),
// which is what every CUDA host compiler relies on.
static_assert(sizeof(std::complex<float>) == sizeof(cuFloatComplex) &&
                  sizeof(std::complex<double>) == sizeof(cuDoubleComplex),
              "cufinufft.hpp: std::complex and CUDA complex layouts differ");

} // namespace detail

// RAII NUFFT plan on the GPU. Construct with the transform type (1, 2 or 3),
// the mode counts (one entry per dimension; the entry count selects the
// dimension), the sign of the exponential, the tolerance, and optionally the
// number of transforms and the options. Then call setpts() once, and
// execute() any number of times.
//
//   cufinufft::plan p(1, {64, 64}, +1, 1e-9);        // 2d type 1, double
//   p.setpts(d_x, d_y);                               // device spans
//   p.execute(d_coeffs, d_modes);                     // sizes checked
template<typename T> class plan {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                "cufinufft::plan<T>: T must be float or double");
  using api = detail::c_api<T>;

public:
  plan(int type, std::initializer_list<std::int64_t> n_modes, int iflag, T tol,
       int ntrans = 1, const cufinufft_opts &opts = default_opts<T>()) {
    init(type, static_cast<int>(n_modes.size()), n_modes.begin(), iflag, ntrans, tol,
         &opts);
  }

  // Same as above, for mode counts held in a container (vector, array, ...).
  plan(int type, std::span<const std::int64_t> n_modes, int iflag, T tol, int ntrans = 1,
       const cufinufft_opts &opts = default_opts<T>()) {
    init(type, static_cast<int>(n_modes.size()), n_modes.data(), iflag, ntrans, tol,
         &opts);
  }

  // Types 1 and 2: pass dim source-point spans (x; x,y; or x,y,z).
  // Type 3: pass dim source-point spans followed by dim target-frequency
  // spans (x,s; x,y,s,t; or x,y,z,s,t,u). nj and nk come from the span
  // sizes. All spans point into device memory.
  template<typename... S>
    requires(std::is_convertible_v<S, std::span<const T>> && ...)
  void setpts(const S &...pts) {
    const std::array<std::span<const T>, sizeof...(S)> a{std::span<const T>(pts)...};
    const std::size_t want = (type_ == 3) ? 2 * std::size_t(dim_) : std::size_t(dim_);
    if (a.size() != want)
      throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                  "cufinufft::plan::setpts: type " + std::to_string(type_) + " in " +
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
                    "cufinufft::plan::setpts: source point arrays differ in size");
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
                      "cufinufft::plan::setpts: target point arrays differ in size");
      nk = detail::checked_count(a[std::size_t(dim_)], "cufinufft::plan::setpts");
    }
    do_setpts(detail::checked_count(a[0], "cufinufft::plan::setpts"), x, y, z, nk, s, t,
              u);
  }

  // execute: weights is cj (one value per NU point per transform), result is
  // fk (one per mode per transform, or per target for type 3), both in
  // device memory. Buffer sizes are checked against the plan.
  void execute(std::span<std::complex<T>> weights, std::span<std::complex<T>> result) {
    check_io("cufinufft::plan::execute", weights, result);
    auto *w = reinterpret_cast<typename api::cux *>(weights.data());
    auto *r = reinterpret_cast<typename api::cux *>(result.data());
    check(api::execute(handle_.get(), w, r), "cufinufft::plan::execute");
  }

private:
  std::unique_ptr<typename api::plan_s, detail::plan_deleter<T>> handle_{};
  int type_ = 0, dim_ = 0, ntrans_ = 0;
  std::int64_t nmodes_prod_ = 0, nj_ = 0, nk_ = 0;
  bool has_pts_ = false;

  void init(int type, int dim, const std::int64_t *n_modes, int iflag, int ntrans, T tol,
            const cufinufft_opts *opts) {
    typename api::plan_s *raw = nullptr;
    const int ier = api::makeplan(type, dim, n_modes, iflag, ntrans, tol, &raw, opts);
    if (ier) throw error(ier, "cufinufft::plan::plan");
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
    ensure_valid("cufinufft::plan::setpts");
    if (nk > std::numeric_limits<int>::max())
      throw error(FINUFFT_ERR_INVALID_ARGUMENT,
                  "cufinufft::plan::setpts: nk exceeds the int range cuFINUFFT "
                  "supports");
    check(api::setpts(handle_.get(), nj, x, y, z, static_cast<int>(nk), s, t, u),
          "cufinufft::plan::setpts");
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
//   cufinufft::plan p(1, {64}, +1, 1e-6);   // plan<double>
//   cufinufft::plan p(1, {64}, +1, 1e-6f);  // plan<float>
template<typename T> plan(int, std::initializer_list<std::int64_t>, int, T) -> plan<T>;
template<typename T> plan(int, std::span<const std::int64_t>, int, T) -> plan<T>;

} // namespace cufinufft

#endif // CUFINUFFT_HPP
