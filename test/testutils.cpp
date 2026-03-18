/* unit tests for utils module.

   Usage: ./testutils{f}

   Pass: exit code 0. (Stdout should indicate passed)
   Fail: exit code>0. (Stdout may indicate what failed)

   June 2023: switched to pass-fail tests within the executable (more clear,
   and platform-indep, than having to compare the text output)

   Suggested compile. double-prec:
   g++ -std=c++17 -fopenmp testutils.cpp -I../include ../src/utils.o
       ../src/utils.o -o testutils -lgomp
   single-prec:
   g++ -std=c++17 -fopenmp testutils.cpp
       -I../include ../src/utils.o -o testutilsf -lgomp -DSINGLE
*/

// This switches FLT macro from double to float if SINGLE is defined, etc...

#include "finufft/memory.hpp"
#include "finufft/utils.hpp"
#include "finufft_common/trig.hpp"
#include "utils/norms.hpp"
#include <cstdint>
#include <finufft/test_defs.hpp>
#include <type_traits>

namespace finufft::common {
double cyl_bessel_i_custom(double nu, double x) noexcept;
} // namespace finufft::common

using namespace finufft::utils;

int main(int argc, char *argv[]) {
#ifdef SINGLE
  printf("testutilsf started...\n");
#else
  printf("testutils started...\n");
#endif

  // test next235even...
  // Barnett 2/9/17, made smaller range 3/28/17. pass-fail 6/16/23
  // The true outputs from {0,1,..,99}:
  const BIGINT next235even_true[100] = {
      2,  2,  2,  4,  4,  6,  6,  8,  8,  10, 10, 12, 12, 16, 16, 16, 16, 18,  18,  20,
      20, 24, 24, 24, 24, 30, 30, 30, 30, 30, 30, 32, 32, 36, 36, 36, 36, 40,  40,  40,
      40, 48, 48, 48, 48, 48, 48, 48, 48, 50, 50, 54, 54, 54, 54, 60, 60, 60,  60,  60,
      60, 64, 64, 64, 64, 72, 72, 72, 72, 72, 72, 72, 72, 80, 80, 80, 80, 80,  80,  80,
      80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 96, 96, 96, 96, 96, 96, 100, 100, 100};
  for (BIGINT n = 0; n < 100; ++n) {
    BIGINT o = next235even(n);
    BIGINT t = next235even_true[n];
    if (o != t) {
      printf("next235even(%lld) =\t%lld, error should be %lld!\n", (long long)n,
             (long long)o, (long long)t);
      return 1;
    }
  }
  // various old devel expts and comments for next235even...
  // printf("starting huge next235even...\n");   // 1e11 takes 1 sec
  // BIGINT n=(BIGINT)120573851963;
  // printf("next235even(%ld) =\t%ld\n",n,next235even(n));
  // double* a; printf("%g\n",a[0]);  // do deliberate segfault for bash debug!

  // test Gauss-Legendre quadrature...
  const int n = 16;
  std::vector<double> x(n), w(n);
  finufft::common::gaussquad(n, x.data(), w.data());
  auto f = [](double x) {
    return sin(4 * x + 1.0) + 0.3;
  }; // a test func f(x)
  auto fp = [](double x) {
    return 4 * cos(4 * x + 1.0);
  }; // its deriv f'(x)
  double I = 0;
  for (int i = 0; i < n; ++i) I += w[i] * fp(x[i]);
  double Iex = f(1.0) - f(-1.0);
  double err = std::abs(I - Iex);
  if (err > 1e-14) { // for the above func, err should be 4e-14
    printf("fail: gaussquad error %g\n", err);
    return 1;
  }

  // test vector norms and norm difference routines... now pass-fail 6/16/23
  BIGINT M = 1e4;
  std::vector<CPX> a(M), b(M);
  for (BIGINT j = 0; j < M; ++j) {
    a[j] = CPX(1.0, 0.0);
    b[j] = a[j];
  }
  constexpr FLT EPSILON = std::numeric_limits<FLT>::epsilon();
  FLT relerr            = 2.0 * EPSILON; // 1 ULP, fine since 1.0 rep exactly
  if (std::abs(infnorm(M, &a[0]) - 1.0) > relerr) return 1;
  if (std::abs(twonorm(M, &a[0]) - std::sqrt((FLT)M)) > relerr * std::sqrt((FLT)M)) return 1;
  b[0] = CPX(0.0, 0.0); // perturb b from a
  if (std::abs(errtwonorm(M, &a[0], &b[0]) - 1.0) > relerr) return 1;
  if (std::abs(std::sqrt((FLT)M) * relerrtwonorm(M, &a[0], &b[0]) - 1.0) > relerr) return 1;

  // test custom cis/polar approximation used by type-3 setpts and makeplan
  static_assert(finufft::common::math::default_approx_digits<float> == 7);
  static_assert(finufft::common::math::default_approx_digits<double> == 15);
  {
    auto check_trig = [](auto max_err) {
      using T               = decltype(max_err);
      constexpr T magnitude = T(1.25);
      for (int i = -2000; i <= 2000; i += 17) {
        const T angle    = T(0.125) * T(i) + T(1e-4) * T(i) * T(i);
        const auto got_c = finufft::common::math::cis(angle, magnitude);
        const auto got_p = finufft::common::math::polar(magnitude, angle);
        const auto ref   = std::polar(magnitude, angle);
        const auto err   = (std::max)(std::abs(got_c - ref), std::abs(got_p - ref));
        if (err > max_err) {
          printf("fail: trig<%s> angle=%g err=%g bound=%g\n",
                 std::is_same_v<T, float> ? "float" : "double", (double)angle,
                 (double)err, (double)max_err);
          return false;
        }
      }
      return true;
    };
    if (!check_trig(1e-6f)) return 1;
    if (!check_trig(2e-12)) return 1;
  }

  // type-3 cache keys must follow source/target values, not pointer identity
  {
    const auto run_type3 = [](const std::vector<FLT> &x, const std::vector<FLT> &s,
                              const std::vector<CPX> &c, std::vector<CPX> &out) {
      finufft_opts opts;
      FINUFFT_DEFAULT_OPTS(&opts);
      opts.nthreads = 1;

      FINUFFT_PLAN plan = nullptr;
      BIGINT nmodes[1]  = {0};
      int ier           = FINUFFT_MAKEPLAN(3, 1, nmodes, +1, 1, (FLT)1e-12, &plan, &opts);
      if (ier != 0) {
        printf("fail: type3 cache test makeplan ier=%d\n", ier);
        return false;
      }

      ier = FINUFFT_SETPTS(plan, (BIGINT)x.size(), x.data(), nullptr, nullptr,
                           (BIGINT)s.size(), s.data(), nullptr, nullptr);
      if (ier != 0) {
        printf("fail: type3 cache test setpts ier=%d\n", ier);
        FINUFFT_DESTROY(plan);
        return false;
      }

      out.resize(s.size());
      ier = FINUFFT_EXECUTE(plan, const_cast<CPX *>(c.data()), out.data());
      if (ier != 0) {
        printf("fail: type3 cache test execute ier=%d\n", ier);
        FINUFFT_DESTROY(plan);
        return false;
      }

      FINUFFT_DESTROY(plan);
      return true;
    };

    const auto check_reused_plan = [&](std::vector<FLT> x, std::vector<FLT> s,
                                       BIGINT mutate_index, FLT delta, bool mutate_targets,
                                       const char *label) {
      const std::vector<CPX> c = {
          CPX(0.4, -0.2), CPX(-0.3, 0.1), CPX(0.2, 0.5), CPX(-0.1, -0.4), CPX(0.3, 0.2)};

      finufft_opts opts;
      FINUFFT_DEFAULT_OPTS(&opts);
      opts.nthreads = 1;

      FINUFFT_PLAN plan = nullptr;
      BIGINT nmodes[1]  = {0};
      int ier           = FINUFFT_MAKEPLAN(3, 1, nmodes, +1, 1, (FLT)1e-12, &plan, &opts);
      if (ier != 0) {
        printf("fail: %s makeplan ier=%d\n", label, ier);
        return false;
      }

      ier = FINUFFT_SETPTS(plan, (BIGINT)x.size(), x.data(), nullptr, nullptr,
                           (BIGINT)s.size(), s.data(), nullptr, nullptr);
      if (ier != 0) {
        printf("fail: %s first setpts ier=%d\n", label, ier);
        FINUFFT_DESTROY(plan);
        return false;
      }

      if (mutate_targets)
        s[mutate_index] += delta;
      else
        x[mutate_index] += delta;

      ier = FINUFFT_SETPTS(plan, (BIGINT)x.size(), x.data(), nullptr, nullptr,
                           (BIGINT)s.size(), s.data(), nullptr, nullptr);
      if (ier != 0) {
        printf("fail: %s second setpts ier=%d\n", label, ier);
        FINUFFT_DESTROY(plan);
        return false;
      }

      std::vector<CPX> reused_out(s.size()), fresh_out;
      ier = FINUFFT_EXECUTE(plan, const_cast<CPX *>(c.data()), reused_out.data());
      if (ier != 0) {
        printf("fail: %s reused execute ier=%d\n", label, ier);
        FINUFFT_DESTROY(plan);
        return false;
      }
      FINUFFT_DESTROY(plan);

      if (!run_type3(x, s, c, fresh_out)) return false;

      const FLT rel_err = relerrtwonorm((BIGINT)fresh_out.size(), fresh_out.data(),
                                        reused_out.data());
      if (rel_err > (FLT)1e-12) {
        printf("fail: %s stale cache rel_err=%g\n", label, (double)rel_err);
        return false;
      }
      return true;
    };

    std::vector<FLT> x = {-1.75, -0.5, 0.1, 0.6, 1.8};
    std::vector<FLT> s = {-2.0, -0.75, 0.25, 1.1, 2.3};
    if (!check_reused_plan(x, s, 2, (FLT)0.55, true, "type3 target cache")) return 1;
    if (!check_reused_plan(x, s, 2, (FLT)-0.45, false, "type3 source cache")) return 1;
  }

  // test reclaimable workspace allocator...
  using RM                    = finufft::ReclaimableMemory;
  constexpr size_t align_mask = RM::ALIGNMENT - 1;

  RM buf;
  buf.mark_reclaimable(); // no-op before allocation
  if (!buf.allocate(0) || buf.size() != 0) return 1;

  // --- small-buffer path (heap via aligned_alloc) ---
  constexpr size_t small_nbytes = 8192;
  static_assert(small_nbytes < RM::MIN_MMAP_SIZE,
                "test needs a size below mmap threshold");
  if (!buf.allocate(small_nbytes) || buf.data() == nullptr || buf.size() != small_nbytes)
    return 1;
  if ((reinterpret_cast<std::uintptr_t>(buf.data()) & align_mask) != 0u) return 1;

  void *ptr = buf.data();
  if (!buf.allocate(small_nbytes) || buf.data() != ptr) return 1; // same-size reuse
  buf.mark_reclaimable(); // no-op for small buffers, should be safe

  RM moved = std::move(buf);
  if (buf.data() != nullptr || buf.size() != 0) return 1;
  if (moved.data() != ptr || moved.size() != small_nbytes) return 1;
  moved.mark_reclaimable();

  // --- large-buffer path (mmap / VirtualAlloc) ---
  constexpr size_t large_nbytes = RM::MIN_MMAP_SIZE * 2;
  RM large_buf;
  if (!large_buf.allocate(large_nbytes) || large_buf.data() == nullptr ||
      large_buf.size() != large_nbytes)
    return 1;
  // mmap returns page-aligned memory
  if ((reinterpret_cast<std::uintptr_t>(large_buf.data()) & 4095u) != 0u) return 1;

  void *lptr = large_buf.data();
  if (!large_buf.allocate(large_nbytes) || large_buf.data() != lptr)
    return 1; // same-size
  // Smaller request reuses existing allocation (>= check)
  if (!large_buf.allocate(large_nbytes / 2) || large_buf.data() != lptr) return 1;
  large_buf.mark_reclaimable(); // exercises madvise/MEM_RESET path

  RM large_moved = std::move(large_buf);
  if (large_buf.data() != nullptr || large_buf.size() != 0) return 1;
  if (large_moved.data() != lptr || large_moved.size() != large_nbytes) return 1;
  large_moved.mark_reclaimable();

#if defined(__cpp_lib_math_special_functions)
  // std::cyl_bessel_i present: compare std vs custom series
  for (double x = 0.0; x <= 42.0; x += 0.5) {
    double stdv    = std::cyl_bessel_i(0, x);
    double custom  = finufft::common::cyl_bessel_i_custom(0, x);
    double rel_err = std::abs(1.0 - stdv / custom);
    if (rel_err > std::numeric_limits<double>::epsilon() * 20) {
      printf("fail: Bessel mismatch at x=%g: std=%g custom=%g rel_err=%g\n", x, stdv,
             custom, rel_err);
      return 1;
    }
  }
#else
  printf("Bessel comparison test skipped. std bessel function not available.\n");
#endif

#ifdef SINGLE
  printf("testutilsf passed.\n");
#else
  printf("testutils passed.\n");
#endif
  return 0;
}
