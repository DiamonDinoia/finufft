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

#include <iostream>

#include <finufft/heuristics.hpp> // complexity-based upsampfac (sigma) picker
#include <finufft/test_defs.hpp>
#include <finufft/utils.hpp>

#include "utils/norms.hpp"

namespace finufft::common {
double cyl_bessel_i_custom(double nu, double x) noexcept;
} // namespace finufft::common

using namespace finufft::common;
using namespace finufft::heuristics;

int main(int argc, char *argv[]) {
#ifdef SINGLE
  printf("testutilsf started...\n");
#else
  printf("testutils started...\n");
#endif

  // test next235...
  // Barnett 2/9/17, made smaller range 3/28/17. pass-fail 6/16/23
  // The true outputs from {0,1,..,99}:
  const BIGINT next235even_true[100] = {
      2,  2,  2,  4,  4,  6,  6,  8,  8,  10, 10, 12, 12, 16, 16, 16, 16, 18,  18,  20,
      20, 24, 24, 24, 24, 30, 30, 30, 30, 30, 30, 32, 32, 36, 36, 36, 36, 40,  40,  40,
      40, 48, 48, 48, 48, 48, 48, 48, 48, 50, 50, 54, 54, 54, 54, 60, 60, 60,  60,  60,
      60, 64, 64, 64, 64, 72, 72, 72, 72, 72, 72, 72, 72, 80, 80, 80, 80, 80,  80,  80,
      80, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 96, 96, 96, 96, 96, 96, 100, 100, 100};
  for (BIGINT n = 0; n < 100; ++n) {
    BIGINT o = next235(n, 2);
    BIGINT t = next235even_true[n];
    if (o != t) {
      printf("next235(%lld, 2) =\t%lld, error should be %lld!\n", (long long)n,
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

#ifndef SINGLE
  // Complexity-based upsampfac (sigma) picker (finufft/heuristics.hpp). The block
  // exercises both precisions explicitly, so it runs once in the double build.
  {
    const double eps_d = std::numeric_limits<double>::epsilon();
    const double eps_f = std::numeric_limits<float>::epsilon();
    const int ns_d = MAX_NSPREAD<double>, ns_f = MAX_NSPREAD<float>;

    // (A) ns is non-increasing as sigma rises (the minimizer enumerates one candidate
    // per achievable width). Double holds over the whole auto range; float only above
    // FLOAT_CC_UPSAMPFAC_LIMIT, since below it the catastrophic-cancellation guard caps
    // ns low, so ns jumps up at the threshold.
    const double tols[] = {1e-3, 1e-6, 1e-10, 1e-13};
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type)
        for (double tol : tols) {
          int prev_d = 1 << 30, prev_f = 1 << 30;
          for (double s = MIN_AUTO_UPSAMPFAC; s <= MAX_AUTO_UPSAMPFAC + 1e-9; s += 0.05) {
            const int nd = kernel_width_at<double>(tol, dim, type, s);
            if (nd > prev_d) {
              printf("fail: ns(double) rose: dim=%d type=%d tol=%.0e sigma=%.2f\n", dim,
                     type, tol, s);
              return 1;
            }
            prev_d = nd;
            if (s < FLOAT_CC_UPSAMPFAC_LIMIT) continue; // skip float CC-capped region
            const int nf = kernel_width_at<float>(tol, dim, type, s);
            if (nf > prev_f) {
              printf("fail: ns(float) rose: dim=%d type=%d tol=%.0e sigma=%.2f\n", dim,
                     type, tol, s);
              return 1;
            }
            prev_f = nf;
          }
        }

    // (B) The narrow-kernel lever is real: at tight tol, ns strictly drops from
    // sigma 2.0 to 2.5 (double, dim 3), so higher sigma can pay off.
    if (!(kernel_width_at<double>(1e-13, 3, 1, 2.5) <
          kernel_width_at<double>(1e-13, 3, 1, 2.0))) {
      printf("fail: expected ns(2.5) < ns(2.0) at tol=1e-13 dim=3\n");
      return 1;
    }

    // (C) sigma=2.5 is feasible down to eps_mach for every dim/type, both precisions ->
    // analytic_upsampfac never returns an infeasible sigma for any tol the pipeline
    // forwards (it clamps tol up to eps_mach first).
    for (int dim = 1; dim <= 3; ++dim)
      for (int type = 1; type <= 3; ++type) {
        const double maxN = 256;
        if (!upsampfac_feasible(MAX_AUTO_UPSAMPFAC, eps_d, dim, type, eps_d, ns_d, false,
                                maxN) ||
            !upsampfac_feasible(MAX_AUTO_UPSAMPFAC, eps_f, dim, type, eps_f, ns_f, true,
                                maxN)) {
          printf("fail: sigma=2.5 infeasible at eps_mach: dim=%d type=%d\n", dim, type);
          return 1;
        }
      }

    // (D) analytic_upsampfac returns a sigma that is itself feasible, for a range of
    // achievable tols (its contract: the pick always survives the real plan).
    for (double tol : tols) {
      const double maxN = 1e4;
      const double s = analytic_upsampfac(tol, 2, 1, eps_d, ns_d, false, maxN);
      if (!(s >= MIN_AUTO_UPSAMPFAC - 1e-9 && s <= MAX_AUTO_UPSAMPFAC + 1e-9) ||
          !upsampfac_feasible(s, tol, 2, 1, eps_d, ns_d, false, maxN)) {
        printf("fail: analytic sigma %.3f not feasible/in range at tol=%.0e\n", s, tol);
        return 1;
      }
    }

    // (E) Density drives the pick: a spread-dominated transform (many points, small
    // grid) chooses a larger sigma than an FFT-dominated one (few points, large grid).
    {
      const int dim = 3, type = 1, nthr = 1;
      const double tol = 1e-13; // tight enough that ns drops across [2.0,2.5]
      const double dense_modes[3] = {64, 64, 64};
      const double sparse_modes[3] = {512, 512, 512};
      const double sigma_dense =
          best_type12<double>(tol, dim, type, nthr, dense_modes, /*npts=*/5e7).sigma;
      const double sigma_sparse =
          best_type12<double>(tol, dim, type, nthr, sparse_modes, /*npts=*/1e3).sigma;
      if (!(sigma_dense > sigma_sparse) || !(sigma_dense > MAX_CHECK_SIGMA - 1e-9)) {
        printf("fail: dense sigma (%.3f) should exceed sparse (%.3f) and 2.0\n",
               sigma_dense, sigma_sparse);
        return 1;
      }
    }
  }
#endif

#ifndef SINGLE
  // Blocked-spreading decomposition picker (heuristics::n_subproblems).
  // Asserts properties, not fitted values: the constants are node-dependent (the
  // optimum moves in opposite directions across microarchitectures), the
  // invariants below are not. Runs once, in the double build.
  {
    const double bin2[2] = {16, 4}, bin3[3] = {16, 4, 4};
    // 10 points on 128 threads covers npts < nthreads, where the elected count
    // exceeds the point count and only the caller's clip to M keeps it sane.
    const double npts_list[] = {10, 1e4, 1e5, 1e6, 1e7, 2.24e7, 1e8};
    const int nthr_list[] = {1, 6, 16, 32, 128};
    // 1e7 occupied bins is enough overhead to saturate the SP_MAX scratch-RAM cap,
    // where snapping nb down to a multiple of nthreads could otherwise breach it.
    const double occ_list[] = {1.0, 1e2, 1e4, 1e6, 1e7};
    // ns is the only geometry input (cells = prod(bin_size[d] + ns)), so sweep the
    // whole supported width range rather than one value: 3-4 is f32 (which is why
    // this block need not run in the SINGLE build - the picker takes no FLT), 7 is
    // f64 at default tol, 16 the maximum.
    const int ns_list[] = {3, 4, 7, 16};

    for (int dim = 2; dim <= 3; ++dim) {
      const double *bs = (dim == 2) ? bin2 : bin3;
      for (int ns : ns_list)
        for (double npts : npts_list)
          for (int nthr : nthr_list)
            for (double occ : occ_list) {
              const BIGINT nb =
                  n_subproblems(dim, ns, npts, std::min(occ, npts), bs, nthr);
              const double sp = npts / (double)nb;
              // nb below nthreads starves schedule(dynamic,1); a count just above a
              // multiple of nthreads runs a whole extra round for a few stragglers;
              // and sp above SP_MAX blows the per-thread scratch budget.
              if (nb < nthr || nb % nthr != 0 || sp > 1000000) {
                std::cout << "fail: nb=" << nb << " (" << sp
                          << " pts each) for nthr=" << nthr << " dim=" << dim
                          << " ns=" << ns << " npts=" << npts << " occ=" << occ << "\n";
                return 1;
              }
            }
    }

    // Small npts on many threads must not shatter into tiny subproblems: each one
    // pays for a whole padded subgrid, so nb must stay at nthr here. (Regression:
    // an earlier unfloored balance cap elected 527 subproblems for npts=1e4 at 128
    // threads, costing up to +40%.)
    for (double npts : {1e4, 1e5})
      for (int nthr : {32, 128})
        if (n_subproblems(3, 7, npts, npts / 5, bin3, nthr) != nthr) {
          std::cout << "fail: npts=" << npts << " on " << nthr
                    << " threads shattered into "
                    << n_subproblems(3, 7, npts, npts / 5, bin3, nthr)
                    << " subproblems\n";
          return 1;
        }

    // Clustering must not elect a larger subproblem than the uniform case: packing
    // the same points into fewer bins lowers the per-point subgrid overhead, which
    // is the whole reason this reads measured occupancy instead of mean density.
    for (int dim = 2; dim <= 3; ++dim) {
      const double *bs = (dim == 2) ? bin2 : bin3;
      if (n_subproblems(dim, 7, 1e7, 1e4, bs, 16) <
          n_subproblems(dim, 7, 1e7, 1e6, bs, 16)) {
        printf("fail: clustered elects fewer subproblems than uniform in %dD\n", dim);
        return 1;
      }
    }

    // Concentrated points (occupancy far above one subgrid's worth) take the sqrt
    // branch of the ramp, where sp falls below SP_REF. The tests above never reach
    // it: at 5 pts/bin the overhead ratio stays above 1 and the ramp is the identity.
    // Measured: 3D at ~5e3 pts/bin wants ~1e4 pts/subproblem, 3x below SP_REF.
    for (int dim = 2; dim <= 3; ++dim) {
      const double *bs = (dim == 2) ? bin2 : bin3;
      const auto conc = n_subproblems(dim, 7, 1e7, 1e7 / 5000, bs, 22);
      const auto unif = n_subproblems(dim, 7, 1e7, 1e7 / 30, bs, 22);
      if (conc <= unif) {
        std::cout << "fail: " << dim << "D concentrated elects " << conc
                  << " subproblems, not more than uniform's " << unif << "\n";
        return 1;
      }
    }

    // ...but the ramp must stop at one bin's worth of points. Below that a
    // subproblem covers no less of the grid than the bin it sits in, so splitting
    // further is pure redundancy - and unbounded splitting is the small-npts
    // shattering defect again, arriving from the concentrated side.
    // Two floors outrank this one and are exempted: nb == nthreads (fewer
    // subproblems than threads starves the schedule outright) and SP_MAX, which
    // caps per-thread scratch however few bins the points occupy.
    for (double npts : {1e6, 1e8})
      for (double occ : {10.0, 1e3}) { // pathological: all mass in a few bins
        const auto nb = n_subproblems(3, 7, npts, occ, bin3, 22);
        const auto want = std::min(npts / occ, 1000000.0); // one bin, capped by SP_MAX
        if (nb > 22 && npts / (double)nb < want / 2) {
          std::cout << "fail: occupancy " << npts / occ << " pts/bin split into "
                    << npts / (double)nb << " pts/subproblem, below one bin (" << want
                    << ")\n";
          return 1;
        }
      }

    // The up-ramp saturates: a subproblem takes consecutive points in bin order, so
    // past SPAN_MAX bins its padded halo is already negligible and growing further
    // only costs locality. Without the bound the sparse regime N >> npts elects ~12x
    // the measured argmin. The nb == nthreads floor outranks this and is exempted;
    // the factor 2 is the slack snapping to a multiple of nthreads can add back.
    for (int dim = 2; dim <= 3; ++dim) {
      const double *bs = (dim == 2) ? bin2 : bin3;
      for (int ns : ns_list)
        for (double npts : {1e6, 1e8})
          for (double per_bin : {7.0, 25.0, 136.0}) { // sparse: few points per bin
            const auto nb = n_subproblems(dim, ns, npts, npts / per_bin, bs, 16);
            const double sp = npts / (double)nb;
            if (nb > 16 && sp > 2 * 5000.0 * per_bin) {
              std::cout << "fail: " << dim << "D ns=" << ns << " npts=" << npts << " at "
                        << per_bin << " pts/bin elected " << sp
                        << " pts/subproblem, over the " << 5000.0 * per_bin
                        << " span bound\n";
              return 1;
            }
          }
    }

    // No occupancy measure (1D, or an unsorted point set): defer to the caller's cap.
    if (n_subproblems(1, 7, 1e7, 1e5, bin2, 16) ||
        n_subproblems(2, 7, 1e7, 0, bin2, 16)) {
      printf("fail: expected 0 (defer to spopts.max_subproblem_size)\n");
      return 1;
    }
  }
#endif

  // The blocked spread path (nb > 1) must agree with the single-subproblem path.
  // ctest's other transforms only ever reach nb == min(nthreads, M) with whatever
  // subproblem size the heuristic elects, so nothing else pins the decomposition
  // against a nb == 1 reference.
  //
  // spreadinterponly, not a full transform: with the FFT in the loop this compares
  // ducc0's thread-count-dependent rounding (measured 2.4e-11 at 96 threads, and
  // 1.7e-11 between two identical reruns) rather than the spreader, so no
  // eps-scaled bound can hold. Spread-only, the two decompositions agree to
  // ~5e-16 at every thread count from 6 to 96.
  // 3D as well as 2D: add_wrapped_subgrid has a third axis to fold, and a padded
  // subgrid can wrap the grid in one dim while fitting in another.
  for (int dim = 2; dim <= 3; ++dim) {
    const BIGINT M = 200000, Nd = (dim == 2) ? 256 : 64;
    const BIGINT Ntot = (dim == 2) ? Nd * Nd : Nd * Nd * Nd;
    const FLT tol = (FLT)(sizeof(FLT) == 4 ? 1e-4 : 1e-9);
    std::vector<FLT> x(M), y(M), z(M);
    std::vector<CPX> c(M), F_blocked(Ntot), F_single(Ntot);
    for (BIGINT j = 0; j < M; ++j) { // deterministic, spread over the box
      x[j] = (FLT)(M_PI * (2.0 * ((j * 2654435761u) % 1000003) / 1000003.0 - 1.0));
      y[j] = (FLT)(M_PI * (2.0 * ((j * 40503u) % 999983) / 999983.0 - 1.0));
      z[j] = (FLT)(M_PI * (2.0 * ((j * 2246822519u) % 999979) / 999979.0 - 1.0));
      c[j] = CPX((FLT)(1.0 - 2.0 * (j % 3)), (FLT)(0.5 * (j % 5)));
    }
    BIGINT Ns[3] = {Nd, Nd, Nd};
    // arm 0: 1000 pts/subproblem forces nb = 200. arm 1: one thread and no cap is
    // the nb == 1 reference (nb = max(min(nthreads, M), ceil(M/sp))).
    const int sp_forced[2] = {1000, 1 << 30}, nthr_forced[2] = {0, 1};
    std::vector<CPX> *out[2] = {&F_blocked, &F_single};
    for (int k = 0; k < 2; ++k) {
      finufft_opts o;
      FINUFFT_DEFAULT_OPTS(&o);
      o.spreadinterponly = 1;
      o.upsampfac = 2.0; // only sets the kernel when spreadinterponly
      o.spread_max_sp_size = sp_forced[k];
      o.nthreads = nthr_forced[k];
      FINUFFT_PLAN p;
      if (FINUFFT_MAKEPLAN(1, dim, Ns, 1, 1, tol, &p, &o)) {
        printf("fail: makeplan failed in blocked-spread test\n");
        return 1;
      }
      FINUFFT_SETPTS(p, M, x.data(), y.data(), dim == 3 ? z.data() : nullptr, 0, nullptr,
                     nullptr, nullptr);
      FINUFFT_EXECUTE(p, c.data(), out[k]->data());
      FINUFFT_DESTROY(p);
    }
    const auto err = relerrtwonorm(Ntot, F_single.data(), F_blocked.data());
    // nb only changes the add-back summation order, so bound on eps, not tol.
    // Measured 5.1e-16 (f64) / 1.1e-07 (f32), identical at 1, 6 and 22 threads,
    // while a bug losing one point per subproblem boundary gives 200/2e5 = 1e-3.
    if (!(err < 1000 * (FLT)std::numeric_limits<FLT>::epsilon())) {
      printf("fail: %dD blocked (nb=200) vs single-subproblem spread differ: %.3g\n", dim,
             (double)err);
      return 1;
    }
  }

#ifdef SINGLE
  printf("testutilsf passed.\n");
#else
  printf("testutils passed.\n");
#endif
  return 0;
}
