#include <finufft/heuristics.hpp>
#include <finufft/test_defs.hpp>

#include "utils/dirft1d.hpp"
#include "utils/dirft3d.hpp"
#include "utils/norms.hpp"

namespace {

// RAII guard for FINUFFT_PLAN handles — ensures cleanup on all exit paths.
struct PlanGuard {
  FINUFFT_PLAN p = nullptr;
  ~PlanGuard() {
    if (p) FINUFFT_DESTROY(p);
  }
  // Non-copyable, non-movable (simple scope guard)
  PlanGuard()                             = default;
  PlanGuard(const PlanGuard &)            = delete;
  PlanGuard &operator=(const PlanGuard &) = delete;
};

// Check FINUFFT API return codes — prints label on failure and exits.
const auto check = [](int ier, const char *label) {
  if (ier != 0) {
    std::fprintf(stderr, "setpts_replan: %s failed: ier=%d\n", label, ier);
    exit(1);
  }
};

bool upsamp_matches(double actual, double expected) {
  return std::abs(actual - expected) <= (std::is_same_v<FLT, float> ? 1e-6 : 1e-12);
}

void fill_type1_points(std::vector<FLT> &x, std::vector<FLT> &y, std::vector<FLT> &z,
                       std::vector<CPX> &c, FLT shift) {
  for (BIGINT j = 0; j < static_cast<BIGINT>(x.size()); ++j) {
    const FLT t = FLT(j + 1) + shift;
    x[j]        = PI * std::sin(FLT(0.173) * t + FLT(0.11));
    y[j]        = PI * std::sin(FLT(0.117) * t + FLT(0.37));
    z[j]        = PI * std::sin(FLT(0.091) * t - FLT(0.23));
    c[j]        = CPX(std::cos(FLT(0.071) * t), std::sin(FLT(0.049) * t));
  }
}

void fill_type3_points(std::vector<FLT> &x, std::vector<CPX> &c, FLT shift) {
  for (BIGINT j = 0; j < static_cast<BIGINT>(x.size()); ++j) {
    const FLT t = FLT(j + 1) + shift;
    x[j]        = PI * std::sin(FLT(0.213) * t - FLT(0.31));
    c[j]        = CPX(std::cos(FLT(0.097) * t), std::sin(FLT(0.061) * t));
  }
}

void fill_type3_targets(std::vector<FLT> &s, FLT shift) {
  for (BIGINT k = 0; k < static_cast<BIGINT>(s.size()); ++k) {
    const FLT t = FLT(k + 1) + shift;
    s[k]        = FLT(12.0) * std::sin(FLT(0.157) * t + FLT(0.19));
  }
}

int test_type12_auto_replan() {
  constexpr int type     = 1;
  constexpr int dim      = 3;
  constexpr int isign    = +1;
  constexpr int ntrans   = 1;
  constexpr BIGINT Ns[3] = {6, 6, 6};
  constexpr BIGINT N     = Ns[0] * Ns[1] * Ns[2];
#ifdef SINGLE
  constexpr FLT tol         = 1e-4f;
  constexpr FLT allowed_err = 5e-3f;
  constexpr BIGINT Mlo      = 64;
  constexpr BIGINT Mhi      = 2000;
#else
  constexpr FLT tol         = 1e-6;
  constexpr FLT allowed_err = 5e-5;
  constexpr BIGINT Mlo      = 64;
  constexpr BIGINT Mhi      = 432;
#endif

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.nthreads  = 1;
  opts.upsampfac = 0.0;

  const double density_lo = double(Mlo) / double(N);
  const double density_hi = double(Mhi) / double(N);
  const double usf_lo     = finufft::heuristics::bestUpsamplingFactor<FLT>(
      opts.nthreads, density_lo, dim, type, tol);
  const double usf_hi = finufft::heuristics::bestUpsamplingFactor<FLT>(
      opts.nthreads, density_hi, dim, type, tol);

  std::vector<FLT> xlo(Mlo), ylo(Mlo), zlo(Mlo);
  std::vector<CPX> clo(Mlo), Flo(N);
  std::vector<FLT> xhi(Mhi), yhi(Mhi), zhi(Mhi);
  std::vector<CPX> chi(Mhi), F_reuse(N), F_fresh(N), F_ref(N);
  fill_type1_points(xlo, ylo, zlo, clo, FLT(0.0));
  fill_type1_points(xhi, yhi, zhi, chi, FLT(0.5));

  PlanGuard plan, fresh;
  check(FINUFFT_MAKEPLAN(type, dim, Ns, isign, ntrans, tol, &plan.p, &opts),
        "makeplan(auto)");

  check(FINUFFT_SETPTS(plan.p, Mlo, xlo.data(), ylo.data(), zlo.data(), 0, nullptr,
                       nullptr, nullptr),
        "first setpts(auto)");
  auto *impl = reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(plan.p);
  if (!upsamp_matches(impl->opts.upsampfac, usf_lo)) {
    std::fprintf(stderr, "setpts_replan: expected first auto usf %.3g, got %.3g\n",
                 usf_lo, impl->opts.upsampfac);
    return 1;
  }

  check(FINUFFT_EXECUTE(plan.p, clo.data(), Flo.data()), "first execute(auto)");

  check(FINUFFT_SETPTS(plan.p, Mhi, xhi.data(), yhi.data(), zhi.data(), 0, nullptr,
                       nullptr, nullptr),
        "second setpts(auto)");
  if (!upsamp_matches(impl->opts.upsampfac, usf_hi)) {
    std::fprintf(stderr, "setpts_replan: expected second auto usf %.3g, got %.3g\n",
                 usf_hi, impl->opts.upsampfac);
    return 1;
  }

  check(FINUFFT_EXECUTE(plan.p, chi.data(), F_reuse.data()), "second execute(auto)");

  check(FINUFFT_MAKEPLAN(type, dim, Ns, isign, ntrans, tol, &fresh.p, &opts),
        "fresh makeplan(auto)");
  check(FINUFFT_SETPTS(fresh.p, Mhi, xhi.data(), yhi.data(), zhi.data(), 0, nullptr,
                       nullptr, nullptr),
        "fresh setpts(auto)");
  auto *fresh_impl = reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(fresh.p);
  if (!upsamp_matches(fresh_impl->opts.upsampfac, usf_hi)) {
    std::fprintf(stderr, "setpts_replan: expected fresh auto usf %.3g, got %.3g\n",
                 usf_hi, fresh_impl->opts.upsampfac);
    return 1;
  }
  check(FINUFFT_EXECUTE(fresh.p, chi.data(), F_fresh.data()), "fresh execute(auto)");

  dirft3d1(Mhi, xhi, yhi, zhi, chi, isign, Ns[0], Ns[1], Ns[2], F_ref);
  const FLT reuse_vs_fresh = relerrtwonorm(N, F_fresh.data(), F_reuse.data());
  const FLT reuse_vs_ref   = relerrtwonorm(N, F_ref.data(), F_reuse.data());

  if (reuse_vs_fresh > allowed_err || reuse_vs_ref > allowed_err) {
    std::fprintf(
        stderr,
        "setpts_replan: auto type12 mismatch (reuse/fresh=%.3g, reuse/ref=%.3g)\n",
        reuse_vs_fresh, reuse_vs_ref);
    return 1;
  }
  return 0;
}

int test_type12_locked() {
  constexpr int type     = 1;
  constexpr int dim      = 3;
  constexpr int isign    = +1;
  constexpr int ntrans   = 1;
  constexpr BIGINT Ns[3] = {6, 6, 6};
  constexpr BIGINT N     = Ns[0] * Ns[1] * Ns[2];
#ifdef SINGLE
  constexpr FLT tol         = 1e-4f;
  constexpr FLT allowed_err = 5e-3f;
#else
  constexpr FLT tol         = 1e-6;
  constexpr FLT allowed_err = 5e-5;
#endif
  constexpr BIGINT Mlo = 64;
  constexpr BIGINT Mhi = 512;

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.nthreads  = 1;
  opts.upsampfac = 2.0;

  std::vector<FLT> xlo(Mlo), ylo(Mlo), zlo(Mlo);
  std::vector<CPX> clo(Mlo), Flo(N);
  std::vector<FLT> xhi(Mhi), yhi(Mhi), zhi(Mhi);
  std::vector<CPX> chi(Mhi), F_reuse(N), F_fresh(N);
  fill_type1_points(xlo, ylo, zlo, clo, FLT(0.2));
  fill_type1_points(xhi, yhi, zhi, chi, FLT(0.9));

  PlanGuard plan, fresh;
  check(FINUFFT_MAKEPLAN(type, dim, Ns, isign, ntrans, tol, &plan.p, &opts),
        "makeplan(locked)");

  check(FINUFFT_SETPTS(plan.p, Mlo, xlo.data(), ylo.data(), zlo.data(), 0, nullptr,
                       nullptr, nullptr),
        "first setpts(locked)");
  auto *impl = reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(plan.p);
  if (!upsamp_matches(impl->opts.upsampfac, opts.upsampfac)) {
    std::fprintf(stderr, "setpts_replan: locked usf changed after first setpts\n");
    return 1;
  }

  check(FINUFFT_EXECUTE(plan.p, clo.data(), Flo.data()), "first execute(locked)");

  check(FINUFFT_SETPTS(plan.p, Mhi, xhi.data(), yhi.data(), zhi.data(), 0, nullptr,
                       nullptr, nullptr),
        "second setpts(locked)");
  if (!upsamp_matches(impl->opts.upsampfac, opts.upsampfac)) {
    std::fprintf(stderr, "setpts_replan: locked usf changed after second setpts\n");
    return 1;
  }

  check(FINUFFT_EXECUTE(plan.p, chi.data(), F_reuse.data()), "second execute(locked)");

  check(FINUFFT_MAKEPLAN(type, dim, Ns, isign, ntrans, tol, &fresh.p, &opts),
        "fresh makeplan(locked)");
  check(FINUFFT_SETPTS(fresh.p, Mhi, xhi.data(), yhi.data(), zhi.data(), 0, nullptr,
                       nullptr, nullptr),
        "fresh setpts(locked)");
  check(FINUFFT_EXECUTE(fresh.p, chi.data(), F_fresh.data()), "fresh execute(locked)");

  const FLT reuse_vs_fresh = relerrtwonorm(N, F_fresh.data(), F_reuse.data());

  if (reuse_vs_fresh > allowed_err) {
    std::fprintf(stderr, "setpts_replan: locked type12 mismatch (reuse/fresh=%.3g)\n",
                 reuse_vs_fresh);
    return 1;
  }
  return 0;
}

int test_type3_plan_reuse() {
  constexpr int type              = 3;
  constexpr int dim               = 1;
  constexpr int isign             = +1;
  constexpr int ntrans            = 1;
  constexpr BIGINT dummy_modes[3] = {1, 1, 1};
#ifdef SINGLE
  constexpr FLT tol         = 1e-3f;
  constexpr FLT allowed_err = 2e-2f;
#else
  constexpr FLT tol         = 1e-6;
  constexpr FLT allowed_err = 2e-5;
#endif
  constexpr BIGINT M1  = 40;
  constexpr BIGINT NK1 = 48;
  constexpr BIGINT M2  = 57;
  constexpr BIGINT NK2 = 61;

  finufft_opts opts;
  FINUFFT_DEFAULT_OPTS(&opts);
  opts.nthreads  = 1;
  opts.upsampfac = 0.0;

  const double expected_usf =
      finufft::heuristics::bestUpsamplingFactor<FLT>(opts.nthreads, 1.0, dim, type, tol);

  std::vector<FLT> x1(M1), x2(M2), s1(NK1), s2(NK2);
  std::vector<CPX> c1(M1), c2(M2), F1(NK1), F_reuse(NK2), F_fresh(NK2), F_ref(NK2);
  fill_type3_points(x1, c1, FLT(0.0));
  fill_type3_points(x2, c2, FLT(0.7));
  fill_type3_targets(s1, FLT(0.0));
  fill_type3_targets(s2, FLT(0.4));

  PlanGuard plan, fresh;
  check(FINUFFT_MAKEPLAN(type, dim, dummy_modes, isign, ntrans, tol, &plan.p, &opts),
        "makeplan(type3)");

  check(FINUFFT_SETPTS(plan.p, M1, x1.data(), nullptr, nullptr, NK1, s1.data(), nullptr,
                       nullptr),
        "first setpts(type3)");
  auto *impl = reinterpret_cast<FINUFFT_PLAN_T<FLT> *>(plan.p);
  if (!upsamp_matches(impl->opts.upsampfac, expected_usf)) {
    std::fprintf(stderr, "setpts_replan: expected type3 usf %.3g, got %.3g\n",
                 expected_usf, impl->opts.upsampfac);
    return 1;
  }

  check(FINUFFT_EXECUTE(plan.p, c1.data(), F1.data()), "first execute(type3)");

  check(FINUFFT_SETPTS(plan.p, M2, x2.data(), nullptr, nullptr, NK2, s2.data(), nullptr,
                       nullptr),
        "second setpts(type3)");
  if (!upsamp_matches(impl->opts.upsampfac, expected_usf)) {
    std::fprintf(stderr, "setpts_replan: type3 usf changed after second setpts\n");
    return 1;
  }

  check(FINUFFT_EXECUTE(plan.p, c2.data(), F_reuse.data()), "second execute(type3)");

  check(FINUFFT_MAKEPLAN(type, dim, dummy_modes, isign, ntrans, tol, &fresh.p, &opts),
        "fresh makeplan(type3)");
  check(FINUFFT_SETPTS(fresh.p, M2, x2.data(), nullptr, nullptr, NK2, s2.data(), nullptr,
                       nullptr),
        "fresh setpts(type3)");
  check(FINUFFT_EXECUTE(fresh.p, c2.data(), F_fresh.data()), "fresh execute(type3)");

  dirft1d3(M2, x2, c2, isign, NK2, s2, F_ref);
  const FLT reuse_vs_fresh = relerrtwonorm(NK2, F_fresh.data(), F_reuse.data());
  const FLT reuse_vs_ref   = relerrtwonorm(NK2, F_ref.data(), F_reuse.data());

  if (reuse_vs_fresh > allowed_err || reuse_vs_ref > allowed_err) {
    std::fprintf(stderr,
                 "setpts_replan: type3 mismatch (reuse/fresh=%.3g, reuse/ref=%.3g)\n",
                 reuse_vs_fresh, reuse_vs_ref);
    return 1;
  }
  return 0;
}

} // namespace

int main() {
#ifdef SINGLE
  std::printf("setpts_replanf started...\n");
#else
  std::printf("setpts_replan started...\n");
#endif

  if (test_type12_auto_replan() != 0) return 1;
  if (test_type12_locked() != 0) return 1;
  if (test_type3_plan_reuse() != 0) return 1;

#ifdef SINGLE
  std::printf("setpts_replanf passed.\n");
#else
  std::printf("setpts_replan passed.\n");
#endif
  return 0;
}
