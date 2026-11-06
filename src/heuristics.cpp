// Upsampfac (sigma) selection for plans made with upsampfac=0: a flop model and the
// search over kernel widths that minimizes it.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <finufft/plan.hpp>
#include <finufft/simd.hpp> // get_padding, get_padded_simd_width
#include <finufft_common/constants.h>
#include <finufft_common/kernel.h>

using namespace finufft::common;
using namespace finufft::kernel;

static constexpr double SIGMA_EDGE_PAD = 1e-3; // sigma margin above the law boundary

// Per-dim spread work factors: the 2*ns_x-wide complex row on the fast dim x runs as
// ceil(2*ns_x/simd_width) SIMD ops (get_padded_simd_width on x only), then ns_y and
// ns_z count rows. Returns the work per NU point and, in ops, the per-dim factors.
static double spread_flops(bool is_float, double npts, const std::array<int, 3> &ns,
                           int dim, std::array<double, 3> *ops = nullptr) {
  using namespace finufft::spreadinterp;
  const double padded_2ns = 2.0 * ns[0] + (is_float ? get_padding<float>(2 * ns[0])
                                                    : get_padding<double>(2 * ns[0]));
  const double simd       = (double)(is_float ? get_padded_simd_width<float>(2 * ns[0])
                                              : get_padded_simd_width<double>(2 * ns[0]));
  std::array<double, 3> f{std::max(padded_2ns / simd, 1.0), 1.0, 1.0};
  double rows = 1.0;
  for (int idim = 1; idim < dim; ++idim) {
    f[idim] = (double)ns[idim];
    rows *= (double)ns[idim];
  }
  if (ops) *ops = f;
  return npts * f[0] * rows;
}

// Type 1/2: spread the NU points, then the FFT c*G*log2(G) on the fine grid of volume
// G. In a type-3 plan this is the inner type 2, with nk points. c weighs one FFT point
// against one spread op; it rises with threads, as spreading scales ~linearly and the
// FFT sublinearly. Calibrated on ccmlin075 (AVX-512, FFTW).
template<typename TF>
double FINUFFT_PLAN_T<TF>::flops_type12(int ns, double sigma,
                                        const std::array<double, 3> &nmodes) const {
  const double npts = type == 3 ? m.nk : m.nj;
  const double c    = 2.0 * std::sqrt((double)std::max(1, opts.nthreads));
  std::array<double, 3> nf{1.0, 1.0, 1.0};
  for (int idim = 0; idim < dim; ++idim)
    nf[idim] = (double)fine_grid_len(sigma, nmodes[idim], ns);
  double G = 1.0;
  for (int idim = 0; idim < dim; ++idim) G *= nf[idim]; // FFT term on the per-dim nf
  return spread_flops(std::is_same_v<TF, float>, npts, {ns, ns, ns}, dim) +
         c * G * std::log2(G);
}

// Type 3: spread nj sources onto the nhg_type3 grid, then the inner type 2 to nk
// targets at its own best sigma. S holds the target half-widths.
template<typename TF>
double FINUFFT_PLAN_T<TF>::flops_type3(int ns, double sigma,
                                       const std::array<double, 3> &S) const {
  const bool is_float = std::is_same_v<TF, float>;
  std::array<double, 3> nf{1.0, 1.0, 1.0};
  for (int idim = 0; idim < dim; ++idim)
    nf[idim] = (double)std::get<0>(nhg_type3(sigma, m.t3P.X[idim], S[idim], ns, MAX_NF));
  const double sigma2 = best_sigma_for_tol(2, nf);
  if (!std::isfinite(sigma2)) return std::numeric_limits<double>::infinity();
  const int ns2 = clamp_kernel_ns(theoretical_kernel_ns(m.tol, dim, 2, sigma2), sigma2,
                                  dim, is_float);
  return spread_flops(is_float, (double)m.nj, {ns, ns, ns}, dim) +
         flops_type12(ns2, sigma2, nf);
}

// Fewest-flops sigma in [MIN_AUTO_UPSAMPFAC, MAX_AUTO_UPSAMPFAC] that meets tol, or
// infinity if none does. nmodes holds the mode counts for type 1/2 and the target
// half-widths S for type 3.
template<typename TF>
double FINUFFT_PLAN_T<TF>::best_sigma_for_tol(int type,
                                              const std::array<double, 3> &nmodes) const {
  constexpr auto is_float = std::is_same_v<TF, float>;
  const auto eps_mach     = is_float ? std::numeric_limits<float>::epsilon()
                                     : std::numeric_limits<double>::epsilon();
  const auto tol          = m.tol;
  auto sigma              = std::numeric_limits<double>::infinity();
  auto flops              = std::numeric_limits<double>::infinity();
  const int max_ns        = is_float ? MAX_NSPREAD<float> : MAX_NSPREAD<double>;
  auto compute_flops      = [&](double sigma_ns) {
    // price the ns the plan will use at sigma_ns, as setup_spreadinterp sets it
    const int ns_plan = clamp_kernel_ns(theoretical_kernel_ns(tol, dim, type, sigma_ns),
                                        sigma_ns, dim, is_float);
    return type == 3 ? flops_type3(ns_plan, sigma_ns, nmodes)
                     : flops_type12(ns_plan, sigma_ns, nmodes);
  };
  for (int ns = 2; ns <= max_ns; ++ns) {
    const auto sigma_alias = min_sigma_for_aliasing_error(tol, dim, type, ns);
    const auto sigma_deconv = min_sigma_for_deconv_error(tol, dim, ns, eps_mach);
    const auto sigma_ns     = std::max({sigma_alias, sigma_deconv, MIN_AUTO_UPSAMPFAC});
    if (opts.debug > 1)
      printf("[%s] ns=%d: sigma_alias=%.3g sigma_deconv=%.3g sigma_ns=%.3g\n", __func__,
             ns, sigma_alias, sigma_deconv, sigma_ns);
    // also skips infinity; continue because sigma is not really monotone in ns.
    if (sigma_ns > MAX_AUTO_UPSAMPFAC) continue;
    constexpr double edge_pad = SIGMA_EDGE_PAD;
    const auto sigma_pad      = sigma_ns + edge_pad;
    const auto flops_ns       = compute_flops(sigma_pad);
    if (opts.debug > 1) printf("[%s] ns=%d: flops_ns=%.3g\n", __func__, ns, flops_ns);
    if (flops_ns < flops) {
      flops = flops_ns;
      sigma = sigma_pad;
    }
  }
  return sigma;
}

template double FINUFFT_PLAN_T<float>::best_sigma_for_tol(
    int type, const std::array<double, 3> &nmodes) const;
template double FINUFFT_PLAN_T<double>::best_sigma_for_tol(
    int type, const std::array<double, 3> &nmodes) const;

namespace finufft::kernel {

double perdim_set_cost(bool is_float, double npts, const std::array<int, 3> &ns,
                       const std::array<BIGINT, 3> &nf, int dim, int nthreads,
                       std::array<double, 3> *ops_out) {
  std::array<double, 3> ops;
  const double spread = spread_flops(is_float, npts, ns, dim, &ops);
  if (ops_out) *ops_out = ops;
  double G = 1.0;
  for (int idim = 0; idim < dim; ++idim) G *= (double)nf[idim]; // per-dim nf product
  const double c = 2.0 * std::sqrt((double)std::max(1, nthreads));
  return spread + c * G * std::log2(G);
}

bool perdim_pick_set(double tol, int dim, int type, bool is_float, int nthreads,
                     const std::array<double, 3> &N, double npts,
                     const PerdimSet &uniform, bool allow_nf_change, PerdimSet &best,
                     int debug) {
  constexpr double edge_pad = SIGMA_EDGE_PAD; // above the aliasing-law boundary
  const double eps_mach     = is_float ? std::numeric_limits<float>::epsilon()
                                       : std::numeric_limits<double>::epsilon();
  const int max_ns = is_float ? common::MAX_NSPREAD<float> : common::MAX_NSPREAD<double>;
  // Per-dim candidates: the min-sigma search at this dim's ns_d gives its smallest
  // feasible sigma_d, and fine_grid_len (the set_nf_type12 mirror) its nf_d. A forced
  // upsampfac stays one sigma in every dim: width candidates at the uniform nf, sigma.
  std::array<std::vector<PerdimSet>, 3> cand;
  for (int idim = 0; idim < dim; ++idim) {
    for (int ns = max_ns; ns >= common::MIN_NSPREAD; --ns) { // widest first: a cost
      // tie keeps the wider kernel (measured better error at equal cost)
      double sigma_d = uniform.sigma[idim];
      BIGINT nf_d    = uniform.nf[idim];
      if (allow_nf_change) {
        const double sigma_min =
            std::max({min_sigma_for_aliasing_error(tol, dim, type, ns),
                      min_sigma_for_deconv_error(tol, dim, ns, eps_mach),
                      common::MIN_AUTO_UPSAMPFAC});
        if (!(sigma_min <= common::MAX_AUTO_UPSAMPFAC)) continue; // includes infinity
        nf_d = common::fine_grid_len(sigma_min + edge_pad, N[idim], ns);
        if (nf_d >= MAX_NF) continue;
        sigma_d = (double)nf_d / N[idim];
      }
      // the ns caps run per dim at that dim's sigma_d
      if (clamp_kernel_ns(ns, sigma_d, dim, is_float) != ns) continue;
      PerdimSet s{};
      s.ns[idim]    = ns;
      s.nf[idim]    = nf_d;
      s.sigma[idim] = sigma_d;
      cand[idim].push_back(s);
    }
    if (cand[idim].empty()) { // no width meets tol in this dim; check_sigma refuses
      best = uniform;
      best.cost =
          perdim_set_cost(is_float, npts, best.ns, best.nf, dim, nthreads, &best.ops);
      return false;
    }
  }
  best      = uniform;
  best.cost = perdim_set_cost(is_float, npts, best.ns, best.nf, dim, nthreads, &best.ops);
  const std::vector<PerdimSet> one(1); // stands in for absent dims
  const auto &c1 = dim > 1 ? cand[1] : one;
  const auto &c2 = dim > 2 ? cand[2] : one;
  for (const auto &a : cand[0])
    for (const auto &b : c1)
      for (const auto &e : c2) {
        PerdimSet s{};
        s.ns      = {a.ns[0], b.ns[1], e.ns[2]};
        s.nf      = {a.nf[0], b.nf[1], e.nf[2]};
        s.sigma   = {a.sigma[0], b.sigma[1], e.sigma[2]};
        int ns_lo = s.ns[0], ns_hi = s.ns[0];
        for (int idim = 1; idim < dim; ++idim) {
          ns_lo = std::min(ns_lo, s.ns[idim]);
          ns_hi = std::max(ns_hi, s.ns[idim]);
        }
        // only the dispatched width tuples are realizable (kernel instances)
        if (ns_hi - ns_lo > common::PERDIM_MAX_NS_DIFF) continue;
        // spreadcheck needs every non-trivial grid dim >= 2 * max width
        bool floor_ok = true;
        for (int idim = 0; idim < dim; ++idim)
          floor_ok = floor_ok && s.nf[idim] >= 2 * (BIGINT)ns_hi;
        if (!floor_ok) continue;
        // the same summed estimate check_sigma applies, so auto cannot pick a set the
        // plan later refuses
        if (!perdim_estimates_ok(tol, dim, type, is_float, N, s.sigma, s.ns)) continue;
        s.cost = perdim_set_cost(is_float, npts, s.ns, s.nf, dim, nthreads, &s.ops);
        if (debug > 1)
          printf("[perdim_pick] ns=(%d,%d,%d) nf=(%lld,%lld,%lld) sigma=(%.4g,%.4g,%.4g) "
                 "cost=%.4g\n",
                 s.ns[0], s.ns[1], s.ns[2], (long long)s.nf[0], (long long)s.nf[1],
                 (long long)s.nf[2], s.sigma[0], s.sigma[1], s.sigma[2], s.cost);
        if (s.cost < best.cost) best = s; // strict: a tie keeps the earlier wider set
      }
  return best.ns != uniform.ns || best.nf != uniform.nf;
}

} // namespace finufft::kernel
