#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <limits>
#include <type_traits>
#include <vector>

#include <finufft_common/constants.h>
#include <finufft_common/pswf.h>
#include <finufft_common/spread_opts.h>
#include <finufft_common/utils.h>
#include <finufft_errors.h>

namespace finufft::common {
// Rounding error ROUND_FAC*eps_mach*N_d for one dimension's N_d modes.
// perdim_estimates_ok below sums it over dimensions, so the total rounding floor
// is eps_mach*ROUND_FAC*sum_d N_d. No sigma lowers it. Declared here because
// perdim_estimates_ok calls it; defined in src/common/kernel.cpp.
double rounding_error(double eps_mach, double mode_diagonal);
} // namespace finufft::common

namespace finufft::kernel {

template<class T, class F> std::vector<T> poly_fit(F &&f, int n) {
  static_assert(std::is_floating_point_v<T>, "T must be floating-point");
  /* Expects f, a function handle for arguments on [-1,1], both I/O type T.
     Returns vector of n coefficients a_{n-1}, ... a_1, a_0 of degree-(n-1)
     polynomial that interpolates f at a set of hard-wired Chebyshev nodes.

     Barbone, Fall 2025.
     Barnett 12/29/25-1/13/26 removed a,b to simplify; no poly defn confusion.
  */

  // 1) Type-1 Chebyshev nodes t_k, data samples y_k = f(t_k)
  std::vector<T> t(n), y(n);
  for (int k = 0; k < n; ++k) {
    t[k] = std::cos((T(2 * k + 1) * common::PI) / (T(2) * T(n))); // in (-1,1)
    // t[k]       = std::cos((T(k) * common::PI) / T(n-1)); // type-2 in [-1,1] also ok
    y[k] = static_cast<T>(f(t[k])); // evaluate this sample
  }

  // 2) Newton divided differences on t: coef[j] = f[t0,..,tj]
  std::vector<T> coef = y;
  for (int j = 1; j < n; ++j)
    for (int i = n - 1; i >= j; --i)
      coef[i] = (coef[i] - coef[i - 1]) / (t[i] - t[i - j]);

  // 3) Convert Newton form to monomial coeffs in t (low→high)
  // Multiply a polynomial p(t) by (t - c).
  // Input p holds coefficients low->high (p[0] + p[1] t + p[2] t^2 + ...).
  // Returns r of length p.size()+1 so that r represents (t - c)*p(t).
  // Concretely: r[i] = -c*p[i] + (i>0 ? p[i-1] : 0) for i=0..p.size()-1,
  // and r[p.size()] = p[p.size()-1].
  auto mul_by_linear = [](const std::vector<T> &p, T c) {
    std::vector<T> r(p.size() + 1, T(0));
    for (std::size_t i = 0; i < p.size(); ++i) {
      r[i] += -c * p[i]; // (t - c)*p
      r[i + 1] += p[i];
    }
    return r;
  };

  std::vector<T> c(n, T(0));  // output coefficients in t
  std::vector<T> basis{T(1)}; // Π_{m<j} (t - t_m), low→high
  c[0] += coef[0];
  for (int j = 1; j < n; ++j) {
    basis = mul_by_linear(basis, t[j - 1]);
    for (std::size_t m = 0; m < basis.size(); ++m) c[m] += coef[j] * basis[m];
  }
  std::reverse(c.begin(), c.end()); // convert to Horner ordering (high -> low)
  return c;
}

// The spread/interp kernel phi_beta(z) on z in [-1,1]. Not performance-critical;
// used only for polynomial interpolation (precompute_horner_coeffs). Always double.
// Defined in src/common/kernel.cpp.
std::function<double(double)> kernel_definition_lambda(double beta, int kerformula);
std::function<double(double)> kernel_definition_lambda(const finufft_spread_opts &spopts);

// The one beta law: switch on kerformula and return the shape parameter for a kernel of
// width nspread at upsampling ratio upsampfac (the realized nf/N for per-dimension
// calls). Defined in src/common/kernel.cpp.
double kernel_beta_given_ns(int nspread, double upsampfac, int kerformula);

int theoretical_kernel_ns(double tol, int dim, int type, double sigma);

double min_sigma_for_aliasing_error(double tol, int dim, int type, int ns);

double min_sigma_for_deconv_error(double tol, int dim, int ns, double eps_mach);

// Clamp a theoretical kernel width to the width the plan will actually use: clip to
// [MIN_NSPREAD, MAX_NSPREAD] (widest spreadinterp template compiled),
// then the float catastrophic-cancellation guard. Shared by setup_spreadinterp() and
// the upsampfac heuristic so they agree on the width.
int clamp_kernel_ns(int ns, double upsampfac, int dim, bool is_float);

// The three refusal estimates on per-dimension sigmas sigma_d (= nf_d / N_d in auto
// mode): rounding at N_d, aliasing and deconvolution at width ns_d. check_sigma
// (setpts.hpp) and the per-dimension selector (src/heuristics.cpp) share this one
// test, so the two cannot disagree. Each estimate is summed over dimensions (the
// safe combination); every sum must be <= tol. The aliasing and deconvolution
// terms carry a 1/dim share of the fitted uniform-law prefactors, so equal
// dimensions reproduce those two laws exactly. The rounding term sums as
// sum_d N_d, not the uniform law's 2-norm. sums, when given, receives the
// (rounding, aliasing, deconvolution) sums. Defined in src/common/kernel.cpp.
bool perdim_estimates_ok(
    double tol, int dim, int type, bool is_float, const std::array<double, 3> &n,
    const std::array<double, 3> &sigma_d, const std::array<int, 3> &ns,
    std::array<double, 3> *sums = nullptr);

// One per-dimension (ns, nf) set for the retune, its realized ratios and its modeled
// cost. ops holds the per-dim spread factors: the padded 2*ns_x-wide row on the fast
// dim x (get_padded_simd_width on x only), then the ns_y and ns_z row counts.
struct PerdimSet {
  std::array<int, 3> ns{};       // kernel width per dimension
  std::array<BIGINT, 3> nf{};    // fine grid length per dimension
  std::array<double, 3> sigma{}; // realized ratio nf_d / N_d
  std::array<double, 3> ops{};   // per-dim spread work factors
  double cost = std::numeric_limits<double>::infinity();
};

// Cost of one (ns, nf) set: npts * prod_d ops_d spread work plus the FFT
// c * G * log2(G) on the per-dim fine grid G = prod_d nf_d. Defined in
// src/heuristics.cpp (the ops need get_padded_simd_width).
double perdim_set_cost(bool is_float, double npts, const std::array<int, 3> &ns,
                       const std::array<BIGINT, 3> &nf, int dim, int nthreads,
                       std::array<double, 3> *ops_out = nullptr);

// Cost-minimal per-dimension (ns, nf) set meeting the summed error estimate, over the
// per-dim min-sigma candidates (allow_nf_change, auto) or the width-only candidates at
// the uniform nf (forced upsampfac stays one sigma in every dim). Width tuples must
// stay within PERDIM_MAX_NS_DIFF, the dispatched kernel set. best holds the winner
// (the uniform set when nothing beats or meets it); returns true when it differs from
// uniform. Defined in src/heuristics.cpp. debug > 1 prints the candidate costs.
bool perdim_pick_set(double tol, int dim, int type, bool is_float, int nthreads,
                     const std::array<double, 3> &N, double npts,
                     const PerdimSet &uniform, bool allow_nf_change, PerdimSet &best,
                     int debug);

void set_kernel_shape_given_ns(finufft_spread_opts &opts, int debug);

// Analytic PSWF self-FT parameters. The order-0 PSWF is an eigenfunction of the
// finite Fourier transform, so the kernel's continuous FT is a rescaled copy of the
// kernel itself: phihat(xi) = prefac * phi(grid_scale*xi), with phi the Horner
// approximant (evaluate_kernel_runtime, grid-unit arg), grid_scale = J2^2/beta,
// J2 = nspread/2, and prefac = phihat(0) = int phi dx (as phi(0)=1). That integral
// is closed-form from the panel coeffs: on panel i, phi = sum_j coeffs[j*stride+i]
// z^(nc-1-j), z in [-1,1], dz = 2dx, so odd powers cancel and the whole setup is
// ~ns.nc/2 flops, no quadrature. Integrating the approximant rather than the exact
// PSWF makes phihat the FT of the function actually spread. Valid only inside the
// kernel support, |xi| <= 1/grid_scale = 2.beta/nspread; type-3 deconv evaluates at
// |xi| <= pi/upsampfac (= h.gam.S, setpts), and over every reachable (tol, dim,
// precision, upsampfac >= 1.1) the margin 2.beta/nspread - pi/upsampfac is >= 0.47
// for kerformula 7-9. Returns (grid_scale, prefac). Used by the type-3 per-target
// evaluator (Kernel_onedim_FT); type-1/2 fseries keeps quadrature. GPU-reusable.
// Barbone 7/23/26.
template<class TF>
inline std::tuple<double, double> pswf_selfft_params(
    int nspread, double beta, const TF *coeffs, int nc, int stride) {
  double prefac = 0;
  for (int i = 0; i < nspread; ++i)
    for (int j = nc - 1; j >= 0; j -= 2) // even powers of z only
      prefac += double(coeffs[j * stride + i]) / (nc - j);
  const double J2 = nspread / 2.0; // half-width of kernel z-support
  return {J2 * J2 / beta, prefac};
}

// min and max number of poly coeffs allowed (compiled) for a given spread width ns.
// Since for low upsampfacs, ns=16 can need only nc~12, allow such low nc here.
// Note: spreadinterp.cpp compilation time grows with the gap between these bounds...
inline constexpr int min_nc_given_ns(int ns) {
  return std::max(common::MIN_NC, ns - 4); // note must stay in bounds from constants.h
}
inline constexpr int max_nc_given_ns(int ns) {
  return std::min(common::MAX_NC, ns + 3); // "
}

template<int NS, int NC> inline constexpr bool ValidKernelParams() noexcept {
  // NS = nspread (kernel width), NC = # poly coeffs in Horner evaluator.
  // Defines the compiled range of NC for each NS, in spreadinterp.
  // Other instantiations can be
  // compiled away at call sites using if constexpr to reduce binary size.
  // Barbone Dec 2025.
  // AHB changed to use the above two expressions, but needs checking if compile-time ok
  return (NC >= min_nc_given_ns(NS)) && (NC <= max_nc_given_ns(NS));
}

} // namespace finufft::kernel

namespace finufft::common {

// Fine-grid length one dimension gets at this sigma, mirroring set_nf_type12
// (makeplan.hpp): ceil(sigma*n_modes), floored at 2*ns, rounded up to the next even
// 2,3,5-smooth number. set_nf_type12 owns the BIGINT/MAX_NF version; this is the
// read-only mirror for the upsampfac selector.
BIGINT fine_grid_len(double sigma, double n_modes, int ns);

// Type-3 fine grid for one dimension, given the source/freq interval half-widths X,S.
// Clamps so X*S>=1 (handling X and/or S == 0), sizes the upsampled grid nf (floored at
// 2*ns, then next235-rounded unless it already exceeds max_nf, the allocation guard),
// and returns spacing h=2pi/nf and x-rescale gam=nf/(2*sigma*S). All math in double;
// callers narrow h/gam to their precision. Single source of truth for set_nhg_type3 on
// CPU (setpts.hpp) and GPU (cuda/makeplan.cu) and the type-3 upsampfac cost model
// (src/heuristics.cpp). Barnett 6/12/17 logic; extracted Barbone 6/26.
// Returns the tuple (nf, h, gam); destructure with structured bindings.
std::tuple<BIGINT, double, double> nhg_type3(double sigma, double X, double S, int ns,
                                             BIGINT max_nf);

} // namespace finufft::common
