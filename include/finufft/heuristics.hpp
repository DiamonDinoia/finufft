#pragma once

// Complexity-based upsampfac (sigma) selection for the type-1/2 and type-3 setpts
// paths: cost-model primitives + a generic minimizer. Home for the tuning constants.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <finufft/simd.hpp>            // get_padding<TF>, get_padded_simd_width<TF>
#include <finufft_common/constants.h>   // PI, MAX_CHECK_SIGMA, MIN/MAX_AUTO_UPSAMPFAC
#include <finufft_common/kernel.h>      // ns formulas, feasibility, fine_grid_len
#include <finufft_common/spread_opts.h> // finufft_spread_opts

namespace finufft::heuristics {

// The cost-minimizing feasible upsampfac and its predicted cost.
struct sigma_info {
  double sigma; // cost-minimizing feasible upsampfac
  double cost;  // its predicted cost (same flop units across all transform types)
};

// --- cost primitives (all in the same arbitrary-but-consistent flop units) ---

// Inner spread work per NU point: the 2*ns-wide complex row is vectorized, so it
// runs as ceil(2*ns / simd_width) SIMD ops, not 2*ns scalars. Precision/ISA-adaptive
// (wider lanes in f32, fewer ops); measured ns-per-point scales as this x ns^(dim-1),
// so the inner row must be counted in vectors — counting padded scalars (~2*ns)
// over-weights wide kernels and biases the pick toward too-large sigma.
template<typename TF> inline double spread_row(int ns) {
  const double padded_2ns = 2.0 * ns + finufft::spreadinterp::get_padding<TF>(2 * ns);
  const double simd = (double)finufft::spreadinterp::get_padded_simd_width<TF>(2 * ns);
  return std::max(padded_2ns / simd, 1.0);
}

// spread_row x ns^(dim-1) per-point transverse factor.
template<typename TF> inline double spread_cost(double npts, int ns, int dim) {
  double outer = 1.0; // ns^(dim-1)
  for (int idim = 1; idim < dim; ++idim) outer *= (double)ns;
  return npts * spread_row<TF>(ns) * outer;
}

// FFT cost weight per fine-grid point per log2 unit, relative to one spread flop.
// Rises with threads: spreading parallelizes ~linearly, the FFT sublinearly.
inline double c_fft(int nthreads) {
  // calibrated on ccmlin075 (AVX-512, FFTW); see devel/calibrate_upsampfac.cpp.
  // Centred in a wide flat plateau (C in ~[1.2,3] all pick within 3% of optimum on
  // 1D/2D/3D, f32/f64, single/multi-thread); one shared set covers FFTW and DUCC0.
  constexpr double C_FFT_BASE = 2.0;
  constexpr double K_FFT_THREAD = 0.50;
  return C_FFT_BASE * std::pow((double)std::max(1, nthreads), K_FFT_THREAD);
}

// FFT cost c*G*log2(G), G = fine-grid volume.
inline double fft_cost(double c, const double *nmodes, double sigma, int ns, int dim) {
  double G = 1.0;
  for (int idim = 0; idim < dim; ++idim)
    G *= (double)finufft::common::fine_grid_len(sigma, nmodes[idim], ns);
  return c * G * std::log2(G);
}

// --- candidate enumeration ---

// Kernel width ns the plan would actually use at this (tol, sigma): the theoretical
// width clamped to the compiled/feasible range, exactly as setup_spreadinterp picks it.
// Lets the cost model score each candidate sigma at its real ns.
template<typename TF>
inline int kernel_width_at(double tol, int dim, int type, double sigma) {
  finufft_spread_opts so{};
  so.kerformula = 0;
  so.upsampfac = sigma;
  return finufft::kernel::clamp_kernel_ns<TF>(
      finufft::kernel::theoretical_kernel_ns(tol, dim, type, 0, so), sigma);
}

// Returns the feasible upsampfac (sigma) that minimizes cost, and its predicted cost.
//   cost(sigma, ns): caller-supplied score of a candidate in consistent flop units.
//   maxN:            largest mode count over dims (1 for type 3).
// If tol is unachievable, returns the largest sigma and the plan pipeline reports it.
template<typename TF, class Cost>
sigma_info minimize(double tol, int dim, int type, double maxN, Cost &&cost) {
  using namespace finufft::common;
  constexpr double eps_mach = std::numeric_limits<TF>::epsilon();
  constexpr bool is_float = std::is_same_v<TF, float>;
  const auto feasible = [&](double sigma) {
    return upsampfac_feasible(sigma, tol, dim, type, eps_mach, MAX_NSPREAD<TF>, is_float,
                              maxN);
  };
  const double sigma_min =
      analytic_upsampfac(tol, dim, type, eps_mach, MAX_NSPREAD<TF>, is_float, maxN);
  const int ns_min = kernel_width_at<TF>(tol, dim, type, sigma_min);
  sigma_info best{sigma_min, cost(sigma_min, ns_min)};
  if (!feasible(sigma_min)) return best; // tol unachievable; pipeline reports
  for (int ns_t = ns_min - 1;
       ns_t >= kernel_width_at<TF>(tol, dim, type, MAX_AUTO_UPSAMPFAC); --ns_t) {
    const double s = std::clamp(smallest_sigma_for_ns(tol, dim, type, ns_t), sigma_min,
                                MAX_AUTO_UPSAMPFAC);
    if (!feasible(s)) continue;
    const double c = cost(s, kernel_width_at<TF>(tol, dim, type, s));
    if (c < best.cost) best = {s, c};
  }
  return best;
}

// --- transform-specific selectors ---

// Type 1/2: spread of npts points + FFT. Also the inner type-2 cost for best_type3.
template<typename TF>
sigma_info best_type12(double tol, int dim, int type, int nthreads, const double *nmodes,
                       double npts) {
  const double c = c_fft(nthreads);
  const double maxN = *std::max_element(nmodes, nmodes + dim);
  const auto cost = [&](double sigma, int ns) {
    return spread_cost<TF>(npts, ns, dim) + fft_cost(c, nmodes, sigma, ns, dim);
  };
  return minimize<TF>(tol, dim, type, maxN, cost);
}

// Fine-grid length set_nhg_type3 builds for one dim at this sigma3, from the
// source/target interval half-widths X,S. Thin wrapper over the shared finufft::common
// helper; the cost model only needs the length (no plan to mutate) and ignores the
// MAX_NF allocation guard, so pass max_nf=BIGINT max to always next235-round.
inline double type3_fine_grid_len(double sigma3, double X, double S, int ns3) {
  return (double)std::get<0>(
      finufft::common::nhg_type3(sigma3, X, S, ns3, std::numeric_limits<BIGINT>::max()));
}

// Returns the cost-minimizing upsampfac (sigma3) for a type-3 transform.
// Type 3: outer spread of nj sources (width ns3) onto a fine grid of size
// nfdim(sigma3) ∝ sigma3, then a full inner type-2 NUFFT evaluating nk targets on
// that grid (its own sigma2 re-optimized via best_type12). Minimizing over feasible
// sigma3 trades the outer spread (cheaper at large sigma3, narrow kernel) against the
// inner t2 (cheaper at small sigma3, smaller grid). X,S are the per-dim source/target
// interval half-widths (from arraywidcen).
template<typename TF>
double best_type3(double tol, int dim, int nthreads, double nj, const double *X,
                  const double *S, double nk) {
  const auto cost = [&](double sigma3, int ns3) {
    std::array<double, 3> nmodes{1.0, 1.0, 1.0};
    for (int idim = 0; idim < dim; ++idim)
      nmodes[idim] = type3_fine_grid_len(sigma3, X[idim], S[idim], ns3);
    const double inner = best_type12<TF>(tol, dim, 2, nthreads, nmodes.data(), nk).cost;
    return spread_cost<TF>(nj, ns3, dim) + inner;
  };
  return minimize<TF>(tol, dim, /*type=*/3, /*maxN=*/1.0, cost).sigma;
}

/* Number of subproblems (nb) for dir=1 blocked spreading, or 0 to leave the
  decomposition to spopts.max_subproblem_size.

  Each subproblem pays for a padded subgrid it must zero, write and add back,
  costing "overhead" cells per NU point (measured 0.02 - 40 across configs).
  Bigger subproblems amortize that, so the target size scales with overhead.

  overhead is estimated from the *measured* bin occupancy of the sort, not from
  the mean density nj/N: clustered inputs (MRI, radio astronomy) pack many points
  into few bins, and their true overhead is up to 60x below what mean density
  predicts, which flips the answer from "large" to "small".

  Not cache capacity. L2/L3 miss counts are flat across this parameter (2D at 48
  threads: 11.8x cycles between two settings at equal instructions and equal
  misses), and the best 3D setting runs a subgrid several times the size of L3.
  A cache-budget objective was measured on four microarchitectures and moved away
  from the optimum on all of them. What the level below L2 does is untested; in
  2D the slow settings show an IPC-0.04 serialization signature that is not work,
  not coherency and not a miss at any measured level, and is still unexplained.

  The optimum is machine-dependent in *direction* (3D uniform wants >=625k on
  Sapphire Rapids, 10k-20k on Zen2), so the ramp deliberately sits in the flat
  interior of the response curve rather than at any node's argmin. Three further
  retunes were each measured on one node and contradicted in *sign* on another, so
  all three are declined: a smaller 2D SP_REF (Zen4 wants 200-1000, which costs
  +19% to +75% on Meteor Lake, itself within 5.5% of its 2D argmin at 30000); an
  nthreads term (the two nodes disagree on which way the optimum moves in each
  dim); and a grid-volume term for the sparse regime N >> npts (the clustered
  sparse cell that regresses on Zen4 has its Meteor Lake argmin at the *smallest*
  sp measured, so the correction has the wrong sign here). The sparse regime is
  instead handled by SPAN_MAX below, which bounds the ramp rather than re-scaling it.

  Also declined, measured: OVERHEAD_REF at 2 rather than 5. It does fix the two 2D
  cells that ramp down when their argmin is above SP_REF, but the same pivot destroys
  the concentrated-input win it exists to produce (sparse clustered cells fall from
  1.40x and 1.27x master to 0.88x and 0.86x), for a 4.7% worse geomean overall.

  Known residual losses on Meteor Lake, both at a floor rather than at the ramp, and
  both needing a discriminator that would have to be non-monotonic in occupancy (one
  wants a smaller sp at 136 pts/bin than another does at 25 pts/bin), so no single
  bound corrects them: 3D dense clustered, where the "never split below one bin"
  floor elects 2.2e5 against an argmin of 3e4, landing 17.6% above it where master
  lands 4.6% above; and 3D sparse clustered, where the ramp elects 1.2e5 against an
  argmin of 1e4, landing 59.6% above it where master lands 44.8% above.

  Validated in both precisions on this node. f32 is not a degenerate path: halving ns
  shrinks the overhead estimate, so f32 elects consistently more subproblems (2D
  uniform 752 -> 944), and the measured gain is larger than f64's, +12.7% geomean
  against a 1.008 identity control over 26 changed cells.

  Inputs: dim, ns (kernel width), npts (=nj), n_occupied_bins (bins holding >=1
  pt, 0 if unsorted), bin_size[dim] (sort bin edge lengths), nthreads.
  Returns nb >= 1, or 0 when there is no occupancy measure to work from.
*/
inline BIGINT n_subproblems(int dim, int ns, double npts, double n_occupied_bins,
                            const double *bin_size, int nthreads) {
  // target subproblem size, and the overhead level at which amortizing subgrid
  // cells starts to pay for a larger subproblem. SP_MAX bounds per-thread scratch
  // RAM. Fitted on f64 geometry; at f32 (ns 3-4 rather than 7) the measured effect
  // is roughly neutral, +0.02% to +4% geomean depending on the node, so these are
  // not f32-calibrated.
  constexpr double SP_REF = 30000.0, OVERHEAD_REF = 5.0, SP_MAX = 1e6;
  if (dim == 1 || n_occupied_bins <= 0) return 0; // caller keeps its own cap
  double cells = 1.0; // padded subgrid cells belonging to one sort bin
  for (int d = 0; d < dim; ++d) cells *= bin_size[d] + ns;
  const double overhead = cells * n_occupied_bins / npts; // ~subgrid cells per pt
  // Above OVERHEAD_REF, amortize: bigger subproblems spread the subgrid cost over
  // more points. Below it the points are concentrated, so their subgrids are small
  // and splitting is cheap - keep ramping down, but as a square root, since the
  // linear ramp overshoots the measured optimum by ~3x. sqrt(r) <= r for r >= 1, so
  // this is a bit-identical no-op outside the concentrated regime.
  // Never split below one sort bin's worth of points: such a subproblem has a
  // bounding box no smaller than that bin's, so it adds redundant subgrid work and
  // buys no locality. This bounds the ramp where measurement stops (r >= 0.1).
  const double r = overhead / OVERHEAD_REF;
  const double occ = npts / n_occupied_bins; // points in one sort bin
  // The up-ramp saturates: amortizing the padded subgrid only pays while the halo is a
  // real fraction of the subgrid, and a subproblem takes consecutive points in bin
  // order, so once it spans SPAN_MAX bins the halo is already negligible and growing
  // further only costs locality. Measured on the sparse regime N >> npts, where the
  // unbounded ramp asked for 12x the argmin (2D, npts=2e6, N=2e7, mildly clustered:
  // elected 1.25e5 against an argmin of 1e4, costing 13-22%); the bound turns that cell
  // from 0.94x master into 1.05x in both precisions. SPAN_MAX is the loosest value that
  // still corrects it, so the 3D cell whose argmin genuinely is 1e5 keeps its election.
  constexpr double SPAN_MAX = 5000.0;
  const double sp =
      std::max(std::min(SP_REF * std::max(r, std::sqrt(r)), SPAN_MAX * occ), occ);
  const double nthr = std::max(nthreads, 1);
  // nb below nthreads would starve schedule(dynamic,1), and nb just above a
  // multiple of nthreads runs a whole extra round for a handful of stragglers, so
  // elect a whole multiple k*nthreads. k == 1 means "one round", which is why no
  // floor on the subproblem size is needed; the second term keeps SP_MAX hard
  // after snapping, which is why sp itself needs no upper clamp.
  const double k = std::max(std::max(1.0, std::round(std::ceil(npts / sp) / nthr)),
                            std::ceil(npts / (SP_MAX * nthr)));
  return (BIGINT)(k * nthr); // caller clips to npts when it has fewer points than that
}

} // namespace finufft::heuristics
