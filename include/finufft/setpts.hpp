#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include <cassert>

#include <finufft/plan.hpp>
#include <finufft/spreadinterp.hpp>
#include <finufft/utils.hpp>
#include <finufft_common/kernel.h>

// ---------- local helpers for setpts: --------

// Throws FINUFFT_ERR_EPS_TOO_SMALL when the summed per-dimension estimates exceed
// tol. The three estimates (rounding at N_d, aliasing and deconvolution at ns_d) are
// summed over dimensions (the safe combination) on the per-dimension sigmas: the
// realized ratio nf_d / N_d in auto mode, the forced upsampfac (one value in every
// dim) otherwise. perdim_estimates_ok (kernel.cpp) is the same test the selector
// uses, so the two cannot disagree. opts.allow_eps_too_small warns instead.
template<typename TF> void FINUFFT_PLAN_T<TF>::check_sigma(double /*mode_diagonal*/) {
  using namespace finufft::common;
  using namespace finufft::kernel;
  const double tol = m.tol;
  std::array<double, 3> N{}, sigma_d{};
  std::array<int, 3> ns_d{};
  for (int idim = 0; idim < dim; ++idim) {
    N[idim]       = (double)mstu[idim];
    // A per-dim plan is tested at the realized ratios nf_d / N_d, the same values the
    // selector prices. A uniform plan keeps the one searched or forced sigma, so the
    // refusal contract is the develop one (upsampfac too low refuses). spreadinterponly
    // has no fine grid ratio and keeps spopts in every dim.
    sigma_d[idim] = (m.perdim_kernels && !opts.spreadinterponly && mstu[idim] > 0)
                        ? (double)m.nfdim[idim] / (double)mstu[idim]
                        : m.spopts.upsampfac;
    ns_d[idim]    = m.perdim_kernels ? m.spopts.nspread_dim[idim] : m.spopts.nspread;
  }
  std::array<double, 3> sums;
  const bool feasible = perdim_estimates_ok(tol, dim, type, std::is_same_v<TF, float>, N,
                                            sigma_d, ns_d, &sums);
  if (feasible) return;
  const bool do_throw = !opts.allow_eps_too_small; // opt-in wins
  // an error always speaks; the warning obeys opts.showwarn, as every other warning does
  if (!do_throw && !opts.showwarn) return;
  const char *level = do_throw ? "error" : "warning";
  if (sums[0] > tol)
    // the rounding floor dominates: no upsampfac can meet the tolerance
    fprintf(stderr, "%s %s: tol=%.3g is below the rounding floor %.3g\n", __func__, level,
            tol, sums[0]);
  else {
    // a larger upsampfac fixes the failing aliasing or deconvolution sum
    const bool aliasing_fails = sums[1] > tol;
    const double est          = aliasing_fails ? sums[1] : sums[2];
    fprintf(stderr,
            "%s %s: the summed %s estimate %.3g exceeds tol=%.3g at ns=(%d,%d,%d); "
            "upsampfac too low\n",
            __func__, level, aliasing_fails ? "aliasing" : "deconvolution", est, tol,
            ns_d[0], ns_d[1], ns_d[2]);
  }
  if (do_throw) throw finufft::exception(FINUFFT_ERR_EPS_TOO_SMALL);
}

// ---------- local math routines for type-3 setpts: --------

template<typename TF>
void FINUFFT_PLAN_T<TF>::set_nhg_type3(int idim, TF S, TF X)
/* sets nfdim[idim], t3P.h[idim], and t3P.gam[idim], for type 3 only.
   Inputs:
   idim - which dimension (0,1,2)
   X and S are the xj and sk interval half-widths respectively.
   Reads opts and spopts from the plan.
   Outputs written to plan members:
   nfdim[idim] - size of upsampled grid for this dimension.
   t3P.h[idim] - grid spacing = 2pi/nf
   t3P.gam[idim] - x rescale factor, ie x'_j = x_j/gam (modulo shifts).
   Barnett 2/13/17. Caught inf/nan 3/14/17. io int types changed 3/28/17
   New logic 6/12/17
   Previous args (opts, spopts) are now plan members; outputs (nf, h, gam) are
   written directly to plan members nfdim, t3P.h, t3P.gam.
   Converted to class member, Barbone 2/24/26.
   Body delegated to the shared finufft::common::nhg_type3, Barbone 6/26.
*/
{
  const auto [nf, h, gam] =
      finufft::common::nhg_type3(opts.upsampfac, X, S, m.spopts.nspread, MAX_NF);
  m.nfdim[idim] = nf;
  m.t3P.h[idim] = TF(h);   // upsampled grid spacing
  m.t3P.gam[idim] = TF(gam); // x rescale factor, ie x'_j = x_j/gam (modulo shifts)
}

// --------- setpts user guru interface driver ----------
template<typename TF>
int FINUFFT_PLAN_T<TF>::setpts(BIGINT nj, const TF *xj, const TF *yj, const TF *zj,
                               BIGINT nk, const TF *s, const TF *t, const TF *u) {
  using namespace finufft::utils;
  // Method function to set NU points and do precomputations. Barnett 2020.
  // Barbone (3/4/26): removed warning_code_ plumbing (eps-too-small now throws).
  // See ../docs/cguru.doc for current documentation.
  int d = dim;       // abbrev for spatial dim
  CNTime timer;
  timer.start();
  m.nj = nj; // the user only now chooses how many NU (x,y,z) pts
  if (nj < 0) {
    fprintf(stderr, "[%s] nj (%lld) cannot be negative!\n", __func__, (long long)nj);
    throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  } else if (nj > MAX_NU_PTS) {
    fprintf(stderr, "[%s] nj (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nj);
    throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
  }

  if (type != 3) { // ------------------ TYPE 1,2 SETPTS -------------------
                   // (all we can do is check and maybe bin-sort the NU pts)
    // If upsampfac is not locked by user (auto mode), choose or update it now.
    if (!upsamp_locked) {
      double upsampfac =
          best_sigma_for_tol(type, {(double)mstu[0], (double)mstu[1], (double)mstu[2]});
      // no sigma meets tol: plan at the widest sigma, check_sigma reports it
      if (!std::isfinite(upsampfac)) upsampfac = finufft::common::MAX_AUTO_UPSAMPFAC;
      // Re-plan if this is the first call (upsampfac==0) or if upsampfac changed
      if (upsampfac != opts.upsampfac) {
        opts.upsampfac = upsampfac;
        if (opts.debug)
          printf("[setpts] selected best upsampfac=%.3g (nj=%lld)\n", opts.upsampfac,
                 (long long)nj);
        setup_spreadinterp(); // throws on error
        precompute_horner_coeffs();
        // Perform the planning steps (first call or re-plan on a new upsampfac).
        init_grid_kerFT_FFT();       // throws on error
      }
    }

    double mode_diagonal = 0.0;
    for (int idim = 0; idim < dim; ++idim)
      mode_diagonal += (double)mstu[idim] * mstu[idim];
    // Throws FINUFFT_ERR_EPS_TOO_SMALL if tol is not achievable
    check_sigma(std::sqrt(mode_diagonal));

    m.XYZ   = {xj, yj, zj}; // plan must keep pointers to user's fixed NU pts
    // Invariant: m.padded_ns_dim must equal the runtime mirror of
    // KernelBufferLayout<TF, NS>::stride. Caught here if any path forgot to
    // call precompute_horner_coeffs (or if the trait diverges from the runtime
    // formula).
    for (int idim = 0; idim < dim; ++idim)
      assert(m.padded_ns_dim[idim] ==
             finufft::spreadinterp::kernel_buffer_stride_runtime<TF>(
                 m.spopts.nspread_dim[idim]));
    spreadcheck();          // throws on error
    timer.restart();
    m.sortIndices.resize(nj);
    indexSort();
    if (opts.debug)
      printf("[%s] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)m.didSort,
             timer.elapsedsec());

  } else { // ------------------------- TYPE 3 SETPTS -----------------------
           // (here we can precompute pre/post-phase factors and plan the t2)

    std::array<const TF *, 3> XYZ_in{xj, yj, zj};
    std::array<const TF *, 3> STU_in{s, t, u};
    if (nk < 0) {
      fprintf(stderr, "[%s] nk (%lld) cannot be negative!\n", __func__, (long long)nk);
      throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
    } else if (nk > MAX_NU_PTS) {
      fprintf(stderr, "[%s] nk (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nk);
      throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
    }
    m.nk = nk; // user set # targ freq pts
    m.STU = {s, t, u};

    // pick x, s interval half-widths & shifts/centers (all sigma-independent, so
    // computed before choosing the upsampfac)...
    std::array<TF, 3> S = {0, 0, 0};
    if (opts.debug) printf("\tM=%lld N=%lld\n", (long long)nj, (long long)nk);
    for (int idim = 0; idim < dim; ++idim) {
      // special treatment of nj==0/nk==0, because arraywidcen doesn't work in that case
      if (nj>0)
        arraywidcen(nj, XYZ_in[idim], &(m.t3P.X[idim]), &(m.t3P.C[idim]));
      else
        m.t3P.X[idim] = m.t3P.C[idim] = 0;
      if (nk>0)
        arraywidcen(nk, STU_in[idim], &S[idim], &(m.t3P.D[idim]));
      else
        S[idim] = m.t3P.D[idim] = 0;
    }

    // For type 3 with deferred upsampfac (not locked by user), pick & persist
    // sigma3 now, so set_nhg_type3 and the inner t2 plan use it.
    if (!upsamp_locked) {
      double upsampfac =
          best_sigma_for_tol(type, {(double)S[0], (double)S[1], (double)S[2]});
      // no sigma meets tol: plan at the widest sigma
      if (!std::isfinite(upsampfac)) upsampfac = finufft::common::MAX_AUTO_UPSAMPFAC;
      if (upsampfac != opts.upsampfac) {
        opts.upsampfac = upsampfac;
        if (opts.debug > 1)
          printf("[setpts t3] selected upsampfac=%.2f (persisted)\n", opts.upsampfac);
        setup_spreadinterp(); // throws on error
        precompute_horner_coeffs();
      }
    }
    // A user-locked type-3 upsampfac is used as given: type 3 has no check_sigma.

    // ...then # fine grid pts (nf) per dim, now that sigma3 (and hence ns) is fixed.
    for (int idim = 0; idim < dim; ++idim) {
      set_nhg_type3(idim, S[idim], m.t3P.X[idim]); // applies twist i)
      if (opts.debug) // report on choices of shifts, centers, etc...
        printf("\tX%d=%.3g C%d=%.3g S%d=%.3g D%d=%.3g gam%d=%g nf%d=%lld h%d=%.3g\t\n",
               idim, m.t3P.X[idim], idim, m.t3P.C[idim], idim, S[idim], idim,
               m.t3P.D[idim], idim, m.t3P.gam[idim], idim, (long long)m.nfdim[idim],
               idim, m.t3P.h[idim]);
    }
    for (int idim = dim; idim < 3; ++idim)
      m.t3P.C[idim] = m.t3P.D[idim] = 0.0; // their defaults if dim 2 unused, etc

    if (nf() * batchSize > MAX_NF) {
      fprintf(stderr,
              "[%s t3] fwBatch would be bigger than MAX_NF, not attempting memory "
              "allocation!\n",
              __func__);
      throw finufft::exception(FINUFFT_ERR_MAXNALLOC);
    }

    // alloc rescaled NU src pts x'_j (in X etc), rescaled NU targ pts s'_k ...
    // We do this by resizing Xp, Yp, and Zp, and pointing X, Y, Z to their data;
    // this avoids any need for explicit cleanup.
    for (int idim = 0; idim < dim; ++idim) {
      m.XYZp[idim].resize(nj);
      m.XYZ[idim] = m.XYZp[idim].data();
      m.STUp[idim].resize(nk);
    }

    // always shift as use gam to rescale x_j to x'_j, etc (twist iii)...
    std::array<TF, 3> ig = {0, 0, 0};
    for (int idim = 0; idim < dim; ++idim) ig[idim] = 1.0 / m.t3P.gam[idim];
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT j = 0; j < nj; ++j) {
      for (int idim = 0; idim < dim; ++idim)
        m.XYZp[idim][j] = (XYZ_in[idim][j] - m.t3P.C[idim]) * ig[idim]; // rescale x_j
    }

    // set up prephase array...
    TF isign = (fftSign >= 0) ? 1 : -1;
    m.prephase.resize(nj);
    if (m.t3P.D[0] != 0.0 || m.t3P.D[1] != 0.0 || m.t3P.D[2] != 0.0) {
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
      for (BIGINT j = 0; j < nj; ++j) { // ... loop over src NU locs
        TF phase = 0;
        for (int idim = 0; idim < dim; ++idim) phase += m.t3P.D[idim] * XYZ_in[idim][j];
        m.prephase[j] = std::polar(TF(1), isign * phase); // Euler
      }
    } else
      for (BIGINT j = 0; j < nj; ++j)
        m.prephase[j] = {1.0, 0.0}; // *** or keep flag so no mult in exec??

    // create a 1D phihat evaluator
    Kernel_onedim_FT onedim_phihat(*this);

    // (old STEP 3a) Compute deconvolution post-factors array (per targ pt)...
    // (exploits that FT separates because kernel is prod of 1D funcs)
    m.deconv.resize(nk);
    // C can be nan or inf if M=0, no input NU pts
    bool Cfinite = std::isfinite(m.t3P.C[0]) && std::isfinite(m.t3P.C[1]) &&
                   std::isfinite(m.t3P.C[2]);
    bool Cnonzero = m.t3P.C[0] != 0.0 || m.t3P.C[1] != 0.0 || m.t3P.C[2] != 0.0;
    bool do_phase = Cfinite && Cnonzero;
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
    for (BIGINT k = 0; k < nk; ++k) { // .... loop over NU targ freqs
      TF phiHat = 1;
      TF phase  = 0;
      for (int idim = 0; idim < dim; ++idim) {
        auto tSTUin = STU_in[idim][k];
        // rescale the target s_k etc to s'_k etc...
        auto tSTUp =
            m.t3P.h[idim] * m.t3P.gam[idim] * (tSTUin - m.t3P.D[idim]); // |s'_k| < pi/R
        phiHat *= onedim_phihat(tSTUp);
        if (do_phase) phase += (tSTUin - m.t3P.D[idim]) * m.t3P.C[idim];
        m.STUp[idim][k] = tSTUp;
      }
      m.deconv[k] =
          do_phase ? std::polar(TF(1) / phiHat, isign * phase) : TF(1) / phiHat;
    }
    if (opts.debug)
      printf("[%s t3] phase & deconv factors:\t%.3g s\n", __func__, timer.elapsedsec());

    // Set up sort for spreading Cp (from primed NU src pts X, Y, Z) to fw...
    timer.restart();
    m.sortIndices.resize(nj);
    m.spopts.spread_direction = 1;
    indexSort();
    if (opts.debug)
      printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)m.didSort,
             timer.elapsedsec());

    // Plan and setpts once, for the (repeated) inner type 2 finufft call...
    timer.restart();
    BIGINT t2nmodes[]   = {m.nfdim[0], m.nfdim[1], m.nfdim[2]}; // t2's input actually fw
    finufft_opts t2opts = opts;                           // deep copy, since not ptrs
    t2opts.modeord      = 0;                              // needed for correct t3!
    t2opts.debug        = std::max(0, opts.debug - 1);    // don't print as much detail
    t2opts.spread_debug = std::max(0, opts.spread_debug - 1);
    t2opts.showwarn     = 0;                              // so don't see warnings 2x
    if (!upsamp_locked)
      t2opts.upsampfac = 0.0; // if the upsampfac was auto, let inner t2 pick it again
    // (...could vary other t2opts here?)
    // MR: temporary hack, until we have figured out the C++ interface.
    FINUFFT_PLAN_T<TF> *tmpplan;
    finufft_makeplan_t<TF>(2, d, t2nmodes, fftSign, batchSize, m.tol, &tmpplan,
                           &t2opts); // throws on error
    // Use a non-const unique_ptr to ensure cleanup if setpts throws, then
    // transfer to the const unique_ptr member.
    std::unique_ptr<FINUFFT_PLAN_T<TF>> guard(tmpplan);
    tmpplan->setpts(nk, m.STUp[0].data(), m.STUp[1].data(), m.STUp[2].data(), 0,
                    nullptr, nullptr,
                    nullptr); // note nk = # output points (not nj); throws on error
    m.innerT2plan = std::move(guard);
    if (opts.debug) {
      // The inner type-2 plan is the one that goes per dim. It prints its own set
      // only when its debug level is >= 1, that is opts.debug > 1 here.
      const auto &ipl = m.innerT2plan;
      const auto &im  = ipl->m;
      const std::array<int, 3> ns_set{im.spopts.nspread_dim[0], im.spopts.nspread_dim[1],
                                      im.spopts.nspread_dim[2]};
      const std::array<BIGINT, 3> nf_set{im.nfdim[0], im.nfdim[1], im.nfdim[2]};
      std::array<double, 3> ops{};
      const double cost = finufft::kernel::perdim_set_cost(
          std::is_same_v<TF, float>, (double)nk, ns_set, nf_set, d, opts.nthreads, &ops);
      for (int idim = 0; idim < d; ++idim)
        printf("[%s] perdim inner d%d: sigma_d=%.4g ns_d=%d nf_d=%lld beta_d=%.4g "
               "cost_spread_d=%.4g cost_fft_d=%lld\n",
               __func__, idim,
               im.perdim_kernels ? (double)im.nfdim[idim] / (double)ipl->mstu[idim]
                                 : im.spopts.upsampfac,
               ns_set[idim], (long long)nf_set[idim], im.spopts.beta_dim[idim], ops[idim],
               (long long)nf_set[idim]);
      printf("[%s] perdim inner cost=%.4g\n", __func__, cost);
      printf("[%s t3] inner t2 plan & setpts: \t%.3g s\n", __func__, timer.elapsedsec());
    }
  }
  return 0;
}
