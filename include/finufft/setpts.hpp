#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

#include <finufft/heuristics.hpp>
#include <finufft/plan.hpp>
#include <finufft/simd.hpp>
#include <finufft/spreadinterp.hpp>
#include <finufft/utils.hpp>
#include <finufft_common/trig.hpp>

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
*/
{
  using namespace finufft::common;
  using namespace finufft::utils;
  int nss = m.spopts.nspread + 1; // since ns may be odd
  TF Xsafe = X, Ssafe = S;       // may be tweaked locally
  if (X == 0.0)                 // logic ensures XS>=1, handle X=0 a/o S=0
    if (S == 0.0) {
      Xsafe = 1.0;
      Ssafe = 1.0;
    } else
      Xsafe = std::max(Xsafe, 1 / S);
  else
    Ssafe = std::max(Ssafe, 1 / X);
  // use the safe X and S...
  auto nfd = TF(2.0 * opts.upsampfac * Ssafe * Xsafe / PI + nss);
  if (!std::isfinite(nfd)) nfd = 0.0;
  m.nfdim[idim] = (BIGINT)nfd;
  // catch too small nf, and nan or +-inf, otherwise spread fails...
  if (m.nfdim[idim] < 2 * m.spopts.nspread) m.nfdim[idim] = 2 * m.spopts.nspread;
  if (m.nfdim[idim] < MAX_NF)                     // otherwise will fail
    m.nfdim[idim] = next235even(m.nfdim[idim]);   // expensive at huge nf
  m.t3P.h[idim]   = TF(2.0 * PI / m.nfdim[idim]); // upsampled grid spacing
  m.t3P.gam[idim] = TF(m.nfdim[idim] / (2.0 * opts.upsampfac * Ssafe)); // x scale fac
}

// --------- setpts user guru interface driver ----------

template<typename TF>
int FINUFFT_PLAN_T<TF>::setpts(BIGINT nj, const TF *xj, const TF *yj, const TF *zj,
                               BIGINT nk, const TF *s, const TF *t, const TF *u) {
  using namespace finufft::utils;
  using namespace finufft::heuristics;
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
    // If upsampfac is not locked by user (auto mode), choose or update it now
    // based on the actual density nj/N(). Re-plan if density changed significantly.
    if (!upsamp_locked) {
      double density   = double(nj) / double(N());
      double upsampfac = bestUpsamplingFactor<TF>(opts.nthreads, density, dim, type, m.tol);
      // Re-plan if this is the first call (upsampfac==0) or if upsampfac changed
      if (upsampfac != opts.upsampfac) {
        opts.upsampfac = upsampfac;
        if (opts.debug)
          printf("[setpts] selected best upsampfac=%.3g (density=%.3g, nj=%lld)\n",
                 opts.upsampfac, density, (long long)nj);
        setup_spreadinterp(); // throws on error
        precompute_horner_coeffs();
        // Perform the planning steps (first call or re-plan due to density change).
        init_grid_kerFT_FFT();       // throws on error
      }
    }

    m.XYZ   = {xj, yj, zj}; // plan must keep pointers to user's fixed NU pts
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
    auto &tc = m.t3cache;
    if (nk < 0) {
      fprintf(stderr, "[%s] nk (%lld) cannot be negative!\n", __func__, (long long)nk);
      throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
    } else if (nk > MAX_NU_PTS) {
      fprintf(stderr, "[%s] nk (%lld) exceeds MAX_NU_PTS\n", __func__, (long long)nk);
      throw finufft::exception(FINUFFT_ERR_NUM_NU_PTS_INVALID);
    }
    m.nk = nk; // user set # targ freq pts
    m.STU = {s, t, u};

    // For type 3 with deferred upsampfac (not locked by user), pick and persist
    // an upsamp now using density=1.0 so that subsequent steps (set_nhg_type3 etc.)
    // have a concrete upsampfac to work with. This choice is persisted so inner
    // type-2 plans will inherit it.
    double upsampfac = bestUpsamplingFactor<TF>(opts.nthreads, 1.0, dim, type, m.tol);
    if (!upsamp_locked && (upsampfac != opts.upsampfac)) {
      opts.upsampfac = upsampfac;
      if (opts.debug)
        printf("[setpts t3] selected upsampfac=%.2f (density=1 used; persisted)\n",
               opts.upsampfac);
      setup_spreadinterp(); // throws on error
      precompute_horner_coeffs();
    }

    // pick x, s intervals & shifts & # fine grid pts (nf) in each dim...
    std::array<TF, 3> S = {0, 0, 0};
    std::array<bool, 3> source_same = {true, true, true};
    std::array<bool, 3> target_same = {true, true, true};
    if (opts.debug) printf("\tM=%lld N=%lld\n", (long long)nj, (long long)nk);
    for (int idim = 0; idim < dim; ++idim) {
      source_same[idim] =
          arraywidcen(nj, XYZ_in[idim], tc.XYZ[idim], &(m.t3P.X[idim]), &(m.t3P.C[idim]));
      target_same[idim] =
          arraywidcen(nk, STU_in[idim], tc.STU[idim], &S[idim], &(m.t3P.D[idim]));
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

    // --- Build cache keys for this call ---
    bool target_coords_same = tc.target_valid;
    bool source_coords_same = tc.source_valid;
    for (int idim = 0; idim < dim; ++idim) {
      target_coords_same = target_coords_same && target_same[idim];
      source_coords_same = source_coords_same && source_same[idim];
    }
    typename M::Type3Cache::TargetKey cur_tkey{nk, m.nfdim, S, m.t3P.D};
    typename M::Type3Cache::SourceKey cur_skey{nj, m.t3P.C, m.t3P.gam, m.t3P.D};

    bool tgt_hit = target_coords_same && tc.targets_match(cur_tkey);
    bool src_hit = source_coords_same && tc.sources_match(cur_skey);

    if (opts.debug)
      printf("[%s t3] cache lookup: targets %s, sources %s\n", __func__,
             tgt_hit ? "HIT" : "miss", src_hit ? "HIT" : "miss");
    if (opts.debug > 1) {
      printf("[%s t3] target cache %s: invPhiHat, STUp, inner t2 plan\n", __func__,
             tgt_hit ? "reuse" : "rebuild");
      printf("[%s t3] source cache %s: primed coords, prephase\n", __func__,
             src_hit ? "reuse" : "rebuild");
    }

    // --- Source-dependent work: XYZp rescaling + prephase ---
    TF isign = (fftSign >= 0) ? 1 : -1;
    if (!src_hit) {
      timer.restart();
      for (int idim = 0; idim < dim; ++idim) tc.XYZp[idim].resize(nj);

      // rescale x_j to x'_j (twist iii)
      std::array<TF, 3> ig = {0, 0, 0};
      for (int idim = 0; idim < dim; ++idim) ig[idim] = 1.0 / m.t3P.gam[idim];
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
      for (BIGINT j = 0; j < nj; ++j) {
        for (int idim = 0; idim < dim; ++idim)
          tc.XYZp[idim][j] = (XYZ_in[idim][j] - m.t3P.C[idim]) * ig[idim];
      }

      // prephase
      tc.prephase.resize(nj);
      if (m.t3P.D[0] != 0.0 || m.t3P.D[1] != 0.0 || m.t3P.D[2] != 0.0) {
        using batch                        = xsimd::batch<TF>;
        constexpr std::size_t batch_width = batch::size;
        constexpr BIGINT batch_size       = static_cast<BIGINT>(batch_width);
        const BIGINT nbatches             = nj / batch_size;
        if (opts.debug > 1) {
          if (nbatches != 0)
            printf("[%s t3] source prephase via xsimd sincos batches + scalar polar tails (W=%zu)\n",
                   __func__, batch_width);
          else
            printf("[%s t3] source prephase via scalar polar\n", __func__);
        }
#pragma omp parallel num_threads(opts.nthreads)
        {
          std::array<TF, batch_width> re_buf;
          std::array<TF, batch_width> im_buf;
          const auto isign_batch = batch(isign);
#pragma omp for schedule(static)
          for (BIGINT jb = 0; jb < nbatches; ++jb) {
            const BIGINT j = jb * batch_size;
            auto phase     = batch(TF(0));
            for (int idim = 0; idim < dim; ++idim)
              phase = xsimd::fma(batch(m.t3P.D[idim]),
                                 batch::load_unaligned(XYZ_in[idim] + j), phase);
            auto [s, c] = finufft::common::math::sincos(isign_batch * phase);
            c.store_unaligned(re_buf.data());
            s.store_unaligned(im_buf.data());
            for (std::size_t lane = 0; lane < batch_width; ++lane)
              tc.prephase[j + static_cast<BIGINT>(lane)] =
                  TC(re_buf[lane], im_buf[lane]);
          }
#pragma omp for schedule(static)
          for (BIGINT j = nbatches * batch_size; j < nj; ++j) {
            TF phase = 0;
            for (int idim = 0; idim < dim; ++idim)
              phase += m.t3P.D[idim] * XYZ_in[idim][j];
            tc.prephase[j] = std::polar(TF(1), isign * phase);
          }
        }
      } else
        for (BIGINT j = 0; j < nj; ++j) tc.prephase[j] = {1.0, 0.0};
      if (opts.debug > 1)
        printf("[%s t3] source cache rebuild:\t%.3g s\n", __func__, timer.elapsedsec());
    } else if (opts.debug > 1) {
      printf("[%s t3] source cache reuse:\t\tno work\n", __func__);
    }
    // XYZ raw pointers always point into the (possibly cached) XYZp vectors
    for (int idim = 0; idim < dim; ++idim) m.XYZ[idim] = tc.XYZp[idim].data();

    // --- Target-dependent work: invPhiHat, STUp, innerT2plan ---
    if (!tgt_hit) {
      // Build into locals for exception safety; move into cache only on success.
      std::vector<TF> new_invPhiHat(nk);
      std::array<std::vector<TF>, 3> new_STUp;
      for (int idim = 0; idim < dim; ++idim) new_STUp[idim].resize(nk);

      Kernel_onedim_FT onedim_phihat(*this);
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
      for (BIGINT k = 0; k < nk; ++k) {
        TF phiHat = 1;
        for (int idim = 0; idim < dim; ++idim) {
          auto tSTUin            = STU_in[idim][k];
          auto tSTUp             = m.t3P.h[idim] * m.t3P.gam[idim] * (tSTUin - m.t3P.D[idim]);
          phiHat                *= onedim_phihat(tSTUp);
          new_STUp[idim][k]      = tSTUp;
        }
        new_invPhiHat[k] = TF(1) / phiHat;
      }

      // Build inner type-2 plan (can throw)
      timer.restart();
      BIGINT t2nmodes[]   = {m.nfdim[0], m.nfdim[1], m.nfdim[2]};
      finufft_opts t2opts = opts;
      t2opts.modeord      = 0;
      t2opts.debug        = std::max(0, opts.debug - 1);
      t2opts.spread_debug = std::max(0, opts.spread_debug - 1);
      t2opts.showwarn     = 0;
      if (!upsamp_locked) t2opts.upsampfac = 0.0;

      FINUFFT_PLAN_T<TF> *tmpplan;
      finufft_makeplan_t<TF>(2, d, t2nmodes, fftSign, batchSize, m.tol, &tmpplan,
                             &t2opts); // throws on error
      std::unique_ptr<FINUFFT_PLAN_T<TF>> guard(tmpplan);
      tmpplan->setpts(nk, new_STUp[0].data(), new_STUp[1].data(), new_STUp[2].data(), 0,
                      nullptr, nullptr, nullptr); // throws on error

      // All allocations/computations succeeded — commit atomically via moves
      tc.invPhiHat    = std::move(new_invPhiHat);
      tc.STUp         = std::move(new_STUp);
      tc.innerT2plan  = std::move(guard);
      for (int idim = 0; idim < dim; ++idim)
        tc.STU[idim].assign(STU_in[idim], STU_in[idim] + nk);
      tc.tkey         = cur_tkey;
      tc.target_valid = true;

      if (opts.debug)
        printf("[%s t3] inner t2 plan & setpts: \t%.3g s\n", __func__, timer.elapsedsec());
    } else if (opts.debug > 1) {
      printf("[%s t3] target cache reuse:\t\tno work\n", __func__);
    }

    // --- Deconv factors (depend on both sources and targets) ---
    if (tgt_hit && src_hit) {
      // Full cache hit: deconv unchanged, skip entirely
      if (opts.debug)
        printf("[%s t3] deconv from full cache (no-op):\t%.3g s\n", __func__,
               timer.elapsedsec());
    } else {
      timer.restart();
      tc.deconv.resize(nk);
      bool Cfinite = std::isfinite(m.t3P.C[0]) && std::isfinite(m.t3P.C[1]) &&
                     std::isfinite(m.t3P.C[2]);
      bool Cnonzero = m.t3P.C[0] != 0.0 || m.t3P.C[1] != 0.0 || m.t3P.C[2] != 0.0;
      bool do_phase = Cfinite && Cnonzero;
      if (do_phase) {
        using batch                        = xsimd::batch<TF>;
        constexpr std::size_t batch_width = batch::size;
        constexpr BIGINT batch_size       = static_cast<BIGINT>(batch_width);
        const BIGINT nbatches             = nk / batch_size;
        if (opts.debug > 1) {
          if (nbatches != 0)
            printf("[%s t3] deconv phasing via xsimd sincos batches + scalar polar tails (W=%zu)\n",
                   __func__, batch_width);
          else
            printf("[%s t3] deconv phasing via scalar polar\n", __func__);
        }
#pragma omp parallel num_threads(opts.nthreads)
        {
          std::array<TF, batch_width> re_buf;
          std::array<TF, batch_width> im_buf;
          const auto isign_batch = batch(isign);
#pragma omp for schedule(static)
          for (BIGINT kb = 0; kb < nbatches; ++kb) {
            const BIGINT k = kb * batch_size;
            auto phase     = batch(TF(0));
            for (int idim = 0; idim < dim; ++idim) {
              const auto stu = batch::load_unaligned(STU_in[idim] + k);
              phase          = xsimd::fma(batch(m.t3P.C[idim]),
                                 stu - batch(m.t3P.D[idim]), phase);
            }
            const auto inv = batch::load_unaligned(tc.invPhiHat.data() + k);
            auto [s, c] = finufft::common::math::sincos(isign_batch * phase);
            (inv * c).store_unaligned(re_buf.data());
            (inv * s).store_unaligned(im_buf.data());
            for (std::size_t lane = 0; lane < batch_width; ++lane)
              tc.deconv[k + static_cast<BIGINT>(lane)] = TC(re_buf[lane], im_buf[lane]);
          }
#pragma omp for schedule(static)
          for (BIGINT k = nbatches * batch_size; k < nk; ++k) {
            TF phase = 0;
            for (int idim = 0; idim < dim; ++idim)
              phase += (STU_in[idim][k] - m.t3P.D[idim]) * m.t3P.C[idim];
            tc.deconv[k] = std::polar(tc.invPhiHat[k], isign * phase);
          }
        }
      } else {
#pragma omp parallel for num_threads(opts.nthreads) schedule(static)
        for (BIGINT k = 0; k < nk; ++k) tc.deconv[k] = TC(tc.invPhiHat[k]);
      }
      if (opts.debug)
        printf("[%s t3] deconv factors:\t\t%.3g s\n", __func__, timer.elapsedsec());
    }

    // --- Sort (depends on source coords + grid; skip on full cache hit) ---
    if (!src_hit || !tgt_hit) {
      timer.restart();
      m.sortIndices.resize(nj);
      m.spopts.spread_direction = 1;
      indexSort();
      if (opts.debug)
        printf("[%s t3] sort (didSort=%d):\t\t%.3g s\n", __func__, (int)m.didSort,
               timer.elapsedsec());
    } else if (opts.debug > 1) {
      printf("[%s t3] sort reuse:\t\t\tno work\n", __func__);
    }

    // Commit source cache key (after all source work succeeded)
    if (!src_hit) {
      for (int idim = 0; idim < dim; ++idim)
        tc.XYZ[idim].assign(XYZ_in[idim], XYZ_in[idim] + nj);
      tc.skey         = cur_skey;
      tc.source_valid = true;
    }
  }
  return 0;
}
