#!/usr/bin/env python3
"""
ES-kernel fitter (power basis, least-squares per piece) with FFT cost
and 'notable sigma' reporting.

Summary
-------
Given a target NUFFT tolerance `tol`, the script searches over (sigma, w, beta, degree)
configurations to fit the ES (exponential of semicircle) kernel on each piece using a
power-basis least-squares approximation evaluated at Chebyshev nodes. The search balances
kernel evaluation/accumulation work ("rank cost") with an estimated FFT cost that grows
with upsampling `sigma`.

Key features
------------
- Chooses (sigma, w, beta, degree) from a specified tolerance `tol` and dimension `dim`.
- The objective minimized is: rank_cost + fft_relative_cost(sigma, dim).
  * rank_cost models kernel evaluation + separable accumulation operations.
  * fft_relative_cost models the additional FFT work from oversampling (`sigma`).
- Corrects the target tolerance by the estimated deconvolution amplification factor
  (Φ(0)/Φ(π/σ))^dim, and enforces an automatic conditioning cap κ_max = m/u (dtype-based).
- Includes a small numeric rounding budget (pre-deconvolution) so float32 results remain
  realistic at tight tolerances.
- Reports the best overall configuration, per-σ winners, and a small set of 'notable'
  σ options that materially improve width/degree/objective.

CLI usage
---------
  python es_kernel_params.py --tol 1e-9 --dim 3 --dtype float64

The `--export-json` flag writes all results (incl. coefficients) to a JSON file.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from numpy.polynomial.legendre import leggauss


# --------------------------------------------------------------------------------------
# ES kernel and Gaussian quadrature utilities
# --------------------------------------------------------------------------------------

def es_kernel(x: np.ndarray, beta: float) -> np.ndarray:
    """ES kernel: φ(x) = exp(β*(sqrt(max(1-x^2,0)) - 1)).
    Defined over |x|≤1, zero-extended outside via the sqrt clip above.
    """
    x = np.asarray(x, dtype=np.float64)
    arg = np.clip(1.0 - x * x, 0.0, None)
    return np.exp(beta * (np.sqrt(arg) - 1.0))


@lru_cache(maxsize=64)
def chebyshev_gauss_nodes(n: int) -> np.ndarray:
    """Chebyshev-Gauss nodes on [-1,1]: t_j = cos((2j-1)π/(2n)), j=1..n."""
    if n <= 0:
        raise ValueError("n>0")
    j = np.arange(1, n + 1, dtype=np.float64)
    return np.cos((2.0 * j - 1.0) * np.pi / (2.0 * n))


@lru_cache(maxsize=64)
def chebyshev_lobatto_nodes(n: int) -> np.ndarray:
    """Chebyshev-Lobatto nodes on [-1,1], including endpoints."""
    if n <= 0:
        raise ValueError("n>0")
    if n == 1:
        return np.array([0.0], dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    return -np.cos(np.pi * k / (n - 1.0))


@lru_cache(maxsize=64)
def leggauss_cached(nquad: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cached Gauss–Legendre nodes/weights with a minimum accuracy floor.
    We always use at least 32 points for robust Φ integrals.
    """
    x, w = leggauss(int(max(32, nquad)))
    return x.astype(np.float64), w.astype(np.float64)


def deconv_log_boost_batch(
        sigma: float, betas: np.ndarray, dim: int, *, omega_scale: float = math.pi, nquad: int = 240
) -> np.ndarray:
    """Return log_boost = dim * (log Φ(0) - log Φ(π/σ)) for each beta.
    Here Φ(ω) is the Fourier transform of the ES kernel. This estimates the
    deconvolution amplification factor at the grid edge ω=π/σ.
    """
    xi_edge = 1.0 / sigma
    omega = omega_scale * xi_edge
    x, w = leggauss_cached(nquad)
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    A = s - 1.0  # ≤ 0
    betas = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    M = np.exp(np.outer(betas, A))
    phi0 = M @ w
    phie = M @ (w * np.cos(omega * x))
    return dim * (np.log(phi0) - np.log(phie))


def tol_with_deconv(tol: float, log_boost: float) -> float:
    """Map post-deconvolution tolerance back to pre-deconv target.
    We fit in the pre-deconv space using tol * exp(log_boost).
    """
    return tol * math.exp(log_boost)


def unit_roundoff(dtype: str) -> float:
    """Return unit roundoff u = eps/2 for the given float dtype."""
    s = str(dtype).lower()
    eps = np.finfo(np.float32).eps if ("32" in s or "single" in s or s == "f4") else np.finfo(np.float64).eps
    return float(eps * 0.5)


def kappa_cap_for_dtype(dtype: str) -> float:
    """Automatic conditioning cap: κ_max = m / u.
    Heuristic m accounts for model mismatch in the fit and light amplification.
    - float32: u≈5.96e-8, m≈5e-2  → κ_max ~ 8e5
    - float64: u≈1.11e-16, m≈1e-4 → κ_max ~ 9e11
    """
    u = unit_roundoff(dtype)
    m = 5e-2 if ("32" in dtype.lower() or "single" in dtype.lower() or dtype.lower() == "f4") else 1e-4
    return m / u


# --------------------------------------------------------------------------------------
# Power-basis LS design & evaluation helpers
# --------------------------------------------------------------------------------------

@lru_cache(maxsize=64)
def vandermonde_power(n_samples: int, max_deg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return Vandermonde matrix [1, t, t^2, ...] at Chebyshev-Gauss nodes and the node vector."""
    t = chebyshev_gauss_nodes(n_samples)
    V = np.empty((n_samples, max_deg + 1), dtype=np.float64)
    V[:, 0] = 1.0
    if max_deg >= 1:
        V[:, 1] = t
        for k in range(2, max_deg + 1):
            V[:, k] = V[:, k - 1] * t
    return V, t


def polyval_matrix(C: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate many power-basis polynomials (columns of C) at points t via Horner.
    C has shape (degree+1, n_polys) in ascending power order.
    """
    C = np.asarray(C, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    ns = t.shape[0]
    y = np.broadcast_to(C[-1, :], (ns, C.shape[1])).copy()
    for k in range(C.shape[0] - 2, -1, -1):
        y *= t[:, None]
        y += C[k, :]
    return y


# --------------------------------------------------------------------------------------
# Fitting per piece and uniform error checking
# --------------------------------------------------------------------------------------

def fit_power(w: int, beta: float, fitd: int, *, max_degree: int, os: int, cutoff: float) -> Tuple[np.ndarray, int]:
    """Least-squares fit on a single piece using Chebyshev-Gauss nodes.
    Returns (trimmed coefficients, used_degree). Rows with max-|coeff| below `cutoff`
    are pruned to avoid storing negligible high powers.
    """
    n_samples = max(int(os * (fitd + 1)), fitd + 1, 30)
    Vmax, t = vandermonde_power(n_samples, max_degree)
    V = Vmax[:, :fitd + 1]
    h = 1.0 / w
    xi = -1.0 + h * (2 * np.arange(1, w + 1) - 1)
    R = es_kernel(xi[None, :] + h * t[:, None], beta)
    C_fit, *_ = np.linalg.lstsq(V, R, rcond=None)  # (fitd+1) x w
    rows_max = np.max(np.abs(C_fit), axis=1)
    idx = np.where(rows_max > cutoff)[0]
    nc = 1 if idx.size == 0 else int(idx.max()) + 1
    return C_fit[:nc, :], nc - 1


def find_min_degree(
        *, w: int, sigma: float, beta: float, tol_fit: float,
        ns_err: int, max_degree: int, os_ls: int, cutoff_factor: float
) -> Optional[Tuple[np.ndarray, int, float]]:
    """Find the minimal degree whose uniform error on a Lobatto grid ≤ tol_fit.
    Binary-searches degree; returns (coeffs, used_degree_after_trim, max_error).
    """
    tL = chebyshev_lobatto_nodes(ns_err)
    h = 1.0 / w
    xi = -1.0 + h * (2 * np.arange(1, w + 1) - 1)
    F = es_kernel(xi[None, :] + tL[:, None] * h, beta)

    def eval_err(C: np.ndarray) -> float:
        return float(np.max(np.abs(F - polyval_matrix(C, tL))))

    fitd = 2
    fail, succ = None, None
    best: Optional[Tuple[np.ndarray, int, float]] = None
    # Exponential growth until a feasible degree is found or max_degree is hit
    while True:
        cutoff = cutoff_factor * tol_fit
        C, d_used = fit_power(w, beta, fitd, max_degree=max_degree, os=os_ls, cutoff=cutoff)
        err = eval_err(C)
        if err <= tol_fit:
            succ = fitd
            best = (C, d_used, err)
            break
        fail = fitd
        if fitd >= max_degree:
            return None
        fitd = max(fitd + 1, 2 * fitd)
        if fitd > max_degree:
            fitd = max_degree

    # Tight binary search between (fail, succ)
    lo = fail if fail is not None else 2
    hi = succ
    while hi - lo > 1:
        mid = (hi + lo) // 2
        cutoff = cutoff_factor * tol_fit
        C, d_used = fit_power(w, beta, mid, max_degree=max_degree, os=os_ls, cutoff=cutoff)
        err = eval_err(C)
        if err <= tol_fit:
            hi = mid
            if (d_used < best[1]) or (d_used == best[1] and err < best[2]):
                best = (C, d_used, err)
        else:
            lo = mid
    return best


# --------------------------------------------------------------------------------------
# Heuristics for search space and cost models
# --------------------------------------------------------------------------------------

def phi_sigma(sigma: float) -> float:
    """Helper: φ(σ) = sqrt(1 - 1/max(σ,1)), used in w prediction."""
    return math.sqrt(max(1.0 - 1.0 / sigma, 0.0))


def predict_w(eps: float, sigma: float, *, C: float = 4.0, b: float = 0.5) -> int:
    """Analytic rule of thumb for kernel width: w ≈ log(C/eps) / (π φ(σ)) + bias b.
    Rounded up to an integer before SIMD alignment.
    """
    return int(math.ceil(math.log(C / eps) / (math.pi * phi_sigma(sigma)) + b))


def beta_from_ws(w: int, sigma: float, *, gamma: float = 0.97) -> float:
    """Default β as a function of (w, σ): β ≈ γ π (1 - 1/(2σ)) w."""
    return float((gamma * np.pi * (1.0 - 1.0 / (2.0 * sigma))) * w)


def choose_sigma_window(tol: float) -> Tuple[float, float]:
    """Choose a small σ search window based on digits of tolerance.
    Tighter tolerances push the window toward σ≈2 where ES kernels are strongest.
    """
    t = -math.log10(tol)
    if t >= 14.0:
        return 1.9, 2.05
    if t >= 12.0:
        return 1.6, 2.0
    return 1.3, 1.9


def rounding_floor(dim: int, d_used: int, w: int, *, u: float, safety: float = 1.5) -> float:
    """Estimate absolute rounding noise (pre-deconvolution) per output value.
    - Horner evaluation across `dim` dims: RMS ≈ u * sqrt(dim*(d_used+1))
    - Separable accumulation across w^dim weights: RMS ≈ u * w^{-dim/2}
    A modest safety factor is applied.
    """
    eval_terms = max(dim * (d_used + 1), 1)
    eval_rms = u * math.sqrt(0.5 * eval_terms)
    accum_rms = u * math.sqrt(0.5) * (w ** (-0.5 * dim))
    return safety * (eval_rms + accum_rms)


def fft_relative_cost(sigma: float, dim: int) -> float:
    """Relative d-D FFT cost at fixed base grid N (up to a constant factor):
      cost ∝ d * σ^d * log2(σ).
    This captures the σ-dependent penalty from oversampling.
    """
    s = max(float(sigma), 1.0)
    return (s ** dim) * dim * math.log(s, 2.0)


# --------------------------------------------------------------------------------------
# Main search routine
# --------------------------------------------------------------------------------------

@dataclass
class KernelFitResult:
    tol: float
    dim: int
    upsampfac: float
    w: int
    d: int
    beta: float
    maxerr: float
    rank_cost: float
    fft_cost: float
    obj_cost: float
    bytes: int
    coeffs: Optional[np.ndarray] = None
    num_floor: float = 0.0
    total_err: float = 0.0
    log_cond: float = 0.0
    cond: float = 1.0


@dataclass
class FitMultiSigma:
    best: KernelFitResult
    per_sigma: List[KernelFitResult]
    notable: List[KernelFitResult]


def fit_es_kernel_for_tol(
        tol: float,
        *,
        dim: int = 3,
        simd_size: int = 2,
        w_min: int = 4,
        w_max: int = 32,
        max_degree: int = 64,
        cutoff_factor: float = 0.5,
        corr_nquad: int = 240,
        omega_scale: float = math.pi,
        dtype: str = "float64",
        notable_max: int = 5,
) -> FitMultiSigma:
    """Search for an ES-kernel configuration meeting `tol` with a balanced objective.

    The procedure:
      1) Choose a small grid of σ values centered near the expected sweet spot.
      2) For each σ, predict a plausible width w and explore nearby SIMD-aligned widths.
      3) For each (σ,w), define a β grid around a closed-form guess and compute the
         deconvolution amplification factor Φ(0)/Φ(π/σ) to map the tolerance back to the
         pre-deconvolution fitting domain.
      4) For each (σ,w,β), find the minimal degree achieving the mapped tolerance, with a
         rounding floor subtracted if necessary.
      5) Score each candidate using rank_cost + fft_cost and keep winners.
    """
    # Sampling density for LS and for the uniform error check; increased for tight tolerances
    t = -math.log10(tol)
    os_ls = 2 + (1 if t > 10 else 0) + (1 if t > 13 else 0) + (1 if t > 14 else 0)
    ns_err = int(24 + 10 * t)
    if ns_err % 2 == 1:
        ns_err += 1

    # σ candidates (small window keeps the search fast & focused)
    smin, smax = choose_sigma_window(tol)
    sigma_grid = np.linspace(smin, smax, 9)

    # Conditioning guard
    u = unit_roundoff(dtype)
    log_cond_cap = math.log(kappa_cap_for_dtype(dtype))

    best_overall: Optional[KernelFitResult] = None
    best_per_sigma: Dict[float, KernelFitResult] = {}

    for sigma in sigma_grid:
        # Predict w and align to SIMD, clamp to allowed range
        w_pred = predict_w(tol, sigma)
        if w_pred < w_min:
            w_pred = w_min
        if w_pred > w_max:
            continue
        if w_pred % simd_size != 0:
            w_pred += simd_size - (w_pred % simd_size)
        if w_pred > w_max:
            continue

        # Explore a small neighborhood of widths around prediction in SIMD tiles
        w_candidates: List[int] = []
        tried = set()
        for delta_tiles in range(0, 1 + (w_max - w_min) // simd_size):
            for sign in ([0] if delta_tiles == 0 else [-1, +1]):
                wc = w_pred + sign * delta_tiles * simd_size
                if wc < w_min or wc > w_max or wc in tried:
                    continue
                tried.add(wc)
                w_candidates.append(wc)
            if len(w_candidates) >= 5:
                break

        # β grid around closed-form β(w,σ)
        beta0 = beta_from_ws(w_pred, sigma)
        tight = (tol <= 3e-14)
        b_lo = (0.85 if tight else 0.9) * beta0
        b_hi = (1.30 if tight else 1.10) * beta0
        beta_grid = np.linspace(b_lo, b_hi, 7 if tight else 5)

        # Precompute deconvolution log-boosts for this σ across β candidates
        boosts = deconv_log_boost_batch(sigma, beta_grid, dim, omega_scale=omega_scale, nquad=corr_nquad)

        best_for_sigma: Optional[KernelFitResult] = None

        for w in w_candidates:
            for beta, lb in zip(beta_grid, boosts):
                # Conditioning guard: skip if predicted κ exceeds dtype-specific cap
                if float(lb) > log_cond_cap:
                    continue

                # Map post-deconv tolerance back to pre-deconv target for fitting
                tol_target_fit = tol_with_deconv(tol, float(lb))

                # First pass: find minimal degree meeting tol_target_fit
                ans = find_min_degree(
                    w=w, sigma=sigma, beta=float(beta), tol_fit=tol_target_fit,
                    ns_err=ns_err, max_degree=max_degree, os_ls=os_ls,
                    cutoff_factor=cutoff_factor,
                )
                if ans is None:
                    continue
                C, d_used, err = ans

                # Apply rounding floor; if necessary, retighten the target and refit
                num_floor = rounding_floor(dim, d_used, w, u=u, safety=1.5)
                if err + num_floor > tol_target_fit:
                    tol_fit2 = max(tol_target_fit - num_floor, 1e-19)
                    ans2 = find_min_degree(
                        w=w, sigma=sigma, beta=float(beta), tol_fit=tol_fit2,
                        ns_err=ns_err, max_degree=max_degree, os_ls=os_ls,
                        cutoff_factor=cutoff_factor,
                    )
                    if ans2 is None:
                        continue
                    C2, d_used2, err2 = ans2
                    num_floor2 = rounding_floor(dim, d_used2, w, u=u, safety=1.5)
                    if err2 + num_floor2 > tol_target_fit:
                        continue
                    C, d_used, err, num_floor = C2, d_used2, err2, num_floor2

                # Cost model
                tiles = float(math.ceil(w / float(simd_size)))
                eval_ops = float(dim) * tiles * float(d_used + 1)
                accum_ops = tiles * float(w ** max(dim - 1, 0))
                rank_cost = eval_ops + accum_ops
                fft_cost = fft_relative_cost(sigma, dim)
                obj_cost = rank_cost + fft_cost

                res = KernelFitResult(
                    tol=tol,
                    dim=dim,
                    upsampfac=float(sigma),
                    w=int(w),
                    d=int(d_used),
                    beta=float(beta),
                    maxerr=float(err),
                    rank_cost=float(rank_cost),
                    fft_cost=float(fft_cost),
                    obj_cost=float(obj_cost),
                    bytes=int((d_used + 1) * w * 8),
                    coeffs=C.astype(np.float64, copy=False),
                    num_floor=float(num_floor),
                    total_err=float(err + num_floor),
                    log_cond=float(lb),
                    cond=float(math.exp(lb) if lb < 700.0 else 1e308),
                )

                # Track best per-σ
                if (best_for_sigma is None) or (res.obj_cost < best_for_sigma.obj_cost):
                    best_for_sigma = res

                # Track global best
                if (best_overall is None) or (res.obj_cost < best_overall.obj_cost):
                    best_overall = res

        if best_for_sigma is not None:
            best_per_sigma[float(sigma)] = best_for_sigma

    if best_overall is None or not best_per_sigma:
        raise RuntimeError("No configuration met the requested tolerance in the search window (κ cap enforced).")

    # Build ordered list of per-σ winners (feasible σ only), lowest σ first
    per_sigma_list = [best_per_sigma[s] for s in sorted(best_per_sigma.keys())]

    # Select 'notable' σ options: begin with the smallest feasible σ and keep those with
    # a meaningful drop in width/degree or a clear objective improvement.
    # Thresholds are fixed internally at 1 for dw and dd; objective improvement threshold is 15%.
    notable: List[KernelFitResult] = []
    last: Optional[KernelFitResult] = None
    notable_dw = 1
    notable_dd = 1
    notable_impr = 0.15
    for res in per_sigma_list:
        if last is None:
            notable.append(res)
            last = res
            if len(notable) >= notable_max:
                break
            continue
        w_drop = (last.w - res.w) >= notable_dw
        d_drop = (last.d - res.d) >= notable_dd
        obj_improve = res.obj_cost <= last.obj_cost * (1.0 - notable_impr)
        if w_drop or d_drop or obj_improve:
            notable.append(res)
            last = res
            if len(notable) >= notable_max:
                break

    return FitMultiSigma(best=best_overall, per_sigma=per_sigma_list, notable=notable)


# --------------------------------------------------------------------------------------
# Command-line interface
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DUCC0-style ES kernel fit (power-basis LS per piece) with FFT cost and notable σ options",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--tol", type=float, required=True, help="target uniform NUFFT tolerance")
    p.add_argument("--dim", type=int, default=3, choices=[1, 2, 3], help="problem dimension")
    p.add_argument("--dtype", type=str, choices=["float32", "float64"], default="float64",
                   help="intended NUFFT dtype (sets κ cap and rounding budget)")
    p.add_argument("--simd-size", type=int, default=2, help="SIMD tile width for pieces")
    p.add_argument("--w-min", type=int, default=4, help="minimum kernel width (pieces)")
    p.add_argument("--w-max", type=int, default=24, help="maximum kernel width (pieces)")
    p.add_argument("--max-degree", type=int, default=64, help="maximum polynomial degree to try")
    p.add_argument("--corr-nquad", type=int, default=240, help="Gauss–Legendre points for Φ integral")
    p.add_argument("--notable-max", type=int, default=5, help="maximum notable σ to list")
    p.add_argument("--export-json", type=str, default=None, help="optional path to write full JSON (incl. coeffs)")
    return p.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        out = fit_es_kernel_for_tol(
            args.tol,
            dim=args.dim,
            simd_size=args.simd_size,
            w_min=args.w_min,
            w_max=args.w_max,
            max_degree=args.max_degree,
            corr_nquad=args.corr_nquad,
            dtype=args.dtype,
            notable_max=args.notable_max,
        )
    except Exception as e:
        sys.stderr.write(f"ERROR: {e}\n")
        return 2

    # Export JSON if requested
    if args.export_json:
        payload: Dict[str, Any] = {
            "best": asdict(out.best),
            "per_sigma": [asdict(r) for r in out.per_sigma],
            "notable": [asdict(r) for r in out.notable],
        }
        # Convert numpy arrays to lists and record shapes
        for section in ("best",):
            if isinstance(payload[section].get("coeffs"), np.ndarray):
                payload[section]["coeffs_shape"] = list(payload[section]["coeffs"].shape)
                payload[section]["coeffs"] = payload[section]["coeffs"].tolist()
        for sec in ("per_sigma", "notable"):
            for rec in payload[sec]:
                if isinstance(rec.get("coeffs"), np.ndarray):
                    rec["coeffs_shape"] = list(rec["coeffs"].shape)
                    rec["coeffs"] = rec["coeffs"].tolist()
        try:
            with open(args.export_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Saved JSON to: {args.export_json}")
        except Exception as e:
            sys.stderr.write(f"ERROR writing JSON: {e}\n")
            return 2
        return 0

    # Console report
    b = out.best
    print("ES kernel fit (power basis) with FFT cost")
    print(f"BEST: tol={b.tol:g}  dim={b.dim}  dtype={args.dtype}")
    print(f"      sigma={b.upsampfac:.6f}  w={b.w}  d={b.d}  beta={b.beta:.6g}")
    print(f"      maxerr={b.maxerr:.3e}  numeric_floor≈{b.num_floor:.3e}  total_est≈{b.total_err:.3e} (pre-deconv)")
    print(f"      κ≈{b.cond:.3g}  logκ={b.log_cond:.3f}  (κ cap≈{kappa_cap_for_dtype(args.dtype):.2e})")
    print(f"      rank_cost={b.rank_cost:.3g}  fft_rel_cost={b.fft_cost:.3g}  obj={b.obj_cost:.3g}  bytes={b.bytes}")
    print(f"      coeffs shape: ({b.d + 1}, {b.w})  (ascending power order per piece)")
    if out.notable:
        print("\nNotable sigma options (lowest σ first):")
        print("  σ        w    d    obj        rank_cost  fft_rel  κ")
        for r in out.notable:
            print(
                f"  {r.upsampfac:7.4f}  {r.w:3d}  {r.d:3d}  {r.obj_cost:9.3g}  {r.rank_cost:9.3g}  {r.fft_cost:7.3g}  {r.cond:>8.2g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
