#!/usr/bin/env python3
import numpy as np
from scipy.optimize import linprog
from pathlib import Path


def load_rows(path: Path):
    rows = []
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.split()
        if len(parts) < 7:
            continue
        sigma, ns, tol, beta0, beta_best, err_best, err_ratio = map(float, parts)
        rows.append((sigma, ns, beta0, beta_best))
    return np.array(rows)


def build_features_factor(rows):
    sigma = rows[:, 0]
    ns = rows[:, 1]
    dsigma = sigma - 1.5
    dsigma2 = dsigma * dsigma
    inv = 1.0 / ns
    inv2 = inv * inv
    inv3 = inv2 * inv

    # Factor model: beta = beta0 * f(sigma, ns)
    features = [
        np.ones_like(ns),
        inv,
        inv2,
        inv3,
        dsigma,
        dsigma * inv,
        dsigma * inv2,
        dsigma * inv3,
        dsigma2,
        dsigma2 * inv,
        dsigma2 * inv2,
    ]
    return np.column_stack(features)


def build_features_beta(rows):
    sigma = rows[:, 0]
    ns = rows[:, 1]
    t = ns * (1.0 - 1.0 / (2.0 * sigma))
    invt = 1.0 / t
    invt2 = invt * invt
    invns = 1.0 / ns
    invns2 = invns * invns
    dsigma = sigma - 1.5
    dsigma2 = dsigma * dsigma

    # Direct beta model: beta ~= X c
    features = [
        np.ones_like(ns),
        t,
        invt,
        invt2,
        invns,
        invns2,
        dsigma,
        dsigma * invns,
        dsigma * invns2,
        dsigma2,
        dsigma2 * invns,
    ]
    return np.column_stack(features)


def fit_factor_ls(rows):
    beta0 = rows[:, 2]
    beta = rows[:, 3]
    X = build_features_factor(rows)
    coef, *_ = np.linalg.lstsq(X, beta / beta0, rcond=None)

    pred = beta0 * (X @ coef)
    rel = (pred - beta) / beta
    stats = {
        "mean": rel.mean(),
        "std": rel.std(),
        "maxabs": np.max(np.abs(rel)),
    }
    return coef, stats


def fit_factor_minimax(rows):
    beta0 = rows[:, 2]
    beta = rows[:, 3]
    X = build_features_factor(rows)

    # Minimax on relative error: minimize t s.t. |Xc - y| <= t*|y|
    y = beta / beta0
    m, n = X.shape
    abs_y = np.abs(y)
    abs_y[abs_y == 0.0] = 1.0

    # Decision variables: [c (n), t (1)]
    c_obj = np.zeros(n + 1)
    c_obj[-1] = 1.0

    A = np.zeros((2 * m, n + 1))
    b = np.zeros(2 * m)
    #  X c - y <= t*|y|
    A[:m, :n] = X
    A[:m, -1] = -abs_y
    b[:m] = y
    # -X c + y <= t*|y|
    A[m:, :n] = -X
    A[m:, -1] = -abs_y
    b[m:] = -y

    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    coef = res.x[:n]
    pred = beta0 * (X @ coef)
    rel = (pred - beta) / beta
    stats = {
        "mean": rel.mean(),
        "std": rel.std(),
        "maxabs": np.max(np.abs(rel)),
    }
    return coef, stats


def fit_beta_minimax(rows):
    beta = rows[:, 3]
    X = build_features_beta(rows)

    # Minimax on relative error: minimize t s.t. |Xc - y| <= t*|y|
    y = beta
    m, n = X.shape
    abs_y = np.abs(y)
    abs_y[abs_y == 0.0] = 1.0

    c_obj = np.zeros(n + 1)
    c_obj[-1] = 1.0

    A = np.zeros((2 * m, n + 1))
    b = np.zeros(2 * m)
    A[:m, :n] = X
    A[:m, -1] = -abs_y
    b[:m] = y
    A[m:, :n] = -X
    A[m:, -1] = -abs_y
    b[m:] = -y

    bounds = [(None, None)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.message}")

    coef = res.x[:n]
    pred = X @ coef
    rel = (pred - beta) / beta
    stats = {
        "mean": rel.mean(),
        "std": rel.std(),
        "maxabs": np.max(np.abs(rel)),
    }
    return coef, stats


def per_ns_polyfit(rows, max_degree=8):
    sigma = rows[:, 0]
    ns = rows[:, 1].astype(int)
    beta = rows[:, 3]

    results = {}
    for n in range(2, 17):
        mask = ns == n
        if not np.any(mask):
            continue
        s = sigma[mask]
        y = beta[mask]
        ds = s - 1.5
        best = None
        for deg in range(1, max_degree + 1):
            # Fit polynomial in dsigma to beta for fixed ns.
            coef = np.polyfit(ds, y, deg)
            pred = np.polyval(coef, ds)
            rel = (pred - y) / y
            stats = {
                "mean": rel.mean(),
                "std": rel.std(),
                "maxabs": np.max(np.abs(rel)),
            }
            if best is None or stats["maxabs"] < best["stats"]["maxabs"]:
                best = {"deg": deg, "coef": coef, "stats": stats}
        results[n] = best
    return results


def per_ns_polyfit_minimax(rows, max_degree=8):
    sigma = rows[:, 0]
    ns = rows[:, 1].astype(int)
    beta = rows[:, 3]

    results = {}
    for n in range(2, 17):
        mask = ns == n
        if not np.any(mask):
            continue
        s = sigma[mask]
        y = beta[mask]
        ds = s - 1.5
        best = None
        for deg in range(1, max_degree + 1):
            # Polynomial basis in dsigma: [1, ds, ds^2, ...]
            X = np.column_stack([ds ** k for k in range(deg + 1)])
            m, p = X.shape
            abs_y = np.abs(y)
            abs_y[abs_y == 0.0] = 1.0

            c_obj = np.zeros(p + 1)
            c_obj[-1] = 1.0
            A = np.zeros((2 * m, p + 1))
            b = np.zeros(2 * m)
            A[:m, :p] = X
            A[:m, -1] = -abs_y
            b[:m] = y
            A[m:, :p] = -X
            A[m:, -1] = -abs_y
            b[m:] = -y
            bounds = [(None, None)] * p + [(0.0, None)]
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            if not res.success:
                continue
            coef = res.x[:p]
            pred = X @ coef
            rel = (pred - y) / y
            stats = {
                "mean": rel.mean(),
                "std": rel.std(),
                "maxabs": np.max(np.abs(rel)),
            }
            if best is None or stats["maxabs"] < best["stats"]["maxabs"]:
                best = {"deg": deg, "coef": coef, "stats": stats}
        results[n] = best
    return results


def per_ns_polyfit_minimax_cheby(rows, max_degree=8):
    sigma = rows[:, 0]
    ns = rows[:, 1].astype(int)
    beta = rows[:, 3]

    results = {}
    for n in range(2, 17):
        mask = ns == n
        if not np.any(mask):
            continue
        s = sigma[mask]
        y = beta[mask]
        smin = s.min()
        smax = s.max()
        if smax == smin:
            continue
        # scale sigma to [-1, 1] for Chebyshev stability
        x = (2.0 * (s - smin) / (smax - smin)) - 1.0
        best = None
        for deg in range(1, max_degree + 1):
            X = np.polynomial.chebyshev.chebvander(x, deg)
            m, p = X.shape
            abs_y = np.abs(y)
            abs_y[abs_y == 0.0] = 1.0

            c_obj = np.zeros(p + 1)
            c_obj[-1] = 1.0
            A = np.zeros((2 * m, p + 1))
            b = np.zeros(2 * m)
            A[:m, :p] = X
            A[:m, -1] = -abs_y
            b[:m] = y
            A[m:, :p] = -X
            A[m:, -1] = -abs_y
            b[m:] = -y
            bounds = [(None, None)] * p + [(0.0, None)]
            res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
            if not res.success:
                continue
            coef = res.x[:p]
            pred = X @ coef
            rel = (pred - y) / y
            stats = {
                "mean": rel.mean(),
                "std": rel.std(),
                "maxabs": np.max(np.abs(rel)),
            }
            if best is None or stats["maxabs"] < best["stats"]["maxabs"]:
                best = {"deg": deg, "coef": coef, "stats": stats, "smin": smin, "smax": smax}
        results[n] = best
    return results


def main():
    path = Path("matlab/test/results/betasweep_text_1D_double.txt")
    rows = load_rows(path)
    if rows.size == 0:
        raise SystemExit("No data rows found in betasweep output.")

    coef_ls, stats_ls = fit_factor_ls(rows)
    print("Least-squares factor coefficients:")
    print(coef_ls)
    print("Relative error stats (LS):")
    print(stats_ls)

    coef_mm, stats_mm = fit_factor_minimax(rows)
    print("Minimax factor coefficients:")
    print(coef_mm)
    print("Relative error stats (minimax):")
    print(stats_mm)

    coef_bm, stats_bm = fit_beta_minimax(rows)
    print("Minimax direct-beta coefficients:")
    print(coef_bm)
    print("Relative error stats (minimax, beta):")
    print(stats_bm)

    per_ns = per_ns_polyfit(rows, max_degree=8)
    print("Per-ns polynomial fits (beta vs dsigma):")
    for n in sorted(per_ns.keys()):
        info = per_ns[n]
        coef = info["coef"]
        stats = info["stats"]
        print(
            f"ns={n:2d} deg={info['deg']} maxabs={stats['maxabs']:.6g} "
            f"std={stats['std']:.6g} mean={stats['mean']:.6g}"
        )
        print(f"  coef (highest deg first): {coef}")

    per_ns_mm = per_ns_polyfit_minimax(rows, max_degree=8)
    print("Per-ns polynomial fits (minimax, beta vs dsigma):")
    for n in sorted(per_ns_mm.keys()):
        info = per_ns_mm[n]
        coef = info["coef"]
        stats = info["stats"]
        print(
            f"ns={n:2d} deg={info['deg']} maxabs={stats['maxabs']:.6g} "
            f"std={stats['std']:.6g} mean={stats['mean']:.6g}"
        )
        print(f"  coef (lowest deg first): {coef}")

    per_ns_mm_cheb = per_ns_polyfit_minimax_cheby(rows, max_degree=8)
    print("Per-ns polynomial fits (minimax, Chebyshev basis):")
    worst = None
    for n in sorted(per_ns_mm_cheb.keys()):
        info = per_ns_mm_cheb[n]
        coef = info["coef"]
        stats = info["stats"]
        print(
            f"ns={n:2d} deg={info['deg']} maxabs={stats['maxabs']:.6g} "
            f"std={stats['std']:.6g} mean={stats['mean']:.6g}"
        )
        print(f"  sigma in [{info['smin']:.3g}, {info['smax']:.3g}], coef (T0..Tdeg): {coef}")
        if worst is None or stats["maxabs"] > worst["stats"]["maxabs"]:
            worst = {"ns": n, "stats": stats}
    if worst:
        print(
            f"Worst-case maxabs across ns (minimax Chebyshev): "
            f"ns={worst['ns']} maxabs={worst['stats']['maxabs']:.6g}"
        )

    # Compact formula search: try a few small bases and report best maxabs.
    sigma = rows[:, 0]
    ns = rows[:, 1]
    beta = rows[:, 3]
    t = ns * (1.0 - 1.0 / (2.0 * sigma))
    invns = 1.0 / ns
    invns2 = invns * invns
    invt = 1.0 / t
    invt2 = invt * invt
    ds = sigma - 1.5
    ds2 = ds * ds

    bases = {
        "b0 + b1*t + b2/invns + b3*ds + b4*ds/invns": [
            np.ones_like(ns), t, invns, ds, ds * invns
        ],
        "b0 + b1*t + b2/invns + b3/invns2 + b4*ds": [
            np.ones_like(ns), t, invns, invns2, ds
        ],
        "b0 + b1*t + b2/invt2 + b3/invns + b4*invns2 + b5*ds": [
            np.ones_like(ns), t, invt2, invns, invns2, ds
        ],
        "b0 + b1*t + b2/invt2 + b3/invns + b4*invns2 + b5*ds + b6*ds/invns": [
            np.ones_like(ns), t, invt2, invns, invns2, ds, ds * invns
        ],
    }

    print("Compact model search (minimax, beta):")
    for name, feats in bases.items():
        X = np.column_stack(feats)
        m, p = X.shape
        abs_y = np.abs(beta)
        abs_y[abs_y == 0.0] = 1.0
        c_obj = np.zeros(p + 1)
        c_obj[-1] = 1.0
        A = np.zeros((2 * m, p + 1))
        b = np.zeros(2 * m)
        A[:m, :p] = X
        A[:m, -1] = -abs_y
        b[:m] = beta
        A[m:, :p] = -X
        A[m:, -1] = -abs_y
        b[m:] = -beta
        bounds = [(None, None)] * p + [(0.0, None)]
        res = linprog(c_obj, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if not res.success:
            print(f"  {name}: linprog failed: {res.message}")
            continue
        coef = res.x[:p]
        pred = X @ coef
        rel = (pred - beta) / beta
        print(
            f"  {name}: maxabs={np.max(np.abs(rel)):.6g} std={rel.std():.6g} mean={rel.mean():.6g}"
        )
        print(f"    coef: {coef}")

    # 2D polynomial fit: beta(ns, sigma) with total degree <= d
    sigma = rows[:, 0]
    ns = rows[:, 1]
    beta = rows[:, 3]

    # Scale to [-1, 1] for numerical stability.
    ns_min, ns_max = 2.0, 16.0
    sig_min, sig_max = 1.25, 2.0
    x = (2.0 * (ns - ns_min) / (ns_max - ns_min)) - 1.0
    y = (2.0 * (sigma - sig_min) / (sig_max - sig_min)) - 1.0

    def build_terms(total_deg):
        terms = []
        powers = []
        for i in range(total_deg + 1):
            for j in range(total_deg + 1 - i):
                terms.append((x ** i) * (y ** j))
                powers.append((i, j))
        X = np.column_stack(terms)
        return X, powers

    print("2D polynomial fit (least squares, total degree):")
    for deg in range(1, 17):
        X, powers = build_terms(deg)
        coef, *_ = np.linalg.lstsq(X, beta, rcond=None)
        pred = X @ coef
        rel = (pred - beta) / beta
        print(
            f"  deg={deg:2d} maxabs={np.max(np.abs(rel)):.6g} "
            f"std={rel.std():.6g} mean={rel.mean():.6g} terms={len(powers)}"
        )

    # LASSO on 2D polynomial basis (sparse model selection).
    try:
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        print(f"LASSO skipped (sklearn unavailable): {exc}")
        return

    max_degree = 8
    X, powers = build_terms(max_degree)
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)

    # Use a log-spaced alpha grid; scale by n_samples for stability.
    alphas = np.logspace(-6, 1, 50)
    lasso = LassoCV(alphas=alphas, cv=5, fit_intercept=True, max_iter=10000)
    lasso.fit(Xs, beta)
    coef = lasso.coef_
    intercept = lasso.intercept_
    pred = Xs @ coef + intercept
    rel = (pred - beta) / beta
    nz = np.sum(np.abs(coef) > 1e-10)

    print("LASSO (2D polynomial, degree 8):")
    print(
        f"  alpha={lasso.alpha_:.6g} nonzero={nz} maxabs={np.max(np.abs(rel)):.6g} "
        f"std={rel.std():.6g} mean={rel.mean():.6g}"
    )
    print("  nonzero terms (i,j,coef_scaled):")
    for (i, j), c in zip(powers, coef):
        if abs(c) > 1e-10:
            print(f"    ({i},{j}) {c}")


if __name__ == "__main__":
    main()
