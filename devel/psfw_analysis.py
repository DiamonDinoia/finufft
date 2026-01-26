#!/usr/bin/env python3
import math
import time
import csv
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

# ============================================================
# 0) Beta fit you provided (beta = A*n + B + C/n), with n = ns = w
# ============================================================

def beta_fit(sigma: float, ns: int) -> float:
    s  = float(sigma)
    n  = float(ns)
    s2 = s*s
    A = -0.00542508 + 2.07540434*s - 0.45372236*s2
    B =  0.69680414 - 0.90730009*s + 0.31118099*s2
    C = -0.39917127
    return A*n + B + C/n


# ============================================================
# 1) Legendre helpers (ported from your C++)
# ============================================================

def legepol(x: float, n: int) -> Tuple[float, float]:
    # returns (P_n(x), P'_n(x))
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0

    pk = 1.0
    pkp1 = x
    for k in range(1, n):
        pkm1 = pk
        pk = pkp1
        pkp1 = ((2*k + 1)*x*pk - k*pkm1) / (k + 1)

    pol = pkp1
    der = n * (x*pkp1 - pk) / (x*x - 1.0)
    return pol, der


def legetayl(pol: float, der: float, x: float, h: float, n: int, k: int) -> Tuple[float, float]:
    # Taylor-like evaluation used by legerts
    done = 1.0
    q0 = pol
    q1 = der * h
    q2 = (2*x*der - n*(n + done)*pol) / (1 - x*x)
    q2 = q2 * h*h / 2.0

    s = q0 + q1 + q2
    sder = q1/h + q2*2.0/h

    if k <= 2:
        return s, sder

    qi = q1
    qip1 = q2
    for i in range(1, k-1):  # i=1..k-2
        d = 2*x*(i+1)*(i+1)/h * qip1 - (n*(n+done) - i*(i+1))*qi
        d = d / ((i+1)*(i+2)) * h*h / (1 - x*x)
        qip2 = d
        s += qip2
        sder += d*(i+2)/h
        qi = qip1
        qip1 = qip2

    return s, sder


def legerts(itype: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gaussian-Legendre roots/weights on [-1,1].
      itype=1 -> roots + weights
      itype=0 -> roots only (weights returned but not meaningful)
    """
    # Determine k based on float precision heuristic in original code
    k = 30
    d = 1.0
    if (d + 1e-24) != d:
        k = 54

    half = n // 2
    ifodd = n - 2*half

    pi_val = math.atan(1.0)*4.0
    h = pi_val / (2.0*n)

    ts = np.zeros(n, dtype=np.float64)
    whts = np.zeros(n, dtype=np.float64)

    # initial approximations for second half
    ii = 0
    for i in range(1, n+1):
        if i < (n//2 + 1):
            continue
        ii += 1
        t = (2.0*i - 1.0) * h
        ts[ii-1] = -math.cos(t)

    # start from center, Newton
    x0 = 0.0
    pol, der = legepol(x0, n)
    x1 = ts[0]

    n2 = (n + 1)//2
    pol3, der3 = pol, der

    for kk in range(1, n2+1):
        if ifodd == 1 and kk == 1:
            ts[kk-1] = x0
            if itype > 0:
                whts[kk-1] = der
            x0 = x1
            x1 = ts[kk] if kk < len(ts) else x1
            pol3, der3 = pol, der
            continue

        ifstop = 0
        for _ in range(10):
            hh = x1 - x0
            pol, der = legetayl(pol3, der3, x0, hh, n, k)
            x1 = x1 - pol/der
            if abs(pol) < 1e-12:
                ifstop += 1
            if ifstop == 3:
                break

        ts[kk-1] = x1
        if itype > 0:
            whts[kk-1] = der

        x0 = x1
        x1 = ts[kk] if kk < len(ts) else x1
        pol3, der3 = pol, der

    # mirror roots
    for i in range(n2, 0, -1):
        ts[i-1+half] = ts[i-1]
    for i in range(1, half+1):
        ts[i-1] = -ts[n - i]

    if itype <= 0:
        return ts, whts

    # mirror weights
    for i in range(n2, 0, -1):
        whts[i-1+half] = whts[i-1]
    for i in range(1, half+1):
        whts[i-1] = whts[n - i]

    # final Gauss weights
    for i in range(n):
        tmp = 1.0 - ts[i]*ts[i]
        whts[i] = 2.0 / tmp / (whts[i]*whts[i])

    return ts, whts


def legeexev(x: float, pexp: np.ndarray) -> float:
    # evaluate Legendre expansion with coefficients pexp[0..n]
    n = len(pexp) - 1
    if n < 0:
        return 0.0
    if n == 0:
        return float(pexp[0])

    pjm2 = 1.0
    pjm1 = x
    val = pexp[0]*pjm2 + pexp[1]*pjm1
    for j in range(2, n+1):
        pj = ((2*j - 1)*x*pjm1 - (j - 1)*pjm2) / j
        val += pexp[j]*pj
        pjm2, pjm1 = pjm1, pj
    return float(val)


def legeFDER(x: float, pexp: np.ndarray) -> Tuple[float, float]:
    # returns (val, der)
    n = len(pexp) - 1
    if n < 0:
        return 0.0, 0.0
    if n == 0:
        return float(pexp[0]), 0.0
    if n == 1:
        return float(pexp[0] + pexp[1]*x), float(pexp[1])

    pjm2, pjm1 = 1.0, x
    derjm2, derjm1 = 0.0, 1.0

    val = pexp[0]*pjm2 + pexp[1]*pjm1
    der = pexp[1]

    for j in range(2, n+1):
        pj = ((2*j - 1)*x*pjm1 - (j - 1)*pjm2) / j
        val += pexp[j]*pj

        derj = (2*j - 1)*(pjm1 + x*derjm1) - (j - 1)*derjm2
        derj /= j
        der += pexp[j]*derj

        pjm2, pjm1 = pjm1, pj
        derjm2, derjm1 = derjm1, derj

    return float(val), float(der)


# ============================================================
# 2) Prolate0 implementation (ported from your C++)
#    We only need prolate0_eval(c,x) for |x|<=1, plus minimal
#    extra bits used by initialization.
# ============================================================

def prosinin(c: float, ts: np.ndarray, whts: np.ndarray, fs: np.ndarray, x: float) -> Tuple[float, float]:
    rint = 0.0
    derrint = 0.0
    for i in range(len(ts)):
        diff = x - ts[i]
        # diff should never be 0 for Gauss nodes unless x exactly equals a node
        if diff == 0.0:
            diff = 1e-300
        sin_term = math.sin(c*diff)
        cos_term = math.cos(c*diff)
        rint += whts[i]*fs[i]*sin_term/diff
        derrint += whts[i]*fs[i]/(diff*diff) * (c*diff*cos_term - sin_term)
    return rint, derrint


def prolcoef(rlam: float, k: int, c: float) -> Tuple[float, float, float, float, float, float]:
    d = k*(k-1)
    d = d / (2*k+1)/(2*k-1)
    uk = d

    d = (k+1)*(k+1)
    d = d / (2*k+3)
    d2 = k*k
    d2 = d2 / (2*k-1)
    vk = (d + d2) / (2*k+1)

    d = (k+1)*(k+2)
    d = d / (2*k+1)/(2*k+3)
    wk = d

    alpha = -c*c*uk
    beta  = rlam - k*(k+1) - c*c*vk
    gamma = -c*c*wk

    return uk, vk, wk, alpha, beta, gamma


def prolmatr(n: int, c: float, rlam: float, ifsymm: int, ifodd: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    done = 1.0
    half = done/2.0

    as_ = np.zeros(n, dtype=np.float64)
    bs_ = np.zeros(n, dtype=np.float64)
    cs_ = np.zeros(n, dtype=np.float64)

    k = 0
    if ifodd > 0:
        for k0 in range(1, n+3, 2):
            k += 1
            uk, vk, wk, alpha, beta, gamma = prolcoef(rlam, k0, c)
            as_[k-1] = alpha
            bs_[k-1] = beta
            cs_[k-1] = gamma
            if ifsymm != 0:
                if k0 > 1:
                    as_[k-1] = as_[k-1] / math.sqrt(k0 - 2 + half) * math.sqrt(k0 + half)
                cs_[k-1] = cs_[k-1] * math.sqrt(k0 + half) / math.sqrt(k0 + half + 2)
    else:
        for k0 in range(0, n+3, 2):
            k += 1
            uk, vk, wk, alpha, beta, gamma = prolcoef(rlam, k0, c)
            as_[k-1] = alpha
            bs_[k-1] = beta
            cs_[k-1] = gamma
            if ifsymm != 0:
                if k0 != 0:
                    as_[k-1] = as_[k-1] / math.sqrt(k0 - 2 + half) * math.sqrt(k0 + half)
                cs_[k-1] = cs_[k-1] * math.sqrt(k0 + half) / math.sqrt(k0 + half + 2)

    return as_, bs_, cs_


def prolql1(n: int, d: np.ndarray, e: np.ndarray) -> int:
    # In-place QL for symmetric tridiagonal (ported from your C++)
    ierr = 0
    if n == 1:
        return 0

    # shift e down
    for i in range(1, n):
        e[i-1] = e[i]
    e[n-1] = 0.0

    for l in range(n):
        j = 0
        while True:
            m = None
            for mm in range(l, n-1):
                tst1 = abs(d[mm]) + abs(d[mm+1])
                tst2 = tst1 + abs(e[mm])
                if tst2 == tst1:
                    m = mm
                    break
            if m is None:
                m = n-1

            if m == l:
                break
            if j == 30:
                return l + 1
            j += 1

            g = (d[l+1] - d[l]) / (2.0*e[l])
            r = math.sqrt(g*g + 1.0)
            g = d[m] - d[l] + e[l] / (g + math.copysign(r, g))
            s = 1.0
            c = 1.0
            p = 0.0

            for i in range(m-1, l-1, -1):
                f = s*e[i]
                b = c*e[i]
                r = math.sqrt(f*f + g*g)
                e[i+1] = r
                if r == 0.0:
                    d[i+1] -= p
                    e[m] = 0.0
                    break
                s = f/r
                c = g/r
                g = d[i+1] - p
                r = (d[i] - g)*s + 2.0*c*b
                p = s*r
                d[i+1] = g + p
                g = c*r - b

            if r == 0.0:
                break
            d[l] -= p
            e[l] = g
            e[m] = 0.0

        # insertion sort step
        if l != 0:
            for i in range(l, 0, -1):
                if d[i] >= d[i-1]:
                    break
                d[i], d[i-1] = d[i-1], d[i]

    return ierr


def prolfact(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns (u,v,w) factors; modifies a in-place in C++; here we copy
    n = len(a)
    aa = a.copy()
    u = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    w = np.zeros(n, dtype=np.float64)

    # eliminate down
    for i in range(n-1):
        d = c[i+1] / aa[i]
        aa[i+1] -= b[i]*d
        u[i] = d

    # eliminate up
    for i in range(n-1, 0, -1):
        d = b[i-1] / aa[i]
        v[i] = d

    # scale diagonal
    for i in range(n):
        w[i] = 1.0 / aa[i]

    return u, v, w


def prolsolv(u: np.ndarray, v: np.ndarray, w: np.ndarray, rhs: np.ndarray) -> None:
    n = len(rhs)
    # down
    for i in range(n-1):
        rhs[i+1] -= u[i]*rhs[i]
    # up
    for i in range(n-1, 0, -1):
        rhs[i-1] -= rhs[i]*v[i]
    # scale
    rhs *= w


def prolfun0(n: int, c: float, eps: float) -> Tuple[int, np.ndarray, int, float]:
    # Builds Legendre expansion coefficients for psi0^c on [-1,1]
    ier = 0
    delta = 1.0e-8
    ifsymm = 1
    numit = 4
    rlam = 0.0
    ifodd = -1

    as_, bs_, cs_ = prolmatr(n, c, rlam, ifsymm, ifodd)

    # C++: prolql1(n/2, bs, as)
    d = bs_.copy()
    e = as_.copy()
    ierr = prolql1(n//2, d, e)
    if ierr != 0:
        return 2048, np.array([]), 0, 0.0

    rkhi = -d[n//2 - 1]
    rlam = -d[n//2 - 1] + delta

    xk = np.ones(n, dtype=np.float64)

    as_, bs_, cs_ = prolmatr(n, c, rlam, ifsymm, ifodd)

    # factorize on size n/2 using (bs, cs, as) order as in C++ call prolfact(bs, cs, as, n/2, ...)
    a = bs_[:n//2]
    b = cs_[:n//2]
    cc = as_[:n//2]
    u, v, wfac = prolfact(a, b, cc)

    # inverse iteration
    for _ in range(numit):
        rhs = xk[:n//2].copy()
        prolsolv(u, v, wfac, rhs)
        # normalize
        dnorm = math.sqrt(float(np.dot(rhs, rhs)))
        rhs /= dnorm
        xk[:n//2] = rhs

    # build coefficient list
    nterms = 0
    half = 0.5
    cs_out = np.zeros(n//2, dtype=np.float64)
    for i in range(n//2):
        if abs(xk[i]) > eps:
            nterms = i + 1
        xk[i] *= math.sqrt(i*2.0 + half)
        cs_out[i] = xk[i]

    # interleave with zeros (even terms)
    # C++: xk[j++]=cs[i]; xk[j++]=0; then nterms*=2
    out = []
    for i in range(nterms+1):
        out.append(float(cs_out[i]))
        out.append(0.0)
    nterms *= 2
    return ier, np.array(out, dtype=np.float64), nterms, rkhi


def prolps0i(c: float, lenw: int) -> Tuple[int, np.ndarray, int, float]:
    # returns (ier, coeffs, nterms, rkhi)
    ns_tab = [48, 64, 80, 92, 106, 120, 130, 144, 156, 168,
              178, 190, 202, 214, 224, 236, 248, 258, 268, 280]
    eps = 1e-16
    n = int(c * 3.0)
    n = n // 2
    i = int(c / 10.0)
    if i <= 19:
        n = ns_tab[i]

    ier, coeffs, nterms, rkhi = prolfun0(n, c, eps)
    return ier, coeffs, nterms, rkhi


def prol0ini(c: float, lenw: int = 10000) -> Tuple[int, np.ndarray]:
    """
    Build work array w[] used by prol0eva.
    Layout follows the C++ code.
    """
    w = np.zeros(lenw, dtype=np.float64)
    ier = 0
    thresh = 45.0
    iw = 11  # 1-based in C++; we will keep same numeric values but store in 0-based array
    w[0] = iw + 0.1
    w[8] = thresh

    ier, coeffs, nterms, rkhi = prolps0i(c, lenw - iw)
    if ier != 0:
        return ier, w

    # store coeffs at w[iw-1 : iw-1+len(coeffs)]
    w[iw-1: iw-1+len(coeffs)] = coeffs

    # if c >= thresh, we are done (no outside [-1,1] support prep)
    if c >= thresh - 1e-10:
        w[7] = c
        w[4] = nterms + 0.1
        # keep = nterms+3, ltot ignored here
        return 0, w

    # For c < thresh, build Gaussian quadrature data for outside interval evaluation
    ngauss = nterms * 2
    lw = nterms + 2
    its = iw + lw
    lts = ngauss + 2
    iwhts = its + lts
    lwhts = ngauss + 2
    ifs = iwhts + lwhts
    lfs = ngauss + 2

    keep = ifs + lfs
    if keep >= lenw:
        return 1024, w

    # store pointers (as float with +0.1) consistent with C++ layout
    w[1] = its + 0.1
    w[2] = iwhts + 0.1
    w[3] = ifs + 0.1

    # nodes and weights
    ts, whts = legerts(1, ngauss)
    w[its-1: its-1+ngauss] = ts
    w[iwhts-1: iwhts-1+ngauss] = whts

    # evaluate prolate on nodes using Legendre expansion coeffs
    # coeffs is stored at iw; length >= nterms (interleaved even/odd)
    # C++ uses legeexev at Gaussian nodes with w+iw as pexp and nterms-1
    pexp = w[iw-1: iw-1 + (nterms)]  # nterms coefficients (0..nterms-1)
    fs = np.zeros(ngauss, dtype=np.float64)
    for i in range(ngauss):
        fs[i] = legeexev(ts[i], pexp[:nterms])  # uses coefficients 0..nterms-1

    w[ifs-1: ifs-1+ngauss] = fs

    # eigenvalue rlam via prosinin at x0=0
    f0 = legeexev(0.0, pexp[:nterms])
    rlam, der = prosinin(c, ts, whts, fs, 0.0)
    rlam = rlam / f0

    w[4] = nterms + 0.1
    w[5] = ngauss + 0.1
    w[6] = rlam
    w[7] = c
    return 0, w


def prol0eva(x: float, w: np.ndarray) -> Tuple[float, float]:
    """
    Evaluate psi0^c(x) and derivative using precomputed workarray w[].
    """
    iw = int(w[0])
    its = int(w[1])
    iwhts = int(w[2])
    ifs = int(w[3])

    nterms = int(w[4])
    ngauss = int(w[5])
    rlam = float(w[6])
    c = float(w[7])
    thresh = float(w[8])

    # outside [-1,1] not needed for your usage, but keep behavior
    if abs(x) > 1.0:
        if c >= thresh - 1e-10:
            return 0.0, 0.0
        ts = w[its-1: its-1+ngauss]
        whts = w[iwhts-1: iwhts-1+ngauss]
        fs = w[ifs-1: ifs-1+ngauss]
        psi0, derpsi0 = prosinin(c, ts, whts, fs, x)
        return psi0/rlam, derpsi0/rlam

    # inside [-1,1] use Legendre expansion coefficients
    pexp = w[iw-1: iw-1 + nterms]  # nterms coefficients
    # C++ uses legeFDER(x, psi0, derpsi0, &w[iw-1], nterms-2)
    # Our legeFDER expects full pexp array. Here pexp length is nterms.
    psi0, derpsi0 = legeFDER(x, pexp[:nterms])
    return psi0, derpsi0


# Cache Prolate0 work arrays by c
_prolate_cache: Dict[float, np.ndarray] = {}

def prolate0_eval(c: float, x: float) -> float:
    key = float(f"{c:.12g}")
    w = _prolate_cache.get(key)
    if w is None:
        ier, w = prol0ini(key, lenw=10000)
        if ier != 0:
            raise RuntimeError(f"prol0ini failed for c={c} with ier={ier}")
        _prolate_cache[key] = w
    psi0, _ = prol0eva(float(x), w)
    return float(psi0)


def pswf(c: float, z: float) -> float:
    # exactly as you specified
    if abs(z) > 1.0:
        return 0.0
    return prolate0_eval(c, z) / prolate0_eval(c, 0.0)


# ============================================================
# 3) First-zero machinery (stable for even real windows)
# ============================================================

def next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def make_x_grid_halfopen(w: int, dx: float) -> np.ndarray:
    # half-open grid [-w/2, w/2) avoids duplicated endpoint
    n = int(round(w / dx))
    return (np.arange(n) - 0.5 * n) * dx


def first_zero_of_real_even_ft(phi_support: np.ndarray, dx: float, pad_factor: int) -> float:
    # enforce even symmetry to suppress tiny odd numerical parts
    phi_support = 0.5 * (phi_support + phi_support[::-1])

    M = next_pow2(pad_factor * len(phi_support))
    phi = np.zeros(M, dtype=np.float64)
    start = (M - len(phi_support)) // 2
    phi[start:start + len(phi_support)] = phi_support

    Phi = dx * np.fft.rfft(np.fft.ifftshift(phi))
    omega = 2.0 * np.pi * np.fft.rfftfreq(M, d=dx)
    y = Phi.real.copy()
    if len(y) < 3:
        raise RuntimeError("FFT grid too small.")

    y[0] = y[1]  # avoid omega=0 edge case
    s = np.sign(y)
    cross = np.where(s[:-1] * s[1:] < 0)[0]
    if len(cross) == 0:
        raise RuntimeError("No zero crossing found (increase pad_factor or refine dx).")

    i = int(cross[0])
    w0, w1 = omega[i], omega[i+1]
    f0, f1 = y[i], y[i+1]
    return float(w0 + (w1 - w0) * (-f0) / (f1 - f0))


def kb_beta_from_first_zero(omega_star: float, w: int) -> float:
    # Kaiser-Bessel first-null inversion:
    # omega0 = (2/w)*sqrt(beta^2 + pi^2)  => beta = sqrt((omega*w/2)^2 - pi^2)
    val = (omega_star * w / 2.0) ** 2 - math.pi ** 2
    return math.sqrt(val) if val > 0 else float("nan")


# ============================================================
# 4) Sweep:
#    - beta := beta_fit(sigma, ns)
#    - build PSWF kernel phi(x)=pswf(beta, z) with z=2x/ns in [-1,1)
#    - compute first spectral zero omega*
#    - beta0 := KB beta matching that omega* (analytic inversion)
#    Output: sigma, ns, beta0, beta
# ============================================================

def sweep_sigma_ns(
    sigma_min: float = 1.25,
    sigma_max: float = 2.00,
    sigma_count: int = 16,
    ns_min: int = 2,
    ns_max: int = 16,
    dx: float = 1.0/32.0,
    pad_factor: int = 32,
    out_csv: str = "sigma_ns_beta0_beta.csv",
) -> None:
    sigmas = np.linspace(sigma_min, sigma_max, sigma_count)

    total = (ns_max - ns_min + 1) * len(sigmas)
    k = 0
    t0 = time.time()

    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["sigma", "ns", "beta0", "beta"])

        for ns in range(ns_min, ns_max + 1):
            x = make_x_grid_halfopen(ns, dx)
            z = 2.0 * x / float(ns)  # z in [-1,1)

            for sigma in sigmas:
                k += 1
                beta = beta_fit(float(sigma), int(ns))

                # PSWF window with c = beta, evaluated at z
                # (loop, since pswf() is scalar + uses cached workarray per c)
                phi = np.zeros_like(z, dtype=np.float64)
                for i in range(len(z)):
                    phi[i] = pswf(beta, float(z[i]))

                # first zero of FT, then KB beta0 from it
                omega0 = first_zero_of_real_even_ft(phi, dx, pad_factor)
                beta0 = kb_beta_from_first_zero(omega0, ns)

                wr.writerow([float(sigma), int(ns), float(beta0), float(beta)])

                # progress
                elapsed = time.time() - t0
                pct = 100.0 * k / total
                rate = k / max(1e-12, elapsed)
                eta = (total - k) / max(1e-12, rate)
                print(
                    f"[{k:4d}/{total}] {pct:6.2f}%  "
                    f"sigma={sigma:6.3f}  ns={ns:2d}  "
                    f"beta0={beta0:12.6g}  beta={beta:12.6g}  "
                    f"ETA~{eta:7.1f}s"
                )

    print(f"\nWrote: {out_csv}")


if __name__ == "__main__":
    # If you see instability in omega0/beta0:
    #   - decrease dx to 1/64
    #   - increase pad_factor to 64
    sweep_sigma_ns()
