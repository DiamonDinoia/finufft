#!/usr/bin/env python3
"""
Benchmark cuFINUFFT methods across dimensions and types.

Example:
  python devel/benchmark_cufinufft_methods.py --dtype c64
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency
    plt = None

try:
    import cupy as cp
except Exception as exc:  # pragma: no cover - runtime env dependent
    print("Error: cupy is required to run this benchmark.", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import cufinufft
except Exception as exc:  # pragma: no cover - runtime env dependent
    print("Error: cufinufft Python package is required to run this benchmark.", file=sys.stderr)
    print(f"Import error: {exc}", file=sys.stderr)
    sys.exit(1)


@dataclass(frozen=True)
class SizeConfig:
    name: str
    n_modes: tuple[int, ...]
    m: int
    n_tgt: int | None = None


@dataclass(frozen=True)
class BenchCase:
    prec: str
    dim: int
    n_trans: int
    tol: float

    def name_with_m(self, n_modes: tuple[int, ...], m_used: int) -> str:
        n_modes_str = "x".join(str(v) for v in n_modes)
        return f"{n_modes_str}_M{m_used}_T{self.n_trans}_{self.prec}"


SIZE_TABLE: dict[int, dict[str, SizeConfig]] = {
    1: {
        "small": SizeConfig("small", (65536,), m=120_000, n_tgt=120_000),
        "medium": SizeConfig("medium", (131072,), m=300_000, n_tgt=300_000),
        "large": SizeConfig("large", (262144,), m=600_000, n_tgt=600_000),
    },
    2: {
        "small": SizeConfig("small", (256, 256), m=200_000, n_tgt=200_000),
        "medium": SizeConfig("medium", (384, 384), m=400_000, n_tgt=400_000),
        "large": SizeConfig("large", (512, 512), m=600_000, n_tgt=600_000),
    },
    3: {
        "small": SizeConfig("small", (64, 64, 64), m=200_000, n_tgt=200_000),
        "medium": SizeConfig("medium", (96, 96, 96), m=350_000, n_tgt=350_000),
        "large": SizeConfig("large", (128, 128, 128), m=500_000, n_tgt=500_000),
    },
}

# Benchmark cases by dim/precision; sizes are tuned at runtime.
BENCH_CASES: list[BenchCase] = [
    BenchCase("f", 1, 1, 1e-4),
    BenchCase("d", 1, 1, 1e-9),
    BenchCase("f", 2, 1, 1e-5),
    BenchCase("d", 2, 1, 1e-9),
    BenchCase("f", 3, 1, 1e-6),
    BenchCase("d", 3, 1, 1e-7),
]


def _parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def _parse_size_list(value: str) -> list[str]:
    return [v.strip().lower() for v in value.split(",") if v.strip()]


def _cuda_time_seconds(fn) -> float:
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    fn()
    end.record()
    end.synchronize()
    return cp.cuda.get_elapsed_time(start, end) / 1000.0


def _rand_points(dim: int, m: int, dtype: cp.dtype) -> tuple[cp.ndarray, ...]:
    pts = [cp.random.uniform(-math.pi, math.pi, size=m, dtype=dtype)]
    if dim >= 2:
        pts.append(cp.random.uniform(-math.pi, math.pi, size=m, dtype=dtype))
    if dim >= 3:
        pts.append(cp.random.uniform(-math.pi, math.pi, size=m, dtype=dtype))
    return tuple(pts)


def _ensure_complex(arr: cp.ndarray, dtype: cp.dtype) -> cp.ndarray:
    if dtype == cp.complex64:
        real = cp.random.standard_normal(arr.shape, dtype=cp.float32)
        imag = cp.random.standard_normal(arr.shape, dtype=cp.float32)
        return real + 1j * imag
    real = cp.random.standard_normal(arr.shape, dtype=cp.float64)
    imag = cp.random.standard_normal(arr.shape, dtype=cp.float64)
    return real + 1j * imag


def _mem_free_bytes() -> int:
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    return int(free_bytes)


def _benchmark_case(
    dim: int,
    nufft_type: int,
    method: int,
    size_cfg: SizeConfig,
    eps: float,
    n_trans: int,
    dtype: cp.dtype,
    warmup: int,
    repeats: int,
) -> dict[str, float | int | str]:
    n_modes = size_cfg.n_modes
    m = size_cfg.m

    plan_args = {
        "nufft_type": nufft_type,
        "n_modes": n_modes if nufft_type != 3 else dim,
        "n_trans": n_trans,
        "eps": eps,
        "dtype": "complex64" if dtype == cp.complex64 else "complex128",
        "gpu_method": method,
    }

    # Plan (warmup + timed)
    plan = None

    def make_plan():
        nonlocal plan
        plan = cufinufft.Plan(**plan_args)

    if warmup > 0:
        make_plan()
        if plan is not None and plan._plan is not None:
            plan._destroy_plan(plan._plan)
            plan._plan = None
        plan = None

    mem_before_plan = _mem_free_bytes()
    plan_time = _cuda_time_seconds(make_plan)
    mem_after_plan = _mem_free_bytes()

    # Points
    real_dtype = cp.float32 if dtype == cp.complex64 else cp.float64
    x = _rand_points(dim, m, real_dtype)
    src_pts = list(x) + [None] * (3 - dim)
    mem_after_src = _mem_free_bytes()

    if nufft_type == 3:
        n_tgt = size_cfg.n_tgt if size_cfg.n_tgt is not None else m
        s = _rand_points(dim, n_tgt, real_dtype)
        tgt_pts = list(s) + [None] * (3 - dim)
        setpts_args = (*src_pts, *tgt_pts)
    else:
        setpts_args = (*src_pts,)

    def setpts():
        plan.setpts(*setpts_args)

    if warmup > 0:
        setpts()
        cp.cuda.get_current_stream().synchronize()

    setpts_time = _cuda_time_seconds(setpts)
    mem_after_setpts = _mem_free_bytes()

    # Data
    if nufft_type in (1, 3):
        c = _ensure_complex(cp.empty((n_trans, m), dtype=dtype), dtype)
        if nufft_type == 3:
            out = cp.empty((n_trans, n_tgt), dtype=dtype)
        else:
            out = cp.empty((n_trans, *n_modes), dtype=dtype)
    else:
        grid_shape = (n_trans,) + n_modes
        c = _ensure_complex(cp.empty(grid_shape, dtype=dtype), dtype)
        out = cp.empty((n_trans, m), dtype=dtype)
    mem_after_data = _mem_free_bytes()

    def exec_once():
        plan.execute(c, out=out)
        cp.cuda.get_current_stream().synchronize()

    # Warmup
    for _ in range(warmup):
        exec_once()

    exec_times = []
    for _ in range(repeats):
        exec_times.append(_cuda_time_seconds(exec_once))

    if plan is not None and plan._plan is not None:
        plan._destroy_plan(plan._plan)
        plan._plan = None

    exec_total = sum(exec_times)

    mem_used_plan = mem_before_plan - mem_after_plan
    mem_used_setpts = mem_after_src - mem_after_setpts
    mem_used_total = mem_before_plan - mem_after_data

    return {
        "dim": dim,
        "type": nufft_type,
        "method": method,
        "size": size_cfg.name,
        "m": m,
        "n_modes": "x".join(str(v) for v in n_modes),
        "n_trans": n_trans,
        "n_tgt": size_cfg.n_tgt if size_cfg.n_tgt is not None else 0,
        "plan_s": plan_time,
        "setpts_s": setpts_time,
        "exec_s_total": exec_total,
        "exec_s_min": min(exec_times),
        "exec_s_max": max(exec_times),
        "dtype": "complex64" if dtype == cp.complex64 else "complex128",
        "mem_free_before_plan": mem_before_plan,
        "mem_free_after_plan": mem_after_plan,
        "mem_free_after_src": mem_after_src,
        "mem_free_after_setpts": mem_after_setpts,
        "mem_free_after_data": mem_after_data,
        "mem_used_plan": mem_used_plan,
        "mem_used_setpts": mem_used_setpts,
        "mem_used_total": mem_used_total,
    }


def _benchmark_param_case(
    case: ParamCase,
    nufft_type: int,
    method: int,
    warmup: int,
    repeats: int,
    max_m: int,
    auto_m_fraction: float,
    autotune: bool,
    m_fraction: float,
    grid_fraction: float,
    m_used_override: int | None = None,
    n_modes_override: tuple[int, ...] | None = None,
    retry_attempts: int = 5,
    retry_factor: float = 0.7,
) -> dict[str, float | int | str]:
    if n_modes_override is None:
        n_modes = case.n_modes
    else:
        n_modes = n_modes_override

    if m_used_override is None:
        m_used = case.m
    else:
        m_used = m_used_override

    if autotune:
        n_modes = _resolve_n_modes(
            case=case,
            nufft_type=nufft_type,
            grid_fraction=grid_fraction,
        )
        m_used = _resolve_m(
            case=case,
            nufft_type=nufft_type,
            method=method,
            max_m=max_m,
            auto_m_fraction=auto_m_fraction,
            m_fraction=m_fraction,
            n_modes_override=n_modes,
        )
    elif m_used <= 0:
        m_used = _resolve_m(
            case=case,
            nufft_type=nufft_type,
            method=method,
            max_m=max_m,
            auto_m_fraction=auto_m_fraction,
            m_fraction=auto_m_fraction,
            n_modes_override=n_modes,
        )

    dtype = cp.complex64 if case.prec == "f" else cp.complex128
    est_total_bytes = _estimate_total_bytes(
        case.dim,
        nufft_type,
        method,
        case.prec,
        case.n_trans,
        n_modes,
        m_used,
    )

    attempts = max(retry_attempts, 1)
    for attempt in range(attempts):
        size_label = f"{'x'.join(str(v) for v in n_modes)}_M{m_used}_T{case.n_trans}"
        size_cfg = SizeConfig(size_label, n_modes, m_used, n_tgt=m_used)
        try:
            row = _benchmark_case(
                dim=case.dim,
                nufft_type=nufft_type,
                method=method,
                size_cfg=size_cfg,
                eps=case.tol,
                n_trans=case.n_trans if case.n_trans > 0 else 1,
                dtype=dtype,
                warmup=warmup,
                repeats=repeats,
            )
            row["mem_est_total"] = est_total_bytes
            return row
        except (RuntimeError, cp.cuda.memory.OutOfMemoryError) as exc:
            if attempt == attempts - 1 or m_used <= 1:
                raise
            next_m = max(int(m_used * retry_factor), 1)
            print(
                f"Warning: benchmark failed for M={m_used} ({exc}). "
                f"Retrying with M={next_m}...",
                flush=True,
            )
            m_used = next_m


def _case_display_name(case: ParamCase, max_m: int) -> str:
    if max_m > 0:
        m_used = min(case.m, max_m) if case.m > 0 else max_m
    else:
        m_used = case.m
    return case.name_with_m(m_used if m_used > 0 else 0)


def _estimate_bytes_per_m(
    dim: int,
    nufft_type: int,
    prec: str,
    n_trans: int,
    method: int,
) -> int:
    real_bytes = 4 if prec == "f" else 8
    complex_bytes = 8 if prec == "f" else 16
    n_trans = max(n_trans, 1)
    if nufft_type == 3:
        # src + tgt points, plus input/output complex data per point
        point_bytes = 2 * dim * real_bytes
        data_bytes = 2 * n_trans * complex_bytes
    else:
        point_bytes = dim * real_bytes
        data_bytes = n_trans * complex_bytes
    # Indices and bookkeeping per point.
    index_bytes = 8
    if method in (2, 3):
        index_bytes += 8
    base_bytes = point_bytes + data_bytes + index_bytes
    # Apply a safety multiplier for overhead and temporary buffers.
    overhead = 4.5
    if nufft_type == 3 and method in (2, 3):
        overhead = 6.0
    return int(base_bytes * overhead)


def _resolve_m(
    case: ParamCase,
    nufft_type: int,
    method: int,
    max_m: int,
    auto_m_fraction: float,
    m_fraction: float,
    n_modes_override: tuple[int, ...] | None,
) -> int:
    m_used = case.m
    if m_used <= 0:
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        complex_bytes = 8 if case.prec == "f" else 16
        n_trans = max(case.n_trans, 1)
        n_modes = n_modes_override if n_modes_override is not None else case.n_modes
        grid_bytes = 0
        if nufft_type in (1, 2):
            n_modes_prod = 1
            for n in n_modes:
                n_modes_prod *= n
            grid_bytes = n_trans * n_modes_prod * complex_bytes
        safety_free = int(free_bytes * 0.75)
        grid_bytes_est = int(grid_bytes * 1.6)
        leftover_bytes = max(safety_free - grid_bytes_est, 0)
        target_bytes = int(min(leftover_bytes, free_bytes * m_fraction))
        bytes_per_m = _estimate_bytes_per_m(case.dim, nufft_type, case.prec, case.n_trans, method)
        if bytes_per_m <= 0:
            raise ValueError("Invalid bytes-per-M estimate.")
        m_used = max(int(target_bytes // bytes_per_m), 1)
    if max_m > 0:
        m_used = min(m_used, max_m)
    return m_used


def _resolve_n_modes(
    case: ParamCase,
    nufft_type: int,
    grid_fraction: float,
) -> tuple[int, ...]:
    if nufft_type not in (1, 2):
        return case.n_modes
    base = case.base_n_modes if case.base_n_modes is not None else case.n_modes
    free_bytes, _ = cp.cuda.runtime.memGetInfo()
    complex_bytes = 8 if case.prec == "f" else 16
    n_trans = max(case.n_trans, 1)
    target_grid_bytes = int(free_bytes * grid_fraction)
    base_prod = 1
    for n in base:
        base_prod *= n
    if base_prod <= 0:
        raise ValueError("Invalid base n_modes for autotune.")
    target_prod = max(int(target_grid_bytes // (n_trans * complex_bytes)), 1)
    scale = (target_prod / base_prod) ** (1.0 / len(base))
    # Only shrink the grid; avoid inflating sizes beyond the base template.
    scale = min(scale, 1.0)
    tuned = tuple(max(int(n * scale), 1) for n in base)
    return tuned


def _estimate_grid_bytes(n_modes: tuple[int, ...], prec: str, n_trans: int) -> int:
    complex_bytes = 8 if prec == "f" else 16
    n_trans = max(n_trans, 1)
    n_modes_prod = 1
    for n in n_modes:
        n_modes_prod *= n
    return n_trans * n_modes_prod * complex_bytes


def _estimate_total_bytes(
    dim: int,
    nufft_type: int,
    method: int,
    prec: str,
    n_trans: int,
    n_modes: tuple[int, ...],
    m_used: int,
) -> int:
    grid_bytes = _estimate_grid_bytes(n_modes, prec, n_trans) if nufft_type in (1, 2) else 0
    grid_bytes = int(grid_bytes * 1.4)
    bytes_per_m = _estimate_bytes_per_m(dim, nufft_type, prec, n_trans, method)
    return grid_bytes + m_used * bytes_per_m


def _plot_results(results: list[dict[str, float | int | str]], title: str, out_dir: str) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting. Please install it.")

    dims = sorted({int(r["dim"]) for r in results})
    types = sorted({int(r["type"]) for r in results})
    methods = sorted({int(r["method"]) for r in results})
    dtypes = sorted({str(r["dtype"]) for r in results})

    bar_width = 0.5
    colors = {
        "plan_s": "#4c78a8",
        "setpts_s": "#f58518",
        "exec_s_total": "#54a24b",
    }

    index = {
        (int(r["dim"]), int(r["type"]), int(r["method"]), str(r["size"]), str(r["dtype"])): r
        for r in results
    }

    for dtype in dtypes:
        for dim in dims:
            for nufft_type in types:
                subset_sizes = []
                seen = set()
                for r in results:
                    if (
                        int(r["dim"]) == dim
                        and int(r["type"]) == nufft_type
                        and str(r["dtype"]) == dtype
                    ):
                        size_label = str(r["size"])
                        if size_label not in seen:
                            seen.add(size_label)
                            subset_sizes.append(size_label)
                if not subset_sizes:
                    continue
                for size in subset_sizes:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    sample_row = None
                    for method in methods:
                        candidate = index.get((dim, nufft_type, method, size, dtype))
                        if candidate is not None:
                            sample_row = candidate
                            break
                    if sample_row is None:
                        plt.close(fig)
                        continue
                    n_modes = sample_row.get("n_modes", "")
                    m_val = int(sample_row.get("m", 0))
                    n_trans = int(sample_row.get("n_trans", 0))
                    ax.set_title(
                        f"{title}\n{dtype}, dim {dim}, type {nufft_type}, "
                        f"N={n_modes}, M={m_val}, ntrans={n_trans}"
                    )

                    x_positions = list(range(len(methods)))
                    plan_vals = []
                    setpts_vals = []
                    exec_vals = []
                    for method in methods:
                        row = index.get((dim, nufft_type, method, size, dtype))
                        if row is None:
                            plan_vals.append(0.0)
                            setpts_vals.append(0.0)
                            exec_vals.append(0.0)
                            continue
                        plan_vals.append(float(row["plan_s"]))
                        setpts_vals.append(float(row["setpts_s"]))
                        exec_vals.append(float(row["exec_s_total"]))

                    ax.bar(x_positions, plan_vals, width=bar_width, color=colors["plan_s"], label="plan")
                    ax.bar(
                        x_positions,
                        setpts_vals,
                        width=bar_width,
                        bottom=plan_vals,
                        color=colors["setpts_s"],
                        label="setpts",
                    )
                    bottoms = [plan_vals[idx] + setpts_vals[idx] for idx in range(len(plan_vals))]
                    ax.bar(
                        x_positions,
                        exec_vals,
                        width=bar_width,
                        bottom=bottoms,
                        color=colors["exec_s_total"],
                        label="exec",
                    )

                    ax.set_xticks(x_positions)
                    ax.set_xticklabels([str(m) for m in methods])
                    ax.set_xlabel("method")
                    ax.set_ylabel("time (s)")
                    ax.legend(loc="upper right")
                    fig.tight_layout()

                    out_path = (
                        Path(out_dir)
                        / f"cufinufft_benchmarks_{dtype}_dim{dim}_type{nufft_type}_{size}.svg"
                    )
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_path)
                    plt.close(fig)


def _validate_sizes(dim: int, sizes: Iterable[str]) -> list[str]:
    valid = set(SIZE_TABLE[dim].keys())
    out = []
    for size in sizes:
        if size not in valid:
            raise ValueError(f"invalid size '{size}' for dim {dim}. Valid: {sorted(valid)}")
        out.append(size)
    return out


def _iter_cases(dims: Sequence[int], sizes: Sequence[str]):
    for dim in dims:
        for size_name in sizes:
            yield dim, SIZE_TABLE[dim][size_name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", default="1,2,3", help="Comma-separated dims to test")
    parser.add_argument("--types", default="1,2,3", help="Comma-separated NUFFT types")
    parser.add_argument("--methods", default="1,2,3", help="Comma-separated GPU methods")
    parser.add_argument("--sizes", default="medium", help="Comma-separated sizes")
    parser.add_argument("--eps", type=float, default=1e-6, help="Requested tolerance")
    parser.add_argument("--n-trans", type=int, default=1, help="Number of transforms")
    parser.add_argument("--dtype", choices=["c64", "c128"], default="c64")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
    parser.add_argument("--repeats", type=int, default=10, help="Timing repeats")
    parser.add_argument("--csv", default="", help="Optional CSV output path")
    parser.add_argument("--plot", dest="plot", action="store_true",
                        help="Generate stacked bar plots (default on)")
    parser.add_argument("--no-plot", dest="plot", action="store_false",
                        help="Disable plot generation")
    parser.set_defaults(plot=True)
    parser.add_argument("--plot-out", default="benchmarks/cuda",
                        help="Output directory for plots")
    parser.add_argument("--title", default="cufinufft benchmarks", help="Plot title")
    parser.add_argument("--paramlist", action="store_true",
                        help="Use perftest/bench.py ParamList sizes (default)")
    parser.add_argument("--no-paramlist", dest="paramlist", action="store_false",
                        help="Use size table instead of ParamList")
    parser.set_defaults(paramlist=True)
    parser.add_argument("--max-m", type=int, default=0,
                        help="Cap M to avoid VRAM issues (0 to disable)")
    parser.add_argument("--auto-m-fraction", type=float, default=0.7,
                        help="Fraction of leftover VRAM to target when auto-sizing M (legacy)")
    parser.add_argument("--autotune", action="store_true",
                        help="Autotune n_modes and M to match target memory fractions")
    parser.set_defaults(autotune=True)
    parser.add_argument("--m-fraction", type=float, default=0.8,
                        help="Target fraction of free VRAM for M (autotune)")
    parser.add_argument("--grid-fraction", type=float, default=0.1,
                        help="Target fraction of free VRAM for grid (autotune)")
    parser.add_argument("--m-retry-attempts", type=int, default=1,
                        help="Retries with smaller M on failure")
    parser.add_argument("--m-retry-factor", type=float, default=0.7,
                        help="Multiplier for M when retrying after failure")
    parser.add_argument("--param-dims", default="1,2,3",
                        help="Filter ParamList to specific dims")
    parser.add_argument("--param-precs", default="f,d",
                        help="Filter ParamList to specific precisions")

    args = parser.parse_args()

    results = []
    types = _parse_int_list(args.types)
    methods = _parse_int_list(args.methods)

    if args.paramlist:
        param_dims = set(_parse_int_list(args.param_dims))
        param_precs = {p.strip() for p in args.param_precs.split(",") if p.strip()}
        for case in PARAM_LIST:
            if case.dim not in param_dims:
                continue
            if case.prec not in param_precs:
                continue
            for nufft_type in types:
                method_for_tune = 2 if nufft_type == 3 else 2
                n_modes = (
                    _resolve_n_modes(case, nufft_type, args.grid_fraction)
                    if args.autotune
                    else case.n_modes
                )
                m_used = _resolve_m(
                    case,
                    nufft_type,
                    method_for_tune,
                    args.max_m,
                    args.auto_m_fraction,
                    args.m_fraction if args.autotune else args.auto_m_fraction,
                    n_modes,
                )
                size_label = f"{'x'.join(str(v) for v in n_modes)}_M{m_used}_T{case.n_trans}_{case.prec}"
                for method in methods:
                    free_bytes, _ = cp.cuda.runtime.memGetInfo()
                    est_total = _estimate_total_bytes(
                        case.dim,
                        nufft_type,
                        method,
                        case.prec,
                        case.n_trans,
                        n_modes,
                        m_used,
                    )
                    print(
                        f"Mem est: free={free_bytes / (1024**3):.2f}GiB "
                        f"grid={_estimate_grid_bytes(n_modes, case.prec, case.n_trans) / (1024**3):.2f}GiB "
                        f"total≈{est_total / (1024**3):.2f}GiB",
                        flush=True,
                    )
                    print(
                        f"Running dim={case.dim} type={nufft_type} method={method} "
                        f"size={size_label}...",
                        flush=True,
                    )
                    results.append(
                        _benchmark_param_case(
                            case=case,
                            nufft_type=nufft_type,
                            method=method,
                            warmup=args.warmup,
                            repeats=args.repeats,
                            max_m=args.max_m,
                            auto_m_fraction=args.auto_m_fraction,
                            autotune=args.autotune,
                            m_fraction=args.m_fraction,
                            grid_fraction=args.grid_fraction,
                            m_used_override=m_used,
                            n_modes_override=n_modes,
                            retry_attempts=args.m_retry_attempts,
                            retry_factor=args.m_retry_factor,
                        )
                    )
    else:
        dims = _parse_int_list(args.dims)
        sizes = _parse_size_list(args.sizes)
        for dim in dims:
            _validate_sizes(dim, sizes)
        dtype = cp.complex64 if args.dtype == "c64" else cp.complex128
        for dim, size_cfg in _iter_cases(dims, sizes):
            for nufft_type in types:
                for method in methods:
                    print(
                        f"Running dim={dim} type={nufft_type} method={method} size={size_cfg.name}...",
                        flush=True,
                    )
                    row = _benchmark_case(
                        dim=dim,
                        nufft_type=nufft_type,
                        method=method,
                        size_cfg=size_cfg,
                        eps=args.eps,
                        n_trans=args.n_trans,
                        dtype=dtype,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    row["mem_est_total"] = _estimate_total_bytes(
                        dim,
                        nufft_type,
                        method,
                        "f" if dtype == cp.complex64 else "d",
                        args.n_trans,
                        size_cfg.n_modes,
                        size_cfg.m,
                    )
                    results.append(row)

    headers = list(results[0].keys()) if results else []
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(results)

    for row in results:
        print(
            "dim={dim} type={type} method={method} size={size} "
            "m={m} n_modes={n_modes} n_tgt={n_tgt} "
            "plan={plan_s:.4f}s setpts={setpts_s:.4f}s "
            "exec_total={exec_s_total:.4f}s exec_min={exec_s_min:.4f}s exec_max={exec_s_max:.4f}s "
            "mem_est={mem_est_total:.0f}B mem_used={mem_used_total:.0f}B "
            "dtype={dtype}".format(**row)
        )

    if args.plot and results:
        _plot_results(results, args.title, args.plot_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
