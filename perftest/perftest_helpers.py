"""Shared helpers for the perftest CI scripts.

These utilities are imported by both ``perftest_pr_head.py`` (the
PR-vs-master comparison) and ``run_perftest_ci.py`` (the multi-tag
matrix). Keeping them here avoids drift between the two scripts.

The reported timing metric is ``min(ms)``: ``perftest.cpp`` emits only
aggregate statistics (no per-run rows), and ``min`` is the canonical
microbenchmark estimator as it is least sensitive to runner noise on
shared GitHub-hosted hardware.
"""

from __future__ import annotations

import io
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from cpuinfo import get_cpu_info

from perftest_config import NRUNS, Params

METRIC_COLUMN = "min(ms)"

EXTRA_ARGS: list[str] = [
    f"--n_runs={NRUNS}",
    "--sort=1",
    "--upsampfact=0",
    "--kerevalmethod=1",
    "--debug=0",
    "--bandwidth=1.0",
]


def build_command(param: Params, transform: int, binary_path: str) -> list[str]:
    """Build the perftest invocation for a single (param, transform) pair.

    Single-threaded runs are pinned to CPU 0 with ``taskset`` so that
    they are not migrated across cores by the kernel.
    """
    perftest_args = param.args() + EXTRA_ARGS + [f"--type={transform}"]
    if param.threads == 1:
        return ["taskset", "-c", "0", binary_path, "--arg"] + perftest_args
    return [binary_path] + perftest_args


def run_perftest(cmd: list[str]) -> pd.DataFrame:
    """Run a perftest command and parse its CSV output.

    Raises CalledProcessError (with stderr surfaced) on non-zero exit.
    """
    try:
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"perftest invocation failed ({' '.join(cmd)}):\n{exc.stderr}"
        ) from exc

    csv_text = "\n".join(
        line for line in result.stdout.splitlines() if not line.startswith("#")
    )
    return pd.read_csv(io.StringIO(csv_text), sep=",").set_index("event")


def read_cmake_metadata(build_dir: Path) -> dict[str, str]:
    """Extract compiler version and arch flags from a CMakeCache.txt."""
    compiler_version = "NA"
    compiler_flags = "NA"
    cache = Path(build_dir) / "CMakeCache.txt"
    with open(cache, "r") as f:
        for line in f:
            if "CMAKE_CXX_COMPILER:FILEPATH=" in line:
                cxx = line.removeprefix("CMAKE_CXX_COMPILER:FILEPATH=").strip()
                compiler_version = subprocess.run(
                    [cxx, "--version"], capture_output=True, text=True
                ).stdout.split("\n")[0]
            elif "FINUFFT_ARCH_FLAGS:STRING=" in line:
                compiler_flags = line.removeprefix("FINUFFT_ARCH_FLAGS:STRING=").strip()
    return {"compiler_version": compiler_version, "compiler_flags": compiler_flags}


def read_ncores(perftest_bin: Path) -> int:
    """Run ``perftest --debug=2`` and parse the ``opts.nthreads`` line."""
    out = subprocess.run(
        [str(perftest_bin), "--debug=2"],
        capture_output=True,
        text=True,
    ).stdout
    match = re.search(r"opts.nthreads=(\d+)", out)
    if not match:
        raise RuntimeError(
            f"could not parse opts.nthreads from {perftest_bin} --debug=2 output"
        )
    return int(match.group(1))


def cpu_metadata() -> dict[str, Any]:
    """Wrap ``py-cpuinfo`` into the dict shape used by both scripts."""
    info = get_cpu_info()
    return {
        "cpu_name": info["brand_raw"],
        "arch": info["arch"],
        "flags": ", ".join(info["flags"]),
    }
