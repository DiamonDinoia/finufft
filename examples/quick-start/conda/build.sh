#!/usr/bin/env bash
# Quick start with conda: build FINUFFT's Python package from source.
#
#   bash examples/quick-start/conda/build.sh          # CPU
#   bash examples/quick-start/conda/build.sh --gpu    # CPU, then the GPU package
#
# Run it from the root of a FINUFFT checkout, with conda already on PATH. CI runs
# this file, so it is the recipe rather than a copy of one.
# No -u: conda re-activates the environment after an install, and its own
# activation scripts read unset variables. The CUDA toolkit's
# cuda-nvcc_activate.sh dies on NVCC_PREPEND_FLAGS under set -u.
set -eo pipefail

# --gpu is the only argument; anything else dies before conda does any work,
# or a typo would quietly read as a CPU-only run.
if [[ $# -ne 0 && ($# -ne 1 || $1 != "--gpu") ]]; then
	echo "usage: bash examples/quick-start/conda/build.sh [--gpu]" >&2
	exit 2
fi

# --yes so a second run replaces the environment instead of stopping on it.
conda env create --yes -f examples/quick-start/conda/environment.yml
eval "$(conda shell.bash hook)"
conda activate finufft-build

# A path, not a name, so pip builds this checkout and never consults PyPI. The
# compilers, FFTW and OpenMP runtime all come from the environment above.
pip install python/finufft
pip install pytest
pytest python/finufft/test

[[ "${1:-}" == "--gpu" ]] || exit 0

# The GPU package needs a CUDA toolkit, in a version the driver on this machine
# supports. NVIDIA's own channel carries a complete one; a machine that already
# has nvcc needs none of it, which is several gigabytes not downloaded.
command -v nvcc >/dev/null || conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
pip install python/cufinufft
# The installed version, not an import: importing loads the CUDA runtime, and
# building needs the toolkit while running needs a device.
python -c "import importlib.metadata as m; print('cufinufft', m.version('cufinufft'))"

# The tests need a card. Jenkins has one and runs them; a GitHub runner has the
# toolkit and no device, and stops at the line above.
#
# --framework is not optional: conftest.py parametrizes every test on it and
# defaults to an empty list, so plain "pytest tests" reports 2932 skips and no
# failure. cupy is the framework that installs as a binary wheel; the others
# tools/ci/cuda-wheel-test.sh covers need a compile or a torch index.
if nvidia-smi -L >/dev/null 2>&1; then
	pip install cupy-cuda12x
	# A framework that imports but cannot reach the card would send every
	# test back to skipped, and a run of nothing but skips still exits 0. The
	# assert is the point: a count of zero has to stop the script.
	python -c "import cupy; n = cupy.cuda.runtime.getDeviceCount(); print('cupy sees', n, 'device(s)'); assert n > 0, 'cupy reached no device'"
	pytest --framework=cupy python/cufinufft/tests
else
	echo "no device: run 'pytest --framework=cupy python/cufinufft/tests' on a machine with a GPU"
fi
