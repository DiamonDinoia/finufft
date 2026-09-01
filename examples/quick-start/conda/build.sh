#!/usr/bin/env bash
set -eo pipefail

if [[ $# -ne 0 && ($# -ne 1 || $1 != "--gpu") ]]; then
	echo "usage: bash examples/quick-start/conda/build.sh [--gpu]" >&2
	exit 2
fi

conda env create --yes -f examples/quick-start/conda/environment.yml
eval "$(conda shell.bash hook)"
conda activate finufft-build

pip install python/finufft
pip install pytest
pytest python/finufft/test

[[ "${1:-}" == "--gpu" ]] || exit 0

command -v nvcc >/dev/null || conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
pip install python/cufinufft
python -c "import importlib.metadata as m; print('cufinufft', m.version('cufinufft'))"

if nvidia-smi -L >/dev/null 2>&1; then
	pip install cupy-cuda12x
	python -c "import cupy; n = cupy.cuda.runtime.getDeviceCount(); print('cupy sees', n, 'device(s)'); assert n > 0, 'cupy reached no device'"
	pytest --framework=cupy python/cufinufft/tests
else
	echo "no device: run 'pytest --framework=cupy python/cufinufft/tests' on a machine with a GPU"
fi
