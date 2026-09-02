#!/bin/bash

# Build cufinufft for the card this pod holds, then run its tests.
# CUDA_ARCH comes from the card itself, so no list of architectures can go
# stale against the hardware.
set -euxo pipefail

nvidia-smi
nvcc --version
g++ --version

# The card reports its own compute capability: "8.6" for sm_86, so strip the dot.
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d .)}"

cmake -G Ninja -B build . -DFINUFFT_USE_CUDA=ON \
	-DFINUFFT_USE_CPU=OFF \
	-DFINUFFT_BUILD_TESTS=ON \
	-DFINUFFT_BUILD_EXAMPLES=ON \
	-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
	-DBUILD_TESTING=ON \
	-DFINUFFT_STATIC_LINKING=OFF
cmake --build build -j "${PARALLEL:-8}"

ctest --test-dir build --output-on-failure --no-tests=error
