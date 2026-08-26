#!/usr/bin/env bash
# Builds and runs the modern C++ interface tests for the CPU library:
# test/cpp_interface (functional checks against direct sums, plus misuse
# cases that must throw finufft::error) and the migrated examples. The GPU
# twin (test/cuda/cufinufft_cpp_interface.cu) rides the regular CUDA pods
# through the main ctest run.
# Honor PARALLEL for the job count, defaulting to 8.
set -euo pipefail

BD=build-cpp-interface

cmake -S . -B "$BD" \
	-DCMAKE_BUILD_TYPE=Release \
	-DFINUFFT_BUILD_TESTS=ON \
	-DBUILD_TESTING=ON \
	-DFINUFFT_BUILD_EXAMPLES=ON

cmake --build "$BD" -j "${PARALLEL:-8}"

ctest --test-dir "$BD" --output-on-failure -R 'cpp_interface|example_'
