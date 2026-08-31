#!/bin/bash

# Install FINUFFT to a staging prefix and consume it the three ways a user does:
# find_package against the install, FetchContent against the sources, and a bare
# compiler line against the installed headers and library.
#
# One script for GitHub and for Jenkins. GitHub runs it on Windows, where no
# Jenkins agent carries a toolchain; Jenkins runs the Linux arms, and is the only
# place the CUDA consumer can be run rather than only linked, because its pods
# have a device.
#
# Environment:
#   LINKING   Static (default) or Shared
#   BACKEND   ducc (default) or fftw
#   CUDA      1 to install and consume cufinufft instead of the CPU library
#   CUDA_ARCH the pod's compute capability, required when CUDA=1
#   OPENMP    ON (default) or OFF, a supported build and a packaging case of its
#             own. A machine whose compiler has no OpenMP runtime, Apple clang
#             without brew libomp, sets it OFF.
#   CONTROLS  1 to also run the two FFTW positive controls (Linux, fftw only)
set -euo pipefail

# Its own leftovers, not the caller's business: the controls below write four
# more trees than the three routes do, and a stale one from the previous arm
# turns a failure analysis into a guessing game.
rm -rf _build _stage _consume _fetch _plain_app _leak _broken _broken_consume \
	_nofetch _userfftw _deffftw broken.log nofetch.log userfftw.log deffftw.log

linking=${LINKING:-Static}
backend=${BACKEND:-ducc}
cuda=${CUDA:-0}
openmp=${OPENMP:-ON}
stage="$PWD/_stage"

static=ON
[[ "$linking" == "Shared" ]] && static=OFF
ducc=ON
[[ "$backend" == "fftw" ]] && ducc=OFF

if [[ "$cuda" == "1" ]]; then
	: "${CUDA_ARCH:?CUDA=1 needs CUDA_ARCH; a default would silently build for the wrong card}"
	consumer=test/cmake_consume/cuda
	# CPU off on purpose: this is the CUDA-only install layout that used to ship
	# cufinufft.h without the headers it includes.
	install_flags=(-DFINUFFT_USE_CUDA=ON -DFINUFFT_USE_CPU=OFF
		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
else
	consumer=test/cmake_consume
	install_flags=(-DFINUFFT_USE_DUCC0=$ducc -DFINUFFT_STATIC_LINKING=$static
		-DFINUFFT_USE_OPENMP=$openmp)
fi

cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release \
	-DFINUFFT_ENABLE_INSTALL=ON \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _build --config Release
cmake --install _build --prefix "$stage" --config Release

# The exported interface must not name a path from the build machine: an
# absolute dependency path is what left the static install of #494 unlinkable,
# and a build-tree path makes the package work only where it was built.
run_app() { # $1 = the consumer's build directory
	# Told apart on purpose: a missing executable is a build that went wrong, and
	# an executable that will not start is a runtime path that went wrong. Both
	# reach bash as 127. Ninja writes app.exe beside the build directory, the
	# Visual Studio generator one level down under the config.
	local app
	for app in "$1/app" "$1/app.exe" "$1/Release/app.exe"; do
		if [[ -x "$app" ]]; then
			"$app"
			return
		fi
	done
	echo "ERROR: $1 built no executable"
	exit 1
}

build_paths() { # $1 = install prefix, prints every offending line
	local targets
	targets=("$1"/lib*/cmake/finufft/finufftTargets*.cmake)
	# An unmatched glob would hand grep a literal path, grep would exit 2, and
	# the `if` below would read that as "clean".
	[[ -f "${targets[0]}" ]] || {
		echo "ERROR: no exported targets file under $1, so the check proves nothing"
		exit 1
	}
	grep -HF -e "$PWD" -e /usr/ -e /opt/ -e /home/ -e /Users/ "${targets[@]}"
}
if build_paths "$stage"; then
	echo "ERROR: a build-machine path leaked into exported finufftTargets"
	exit 1
fi

# The guard on that guard, run wherever the guard runs rather than only under
# CONTROLS: put the shape it hunts for into a copy of the install and require a
# hit. A check that has never fired cannot be told from one that cannot fire.
cp -a "$stage" _leak
sed -i.bak '1i set(FINUFFT_LEAK_CONTROL "/usr/lib/libfftw3.so")' \
	_leak/lib*/cmake/finufft/finufftTargets.cmake
build_paths _leak >/dev/null || {
	echo "ERROR: the build-machine path check does not fire on an injected leak"
	exit 1
}
rm -rf _leak

# The OpenMP flag has to reach the exported interface, not only the build. A
# consumer on a machine that has OpenMP links either way, so an OpenMP-off arm
# that merely builds proves nothing; the export is where a wrong answer shows.
# The two directions are each other's control, checked against install trees for
# Static and Shared, ducc and fftw, before this landed.
if [[ "$cuda" == "0" ]]; then
	exported=OFF
	grep -q "OpenMP::" "$stage"/lib*/cmake/finufft/finufftTargets.cmake && exported=ON
	if [[ "$exported" != "$openmp" ]]; then
		echo "ERROR: built with FINUFFT_USE_OPENMP=$openmp, but the export says $exported"
		exit 1
	fi
fi

cmake -S "$consumer" -B _consume -DCMAKE_BUILD_TYPE=Release \
	-DCMAKE_PREFIX_PATH="$stage" \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded
cmake --build _consume --config Release

# A shared install leaves the loader to find the library. Windows does not read
# these, and puts the DLL beside the executable instead.
export LD_LIBRARY_PATH="$stage/lib:$stage/lib64:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$stage/lib:${DYLD_LIBRARY_PATH:-}"
run_app _consume

# Second route: FetchContent/add_subdirectory, which builds FINUFFT as a
# subproject rather than against an install. It is what the CPM and FetchContent
# recipes in docs/install.rst do, and the path that breaks when a
# top-level-only guard (CTest, docs targets, install rules) is missing. Every
# arm runs it: install_flags reach the subproject too, so the linkage and the
# backend do change what gets built here.
cmake -S "$consumer/fetchcontent" -B _fetch -DCMAKE_BUILD_TYPE=Release \
	-DFINUFFT_SOURCE_DIR="$PWD" \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _fetch --config Release
run_app _fetch

# Third route: no CMake at all. A shared install has to be usable from a plain
# compiler line, which is what a hand-written Makefile, a ctypes load or a Julia
# ccall ends up doing, and it is the only route that reads the installed headers
# and the library without the exported target in between.
#
# Static is excluded on purpose rather than skipped for convenience: a static
# libfinufft leaves its FFT and OpenMP dependencies to the consumer, and the
# exported CMake target is the only thing that knows what they are. Windows is
# excluded because cl takes none of these flags.
if [[ "$cuda" == "0" && "$linking" == "Shared" && "${RUNNER_OS:-}" != "Windows" ]]; then
	libdir=$stage/lib
	[[ -d "$libdir" ]] || libdir=$stage/lib64
	mkdir -p _plain_app
	"${CXX:-c++}" -std=c++17 -O2 test/cmake_consume/main.cpp \
		-I"$stage/include" -L"$libdir" -lfinufft -Wl,-rpath,"$libdir" -o _plain_app/app
	run_app _plain_app
fi

[[ "${CONTROLS:-0}" == "1" ]] || exit 0

if [[ "$cuda" == "1" ]]; then
	# The guard on the guard. Empty the exported link interface in a copy of the
	# install and require the consumer to stop linking: main.cpp calls cudaMalloc
	# and cudaMemcpy itself, so the symbols can only come from CUDA::cudart. A
	# check that has never failed cannot be told from one that cannot fail.
	cp -a _stage _broken
	sed -i 's/CUDA::cudart;CUDA::cufft//' _broken/lib*/cmake/finufft/finufftTargets.cmake
	if cmake -S "$consumer" -B _broken_consume -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_PREFIX_PATH="$PWD/_broken" >broken.log 2>&1 &&
		cmake --build _broken_consume >>broken.log 2>&1; then
		echo "ERROR: consumer built against an empty link interface, so the guard proves nothing"
		exit 1
	fi
	# mold says "undefined symbol: cudaMemcpy", GNU ld "undefined reference to \`cudaMalloc'".
	grep -qE "undefined (reference to|symbol)[ :]*.?cuda" broken.log ||
		{
			echo "ERROR: the consumer failed for some other reason than the empty link interface"
			tail -30 broken.log
			exit 1
		}
	exit 0
fi

# A package manager that forbids downloads (vcpkg passes
# -DFETCHCONTENT_FULLY_DISCONNECTED=ON) used to get an unrelated "install
# TARGETS given target fftw3 which does not exist" at the end of the configure.
# setupFFTW.cmake now says so where it happens; this is the control proving the
# message still appears.
# CPM_SOURCE_CACHE has to go: with a warm cache the disconnected configure
# succeeds, which says nothing about the guard. A developer machine has one set,
# and this control reported exactly that.
if env -u CPM_SOURCE_CACHE cmake -S . -B _nofetch -DFINUFFT_USE_DUCC0=OFF \
	-DFINUFFT_FFTW_LIBRARIES=DOWNLOAD \
	-DFETCHCONTENT_FULLY_DISCONNECTED=ON >nofetch.log 2>&1; then
	echo "ERROR: configure succeeded with fetching disabled"
	exit 1
fi
grep -q "FINUFFT could not fetch FFTW" nofetch.log ||
	{
		echo "ERROR: guard message missing"
		tail -30 nofetch.log
		exit 1
	}

# A hand-supplied FFTW cannot be exported, so a static install leaves the
# consumer to link it. The DEFAULT configure is the negative control: it exports
# its FFTW and must stay silent.
cmake -S . -B _userfftw -DFINUFFT_USE_DUCC0=OFF -DFINUFFT_STATIC_LINKING=ON \
	-DFINUFFT_FFTW_LIBRARIES="fftw3;fftw3f" >userfftw.log 2>&1
grep -q "is supplied by hand" userfftw.log ||
	{
		echo "ERROR: warning missing"
		tail -30 userfftw.log
		exit 1
	}
cmake -S . -B _deffftw -DFINUFFT_USE_DUCC0=OFF -DFINUFFT_STATIC_LINKING=ON \
	>deffftw.log 2>&1
if grep -q "is supplied by hand" deffftw.log; then
	echo "ERROR: DEFAULT warned as well, so the check proves nothing"
	exit 1
fi
