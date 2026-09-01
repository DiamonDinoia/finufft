#!/bin/bash
set -euo pipefail

rm -rf _build _stage _makestage _consume _fetch _cpm _submod _leak _broken _broken_consume \
	_nofetch _userfftw _deffftw broken.log nofetch.log userfftw.log deffftw.log \
	examples/quick-start/makefile/app

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
	consumer=examples/quick-start/cuda/find_package
	fetch_consumer=examples/quick-start/cuda/fetchcontent
	install_flags=(-DFINUFFT_USE_CUDA=ON -DFINUFFT_USE_CPU=OFF
		-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
else
	consumer=examples/quick-start/find_package
	fetch_consumer=examples/quick-start/fetchcontent
	install_flags=(-DFINUFFT_USE_DUCC0=$ducc -DFINUFFT_STATIC_LINKING=$static
		-DFINUFFT_USE_OPENMP=$openmp)
fi

cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release \
	-DFINUFFT_ENABLE_INSTALL=ON \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _build --config Release
cmake --install _build --prefix "$stage" --config Release

run_app() { # $1 = the consumer's build directory
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
	[[ -f "${targets[0]}" ]] || {
		echo "ERROR: no exported targets file under $1, so the check proves nothing"
		exit 1
	}
	local patterns=(-e "$PWD" -e /usr/ -e /opt/ -e /home/ -e /Users/)
	if command -v cygpath >/dev/null; then
		patterns+=(-e "$(cygpath -m "$PWD")")
	fi
	grep -HF "${patterns[@]}" "${targets[@]}"
}
if build_paths "$stage"; then
	echo "ERROR: a build-machine path leaked into exported finufftTargets"
	exit 1
fi

cp -a "$stage" _leak
leak=(_leak/lib*/cmake/finufft/finufftTargets.cmake)
markers=("/usr/lib/libfftw3.so" "$PWD/libfinufft.a")
if command -v cygpath >/dev/null; then
	markers+=("$(cygpath -m "$PWD")/libfinufft.a")
fi
printf 'set(FINUFFT_LEAK_CONTROL "%s")\n' "${markers[@]}" >>"${leak[0]}"
hits=$(build_paths _leak) || {
	echo "ERROR: the build-machine path check does not fire on an injected leak"
	exit 1
}
for marker in "${markers[@]}"; do
	grep -qF "$marker" <<<"$hits" || {
		echo "ERROR: the build-machine path check does not fire on the injected $marker"
		exit 1
	}
done
rm -rf _leak

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

export LD_LIBRARY_PATH="$stage/lib:$stage/lib64:${LD_LIBRARY_PATH:-}"
export DYLD_LIBRARY_PATH="$stage/lib:${DYLD_LIBRARY_PATH:-}"
run_app _consume

cmake -S "$fetch_consumer" -B _fetch -DCMAKE_BUILD_TYPE=Release \
	-DFETCHCONTENT_SOURCE_DIR_FINUFFT="$PWD" \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _fetch --config Release
run_app _fetch

if [[ "$cuda" == "0" && "$backend" == "ducc" ]]; then
	cpm_cache=()
	if [[ -n "${CPM_SOURCE_CACHE:-}" ]]; then
		case "$CPM_SOURCE_CACHE" in
		/*) cpm_cache=(-DCPM_SOURCE_CACHE="$CPM_SOURCE_CACHE") ;;
		*) cpm_cache=(-DCPM_SOURCE_CACHE="$PWD/$CPM_SOURCE_CACHE") ;;
		esac
	fi
	cmake -S examples/quick-start/cpm -B _cpm -DCMAKE_BUILD_TYPE=Release \
		-DCPM_finufft_SOURCE="$PWD" "${cpm_cache[@]}" \
		-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
		"${install_flags[@]}"
	cmake --build _cpm --config Release
	run_app _cpm
fi

if [[ "$cuda" == "0" && "$backend" == "ducc" ]]; then
	cmake -S examples/quick-start/subdirectory -B _submod -DCMAKE_BUILD_TYPE=Release \
		-DFINUFFT_SOURCE_DIR="$PWD" \
		-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
		"${install_flags[@]}"
	cmake --build _submod --config Release
	run_app _submod
fi

if [[ "$cuda" == "0" && "$linking" == "Shared" && "${RUNNER_OS:-}" != "Windows" ]]; then
	libdir=$stage/lib
	[[ -d "$libdir" ]] || libdir=$stage/lib64
	make -C examples/quick-start/makefile PREFIX="$stage" LIBDIR="$libdir" clean app
	run_app examples/quick-start/makefile
	make -C examples/quick-start/makefile clean
fi

if [[ "$cuda" == "0" && "$backend" == "ducc" && "$linking" == "Shared" && "$(uname -s)" == "Linux" ]]; then
	make_stage="$PWD/_makestage"
	make -j4 install PREFIX="$make_stage" FFT=DUCC OMP="$([[ "$openmp" == "ON" ]] && echo ON || echo OFF)"
	make -C examples/quick-start/makefile PREFIX="$make_stage" LIBDIR="$make_stage/lib" clean app
	run_app examples/quick-start/makefile
	make -C examples/quick-start/makefile clean
	make objclean >/dev/null
fi

[[ "${CONTROLS:-0}" == "1" ]] || exit 0

if [[ "$cuda" == "1" ]]; then
	cp -a _stage _broken
	sed -i.bak 's/CUDA::cudart;CUDA::cufft//' _broken/lib*/cmake/finufft/finufftTargets.cmake
	if cmake -S "$consumer" -B _broken_consume -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_PREFIX_PATH="$PWD/_broken" >broken.log 2>&1 &&
		cmake --build _broken_consume >>broken.log 2>&1; then
		echo "ERROR: consumer built against an empty link interface, so the guard proves nothing"
		exit 1
	fi
	grep -qE "undefined (reference to|symbol)[ :]*.?cuda" broken.log ||
		{
			echo "ERROR: the consumer failed for some other reason than the empty link interface"
			tail -30 broken.log
			exit 1
		}
	exit 0
fi

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
