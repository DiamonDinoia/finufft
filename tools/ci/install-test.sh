#!/bin/bash
set -euo pipefail

rm -rf _build _stage _makestage _consume _fetch _cpm _submod _c _fortran _pkgconfig _leak _broken _broken_consume \
	_nofetch _userfftw _deffftw _abslibdir _migrate2d1 _nfft2d1 broken.log nofetch.log userfftw.log deffftw.log abslibdir.log \
	examples/quick-start/makefile/app examples/quick-start/pkgconfig/app

linking=${LINKING:-Static}
backend=${BACKEND:-ducc}
cuda=${CUDA:-0}
openmp=${OPENMP:-ON}
fortran=0
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
	# The Fortran route consumes finufft.fh and the shims that only this option builds.
	# Linux only: elsewhere gfortran belongs to a different toolchain than the C++
	# compiler that built the install, and the mix is the user's to sort out.
	if [[ "$(uname -s)" == "Linux" ]] && command -v gfortran >/dev/null; then
		fortran=1
		install_flags+=(-DFINUFFT_BUILD_FORTRAN=ON)
	fi
fi

cmake -S . -B _build -DCMAKE_BUILD_TYPE=Release \
	-DFINUFFT_ENABLE_INSTALL=ON \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _build --config Release
cmake --install _build --prefix "$stage" --config Release

# Plain strings, not arrays: macOS runners still ship bash 3.2, where an empty array
# under `set -u` is an error.
routes=
skipped=
skip() { # $1 = route, $2 = why it could not run here
	skipped="$skipped $1"
	if [[ "$cuda" == "1" ]]; then
		echo "SKIP $1: CPU route, this arm builds CUDA only"
	else
		echo "SKIP $1: $2"
	fi
}

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
routes="$routes find_package"

# The C API from C, not C++: MSVC has no C99 `double complex`, the same limit that
# gates examples/simple1d1c.c.
if [[ "$cuda" == "0" && "${RUNNER_OS:-}" != "Windows" ]]; then
	cmake -S examples/quick-start/c -B _c -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_PREFIX_PATH="$stage"
	cmake --build _c --config Release
	run_app _c
	routes="$routes c"
else
	skip c "MSVC has no C99 double complex"
fi

if [[ "$cuda" == "0" && "$fortran" == "1" ]]; then
	cmake -S examples/quick-start/fortran -B _fortran -DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_PREFIX_PATH="$stage"
	cmake --build _fortran --config Release
	run_app _fortran
	routes="$routes fortran"
else
	skip fortran "no gfortran from the toolchain that built the install"
fi

pcdir=$stage/lib/pkgconfig
[[ -d "$pcdir" ]] || pcdir=$stage/lib64/pkgconfig
pcstatic=
[[ "$linking" == "Static" ]] && pcstatic=--static

# The recipe is a POSIX compiler line, so it needs a Unix-like driver. On the Windows
# runners `command -v pkg-config` also finds Strawberry Perl's copy, which aborts on
# every invocation, so the route asks it for its version before believing it.
if [[ "$cuda" == "0" && "${RUNNER_OS:-}" != "Windows" ]] && pkg-config --version >/dev/null 2>&1; then
	[[ -f "$pcdir/finufft.pc" ]] || {
		echo "ERROR: the install shipped no finufft.pc, so the pkg-config route proves nothing"
		exit 1
	}
	PKG_CONFIG_PATH="$pcdir" make -C examples/quick-start/pkgconfig STATIC="$pcstatic" clean app
	run_app examples/quick-start/pkgconfig
	make -C examples/quick-start/pkgconfig clean
	routes="$routes pkgconfig"
else
	skip pkgconfig "no working pkg-config, or a compiler line this recipe does not fit"
fi

# docs/nfft_migr.rst embeds both tutorial C codes whole, so CI compiles and runs them
# or the embedded text is a claim with no check. Once per platform: neither code varies
# with the FFT backend or the link mode.
if [[ "$cuda" == "0" && "$backend" == "ducc" && "${RUNNER_OS:-}" != "Windows" ]] &&
	pkg-config --version >/dev/null 2>&1; then
	# shellcheck disable=SC2046 # the flags must word-split
	cc tutorial/migrate2d1_test.c -o _migrate2d1 \
		$(PKG_CONFIG_PATH="$pcdir" pkg-config $pcstatic --cflags --libs finufft) -lm
	./_migrate2d1
	routes="$routes tutorial"
	# nfft2d1_test.c needs NFFT3, which only the image that installs libnfft3-dev has.
	if echo '#include <nfft3.h>' | cc -E -x c - >/dev/null 2>&1; then
		cc tutorial/nfft2d1_test.c -o _nfft2d1 -lnfft3 -lfftw3 -lm
		./_nfft2d1
		routes="$routes nfft"
	else
		skip nfft "no nfft3.h, so this image installs no libnfft3-dev"
	fi
else
	skip tutorial "run once per platform, on the ducc backend, where pkg-config works"
fi

cmake -S "$fetch_consumer" -B _fetch -DCMAKE_BUILD_TYPE=Release \
	-DFETCHCONTENT_SOURCE_DIR_FINUFFT="$PWD" \
	-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
	"${install_flags[@]}"
cmake --build _fetch --config Release
run_app _fetch
routes="$routes fetchcontent"

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
	routes="$routes cpm"
else
	skip cpm "run once per platform, on the ducc backend"
fi

if [[ "$cuda" == "0" && "$backend" == "ducc" ]]; then
	cmake -S examples/quick-start/subdirectory -B _submod -DCMAKE_BUILD_TYPE=Release \
		-DFINUFFT_SOURCE_DIR="$PWD" \
		-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=Embedded \
		"${install_flags[@]}"
	cmake --build _submod --config Release
	run_app _submod
	routes="$routes subdirectory"
else
	skip subdirectory "run once per platform, on the ducc backend"
fi

if [[ "$cuda" == "0" && "$linking" == "Shared" && "${RUNNER_OS:-}" != "Windows" ]]; then
	libdir=$stage/lib
	[[ -d "$libdir" ]] || libdir=$stage/lib64
	make -C examples/quick-start/makefile PREFIX="$stage" LIBDIR="$libdir" clean app
	run_app examples/quick-start/makefile
	make -C examples/quick-start/makefile clean
	routes="$routes compiler-line"
else
	skip compiler-line "needs a shared libfinufft and a POSIX compiler driver"
fi

if [[ "$cuda" == "0" && "$backend" == "ducc" && "$linking" == "Shared" && "$(uname -s)" == "Linux" ]]; then
	make_stage="$PWD/_makestage"
	make -j4 install PREFIX="$make_stage" FFT=DUCC OMP="$([[ "$openmp" == "ON" ]] && echo ON || echo OFF)"
	make -C examples/quick-start/makefile PREFIX="$make_stage" LIBDIR="$make_stage/lib" clean app
	run_app examples/quick-start/makefile
	make -C examples/quick-start/makefile clean
	make objclean >/dev/null
	routes="$routes gnu-make-install"
else
	skip gnu-make-install "the GNU makefile builds shared with DUCC0 on Linux"
fi

echo "ROUTES RUN:$routes"
[[ -z "$skipped" ]] || echo "ROUTES SKIPPED:$skipped"

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
	echo "CONTROL ok: an emptied CUDA link interface breaks the consumer"
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
echo "CONTROL ok: a disconnected FFTW fetch fails with the FINUFFT message"

cmake -S . -B _userfftw -DFINUFFT_USE_DUCC0=OFF -DFINUFFT_STATIC_LINKING=ON \
	-DFINUFFT_FFTW_LIBRARIES="fftw3;fftw3f" >userfftw.log 2>&1
grep -q "is supplied by hand" userfftw.log ||
	{
		echo "ERROR: warning missing"
		tail -30 userfftw.log
		exit 1
	}
grep -qE -- '-lfftw3( |$)' _userfftw/finufft.pc ||
	{
		echo "ERROR: a hand-supplied FFTW is missing from the static pkg-config line"
		grep Libs _userfftw/finufft.pc
		exit 1
	}
echo "CONTROL ok: a hand-supplied FFTW reaches Libs.private"
cmake -S . -B _deffftw -DFINUFFT_USE_DUCC0=OFF -DFINUFFT_STATIC_LINKING=ON \
	>deffftw.log 2>&1
if grep -q "is supplied by hand" deffftw.log; then
	echo "ERROR: DEFAULT warned as well, so the check proves nothing"
	exit 1
fi
echo "CONTROL ok: the hand-supplied warning fires only when FFTW is hand-supplied"

# An absolute libdir puts the .pc outside the prefix, so the relative ${pcfiledir} walk
# cannot reach it and the file has to carry the prefix itself.
cmake -S . -B _abslibdir -DCMAKE_INSTALL_LIBDIR=/opt/finufft/lib >abslibdir.log 2>&1
grep -qE '^prefix=/' _abslibdir/finufft.pc ||
	{
		echo "ERROR: an absolute libdir left the .pc prefix unresolvable"
		grep -E '^(prefix|libdir|includedir)=' _abslibdir/finufft.pc
		exit 1
	}
echo "CONTROL ok: an absolute libdir still yields a resolvable prefix"
