# Quick start: using FINUFFT from your own project

One directory per way of getting FINUFFT into a build. The five CMake and
Makefile routes all build the same program, `main.cpp`, which calls
`finufft1d1` and prints `finufft consume OK`; `conda/` builds the Python
package instead. Copy the one that matches how the project gets its
dependencies.

`tools/ci/install-test.sh` builds and runs the five program routes on every pull
request, and the conda one runs beside them: on a GitHub runner for macOS, and on
a Jenkins pod with a card for the GPU package, which needs a device to test.
`docs/install.rst` includes these files themselves rather than copies of them, so
the page cannot drift from what CI runs.

From a FINUFFT build, the three routes that need no install run as tests:

```bash
cmake -S . -B build -G Ninja -DFINUFFT_BUILD_TESTS=ON -DFINUFFT_BUILD_QUICKSTART=ON
ctest --test-dir build -L quickstart --no-tests=error
```

Each test configures, builds and runs one recipe as a separate project, with
this build's backend, linking options and generator, so it rebuilds FINUFFT three
times. Budget around 13 min from cold on six cores with Ninja; a warm ccache
brings that under a minute. `--no-tests=error` is not optional: `ctest` exits 0
when a label matches nothing, so without it a build configured without
`FINUFFT_BUILD_QUICKSTART` reports success.

| directory | how the project gets FINUFFT |
| --- | --- |
| [`find_package/`](find_package) | FINUFFT is installed already, by a package manager or by `cmake --install` |
| [`fetchcontent/`](fetchcontent) | CMake clones and builds FINUFFT as part of the configure step |
| [`cpm/`](cpm) | the same, through [CPM](https://github.com/cpm-cmake/CPM.cmake), which caches the clone between projects |
| [`subdirectory/`](subdirectory) | FINUFFT is a git submodule, added with `add_subdirectory` |
| [`makefile/`](makefile) | no CMake at all: a compiler line against an installed FINUFFT |
| [`conda/`](conda) | the Python package, built from source inside a conda environment |

`cuda/` is the same pair of recipes for `finufft::cufinufft`, the GPU library.

## Installed already

```bash
cd find_package
cmake -S . -B build -DCMAKE_PREFIX_PATH=<install prefix>
cmake --build build && ./build/app
```

`CMAKE_PREFIX_PATH` is only needed when FINUFFT is somewhere CMake does not
already look.

## Built as part of your project

```bash
cd cpm            # or fetchcontent, or subdirectory
cmake -S . -B build
cmake --build build && ./build/app
```

`fetchcontent` and `cpm` clone FINUFFT at the tag named in the `CMakeLists.txt`;
change that tag to pick a version. `subdirectory` builds a checkout that is
already on disk, and both it and `cpm` default to the checkout these examples
live in, which is what makes the three commands above work unchanged. Point them
somewhere else with `-DFINUFFT_SOURCE_DIR=<checkout>`. FINUFFT's own options work here: add
`-DFINUFFT_USE_DUCC0=ON` to build without an external FFTW, or
`-DFINUFFT_USE_OPENMP=OFF` for a single-threaded library.

## Without CMake

```bash
cd makefile
make PREFIX=<install prefix>            # add LIBDIR=<prefix>/lib64 on RHEL-style prefixes
./app
```

This route needs a shared FINUFFT. A static `libfinufft.a` leaves its FFT and
OpenMP dependencies to whoever links it, and the exported CMake target is the
only thing that knows what they are.
