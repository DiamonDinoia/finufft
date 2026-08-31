# Quick start with conda

`pip install finufft` gives a prebuilt wheel and needs none of this. Build from
source inside a conda environment when the wheel does not fit: to use the CPU
the machine actually has, to get the GPU package, or because conda already owns
the compilers and libraries on that machine.

From the root of a checkout, with conda on PATH:

```bash
bash examples/quick-start/conda/build.sh          # CPU package, then its tests
bash examples/quick-start/conda/build.sh --gpu    # and the GPU package
```

[`build.sh`](build.sh) is the recipe, not a copy of one: CI runs this file on
every pull request, and the docs include it. Read it for the individual
commands. [`environment.yml`](environment.yml) is the
environment it creates.

Two things in it are worth knowing before you adapt it:

- `pip install python/finufft` names a path, not a package, so pip builds this
  checkout and never consults PyPI. (`--no-binary` is a different mechanism, for
  the package-name form: `pip install --no-binary finufft finufft` would build
  the PyPI source tarball, not this checkout.)
- FFTW comes from conda-forge because the source build looks for one and
  otherwise downloads and builds its own. Pass
  `--config-settings=cmake.define.FINUFFT_USE_DUCC0=ON` to use the FFT that
  ships with FINUFFT instead, and drop fftw from the environment.

Building `cufinufft` needs the CUDA toolkit only; running it needs a device as
well. `build.sh --gpu` always prints the installed version and runs the tests
only when `nvidia-smi` finds a card, so the same file covers a GitHub runner,
which has no device, and a Jenkins pod, which has one.

`--framework` is not optional. `conftest.py` parametrizes every cufinufft test
on it and defaults to an empty list, so a plain `pytest python/cufinufft/tests`
reports every test as skipped and cannot fail. `build.sh` passes
`--framework=cupy` and installs `cupy-cuda12x`, which is a binary wheel;
`tools/ci/cuda-wheel-test.sh` covers pycuda, numba, cupy and torch.

The recipe comes from
[@remy-abergel](https://github.com/flatironinstitute/finufft/discussions/649#discussioncomment-12969277)
(issue #668).
