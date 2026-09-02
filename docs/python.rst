Python interface
================

Quick-start examples
--------------------

The easiest way to install is to run::

  pip install finufft

which downloads and installs the latest precompiled binaries from PyPI.
If you would like to compile from source, you can tell ``pip`` to compile the library from source with the option ``--no-binary`` using the command::

  pip install --no-binary finufft finufft

By default, this will use the ``-march=native`` flag when compiling the library, which should result in improved performance.
Note that ``finufft`` has to be specified twice (first as an argument to ``--no-binary`` and second as the package that is to be installed). This option also allows you to switch out the default FFT library (FFTW) for DUCC0 using::

  pip install --no-binary finufft finufft --config-settings=cmake.define.FINUFFT_USE_DUCC0=ON finufft

If you have ``pytest`` installed, you can test it with::

  pytest python/finufft/test

or, without having ``pytest`` you can run the older-style eyeball check::

  python3 python/finufft/test/run_accuracy_tests.py

which should report errors around ``1e-6`` and throughputs around 1-10 million points/sec.
(Please note that the ``finufftpy`` package is obsolete.)
If you would like to compile from source, see :ref:`the Python installation instructions <install-python>`.

Once installed, to calculate a 1D type 1 transform from nonuniform to uniform points, we import ``finufft``, specify the nonuniform points ``x``, their strengths ``c``, and call ``nufft1d1``:

.. code-block:: python

    import numpy as np
    import finufft

.. literalinclude:: ../python/finufft/examples/simple1d1.py
   :language: python
   :start-after: @py_simple1d1_start
   :end-before: @py_simple1d1_end

The input here is a set of complex strengths ``c``, which are used to approximate (1) in :ref:`math`.
That approximation is stored in ``f``, which is indexed from ``-N // 2`` up to ``N // 2 - 1`` (since ``N`` is even; if odd it would be ``-(N - 1) // 2`` up to ``(N - 1) // 2``).
The default tolerance of ``nufft1d1`` is ``1e-6``, though the call above requests the higher accuracy ``eps=1e-9``.
It can be modified further using the ``eps`` argument, for instance ``f = finufft.nufft1d1(x, c, N, eps=1e-12)``.
Note, however, that a lower tolerance (that is, a higher accuracy) results in a slower transform. See ``python/finufft/examples/simple1d1.py`` for the demo code that includes a basic math test (useful to check both the math and the indexing).

On CPU, if ``eps`` is so small that FINUFFT knows the requested accuracy is unattainable,
the Python interface raises ``RuntimeError`` (status ``ier=26``) during plan creation
or ``setpts``. If you want FINUFFT to clamp to the best-achievable accuracy and proceed
instead, pass ``allow_eps_too_small=1``.

For higher dimensions, we specify point locations in more than one dimension, generate a fresh set of complex strengths, and call ``nufft2d1``:

.. literalinclude:: ../python/finufft/examples/simple2d1.py
   :language: python
   :start-after: @py_simple2d1_start
   :end-before: @py_simple2d1_end

See ``python/finufft/examples/simple2d1.py`` for the demo code that includes a basic math test (useful to check both the math and the indexing).

We can also go the other way, from uniform to non-uniform points, using a type 2 transform, calling ``finufft.nufft2d2(x, y, f)`` where ``f`` is a complex array of shape ``(N1, N2)`` holding the input Fourier coefficients.
Now the output is a complex vector of length ``M`` approximating (2) in :ref:`math`, that is the adjoint (but not inverse) of (1). (Note that the default sign in the exponential is negative for type 2 in the Python interface.)

In addition to tolerance ``eps``, we can adjust other options for the transform.
These are listed in :ref:`opts` and are specified as keyword arguments in the Python interface.
For example, to change the mode ordering to FFT style (that is, in each dimension ``Ni = N1`` or ``N2``, the indices go from ``0`` to ``Ni // 2 - 1``, then from ``-Ni // 2`` to ``-1``, since each ``Ni`` is even), we call ``finufft.nufft2d1(x, y, c, (N1, N2), modeord=1)``.

We can also specify a preallocated output array using the ``out`` keyword argument, for instance ``finufft.nufft2d1(x, y, c, out=f)`` where ``f = np.empty((N1, N2), dtype='complex128')``.
In this case, we do not need to specify the output shape since it can be inferred from ``f``.

Note that the above functions are all vectorized, which means that they can take multiple inputs stacked along the first dimension (that is, in row-major order) and process them simultaneously.
This can bring significant speedups for small inputs by avoiding multiple short calls to FINUFFT.
For the 2D type 1 vectorized interface, we would call

.. literalinclude:: ../python/finufft/examples/many2d1.py
   :language: python
   :start-after: @py_many2d1_start
   :end-before: @py_many2d1_end

The output array ``f`` has the shape ``(K, N1, N2)``, as printed above.
See the complete demo in ``python/finufft/examples/many2d1.py``.

More fine-grained control can be obtained using the plan (or `guru`) interface.
Instead of preparing the transform, setting the nonuniform points, and executing the transform all at once, these steps are seperated into different function calls.
This can speed up calculations if multiple transforms are executed for the same grid size, since the same FFTW plan can be reused between calls.
Additionally, if the same nonuniform points are reused between calls, we gain an extra speedup since the points only have to be sorted once.
To perform the call above using the plan interface, we would write

.. literalinclude:: ../python/finufft/examples/guru2d1.py
   :language: python
   :start-after: @py_guru2d1_start
   :end-before: @py_guru2d1_end

See the complete demo in ``python/finufft/examples/guru2d1.py``.
All interfaces support both single and double precision, but for the plan, this must be specified at initialization time using the ``dtype`` argument.
Single precision resolves no finer than about ``max(N_i)`` times the machine epsilon,
so a single-precision plan needs a looser ``eps`` than the same double-precision plan.
The example below asks for ``1e-3`` at 2000 modes:

.. literalinclude:: ../python/finufft/examples/guru2d1f.py
   :language: python
   :start-after: @py_guru2d1f_start
   :end-before: @py_guru2d1f_end

As above, requesting an unattainable ``eps`` now raises ``RuntimeError`` by default.
For exploratory or backwards-compatible workflows that prefer clamp-and-proceed behavior,
pass ``allow_eps_too_small=1`` when constructing the plan or calling the simple interface.

See the complete demo, with math test, in ``python/finufft/examples/guru2d1f.py``.


Full documentation
------------------

.. automodule:: finufft
    :members:
    :member-order: bysource
