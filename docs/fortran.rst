.. _fort:

Usage from Fortran
==========================

We provide Fortran interfaces that are very similar to those in C/C++.
We deliberately use "legacy" Fortran style (in the `terminology
of FFTW <http://www.fftw.org/fftw3_doc/Calling-FFTW-from-Legacy-Fortran.html>`_), enabling the widest applicability and avoiding the complexity of
later Fortran features.
Namely, we use f77, with two features from f90: dynamic allocation
and derived types. The latter is only needed if options must be
changed from default values.

Simple example
~~~~~~~~~~~~~~

To perform a 1D type 1 transform from ``M`` nonuniform points ``xj``
with strengths ``cj``, to ``N`` output modes whose coefficients will be written
into the ``fk`` array, using 9-digit tolerance, the $+i$ imaginary sign,
and default options, the declarations and call are

.. code-block:: fortran

      integer ier,iflag
      integer*8 N,M
      real*8, allocatable :: xj(:)
      real*8 tol
      complex*16, allocatable :: cj(:),fk(:)
      integer*8, allocatable :: null

 !    (...allocate xj, cj, and fk, and fill xj and cj here...)

      tol = 1.0D-9
      iflag = +1
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,null,ier)

which writes the output to ``fk``, and the status to the integer ``ier``.
A status 0 indicates success, otherwise error codes are
in :ref:`here <error>`.
All available OMP threads are used, unless FINUFFT was built single-thread.
(Note that here the unallocated ``null`` is simply a way to pass
a NULL pointer to our C++ wrapper; another would be ``%val(0_8)``.)
For a minimally complete test code demonstrating the above see
``fortran/examples/simple1d1.f``.

To compile (eg using GCC/linux) and link such a program against the FINUFFT
dynamic (.so) library (which links all dependent libraries)::

  gfortran -I $(FINUFFT)/include simple1d1.f -o simple1d1 -L$(FINUFFT)/lib -lfinufft

where ``$(FINUFFT)`` indicates the top-level FINUFFT directory.
Or, using the static library, one must list dependent libraries::

  gfortran -I $(FINUFFT)/include simple1d1.f -o simple1d1 $(FINUFFT)/lib-static/libfinufft.a -lfftw3 -lfftw3_omp -lgomp -lstdc++
  
Alternatively you may want to compile with ``g++`` and use ``-lgfortran`` at the end of the compile statement instead of ``-lstdc++``.
In Mac OSX, replace ``fftw3_omp`` by ``fftw3_threads``, and if you use
clang, ``-lgomp`` by ``-lomp``. See ``makefile`` and ``make.inc.*``.

.. note ::
 Our simple interface is designed to be a near drop-in replacement for the native f90 `CMCL libraries of Greengard-Lee <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_. The differences are: i) we added a penultimate argument in the list which allows options to be changed, and ii) our normalization differs for type 1 transforms (one divides our output by $M$ to match the CMCL output).

Changing options
~~~~~~~~~~~~~~~~

To choose non-default options in the above example, create an options
derived type, set it to default values, change whichever you wish, and pass
it to FINUFFT, for instance

.. code-block:: fortran

      include 'finufft.fh'
      type(nufft_opts) opts
 
 !    (...declare, allocate, and fill stuff as above...)

      call finufft_default_opts(opts)
      opts%debug = 2
      opts%upsampfac = 1.25d0
      call finufft1d1(M,xj,cj,iflag,tol,N,fk,opts,ier)
 
See ``fortran/examples/simple1d1.f`` for the complete code,
and below for the complete list of Fortran subroutines available,
and more complicated examples.

Summary of Fortran interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The naming of routines is as in C/C++.
Eg, ``finufft2d3`` means 2D transform of type 3.
``finufft2d3many`` means applying 2D transforms of type 3 to a stack of many
strength vectors (vectorized interface).
The guru interface has very similar arguments to its C/C++ version.
Compared to C/C++, all argument lists have ``ier`` appended at the end,
to which the status is written.
Assuming a double-precision build of FINUFFT, these routines and arguments are

.. code-block:: fortran

      include 'finufft.fh'

      integer ier,iflag,ntrans,type,dim
      integer*8 M,N1,N2,N3,Nk
      integer*8 plan,n_modes(3)
      real*8, allocatable :: xj(:),yj(:),zj(:), sk(:),tk(:),uk(:)
      real*8 tol
      complex*16, allocatable :: cj(:), fk(:)
      type(nufft_opts) opts

 !    simple interface   
      call finufft1d1(M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d2(M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d3(M,xj,cj,iflag,tol,Nk,sk,fk,opts,ier)
      call finufft2d1(M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d2(M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d3(M,xj,yj,cj,iflag,tol,Nk,sk,tk,fk,opts,ier)
      call finufft3d1(M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d2(M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d3(M,xj,yj,zj,cj,iflag,tol,Nk,sk,tk,uk,fk,opts,ier)

 !    vectorized interface
      call finufft1d1many(ntrans,M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d2many(ntrans,M,xj,cj,iflag,tol,N1,fk,opts,ier)
      call finufft1d3many(ntrans,M,xj,cj,iflag,tol,Nk,sk,fk,opts,ier)
      call finufft2d1many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d2many(ntrans,M,xj,yj,cj,iflag,tol,N1,N2,fk,opts,ier)
      call finufft2d3many(ntrans,M,xj,yj,cj,iflag,tol,Nk,sk,tk,fk,opts,ier)
      call finufft3d1many(ntrans,M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d2many(ntrans,M,xj,yj,zj,cj,iflag,tol,N1,N2,N3,fk,opts,ier)
      call finufft3d3many(ntrans,M,xj,yj,zj,cj,iflag,tol,Nk,sk,tk,uk,fk,opts,ier)

 !    guru interface
      call finufft_makeplan(type,dim,n_modes,iflag,ntrans,tol,plan,opts,ier)
      call finufft_setpts(plan,M,xj,yj,zj,Nk,sk,yk,uk,ier)
      call finufft_exec(plan,cj,fk,ier)
      call finufft_destroy(plan,ier)

These are defined (from the C++ side) in ``fortran/finufft_f.cpp``.
Currently the single-precision subroutines have the same name, but instead
of linking with ``-lfinufft`` one uses ``-lfinufftf`` (after having built
a single-precision FINUFFT). This means that they
cannot be used together in the same executable.


Code examples
~~~~~~~~~~~~~

The ``fortran/examples`` directory contains the following demos.
Each has a math test to check the correctness of some or all outputs::

  simple1d1.f        - 1D type 1, simple interface, default and various opts
  guru1d1.f          - 1D type 1, guru interface, default and various opts
  nufft1d_demo.f     - 1D types 1,2,3, minimally changed from CMCL demo codes
  nufft2d_demo.f     - 2D "
  nufft3d_demo.f     - 3D "
  nufft2dmany_demo.f - 2D types 1,2,3, vectorized (many strengths) interface
  
The last four here are modified from demos in the
`CMCL NUFFT libraries <http://www.cims.nyu.edu/cmcl/nufft/nufft.html>`_.
The first three of these have been changed only to use FINUFFT,
The final tolerance they request is ``tol=1d-16``. For this case FINUFFT
will report a warning that it cannot achieve it, and gets
merely around $10^{-14}$.
The last four demos require direct summation (slow) reference implementations
of the transforms in ``fortran/directft``, modified from their CMCL
counterparts only to remove the $1/M$ prefactor for type 1 transforms.

.. note ::
 The above demos are double precision. Single-precision versions also exist and have an extra ``f`` after the name, ie, as listed by: ``ls fortran/examples/*f.f``. Remember to use ``-lfinufftf`` and ``type(nufft_optsf)`` for single precision, otherwise it will segfault!

All demos have self-contained example GCC
compilation/linking commands in their comment headers.
For dynamic linking so that execution works from any directory, bake in an
absolute path via the compile flag ``-Wl,-rpath,$(FINUFFT)/lib``.

For authorship and licensing of the Fortran wrappers, see
the directory `README <https://github.com/flatironinstitute/finufft/blob/master/fortran/README>`_