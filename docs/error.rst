.. _error:

Error (status) codes
====================

In all FINUFFT interfaces, the returned value ``ier`` is a status indicator.
It is ``0`` if successful, otherwise it is one of the codes below, shared by the
CPU and GPU versions. Codes marked ``[DEPRECATED]`` are no longer returned and
are kept only so the ABI does not change:

.. literalinclude:: ../include/finufft_errors.h
   :language: c
   :start-after: @error_codes_start
   :end-before: @error_codes_end

For any nonzero value of ``ier`` the transform may not have been performed and the output should not be trusted. However, we hope that the value of ``ier`` will help to narrow down the problem.

.. note::
   On CPU, prior to v2.6.0, ``ier=1`` was a warning that still completed the transform at reduced accuracy. The default CPU behavior is now a hard error (``ier=26``). Setting ``opts.allow_eps_too_small=1`` clamps the requested tolerance to machine epsilon and allows the transform to proceed with no warning. GPU behavior is unchanged for now.

FINUFFT sometimes also sends error text to ``stderr`` if it detects faulty input parameters. Please check your terminal output.

If you are getting error codes, please reread the documentation
for your language, then see our :ref:`troubleshooting advice <trouble>`.


Large internal arrays
-----------------------

In case your input parameters demand the allocation of very large arrays, an
internal check is done to see if their size exceeds a rather generous internal
limit, set in ``include/finufft/plan.hpp`` as ``MAX_NF``. The current value in the source code is
``1e12``, which corresponds to about 10TB for double precision.
Allocations beyond this cause a graceful exit with error code ``2`` as above.
Such a large allocation can be due to enormous ``N`` (in types 1 or 2), or ``M``,
but also large values of the space-bandwidth product (loosely, range of :math:`\mathbf{x}_j` points times range of :math:`\mathbf{k}_j` points) for type 3 transforms; see Remark 5 in :ref:`reference FIN <refs>`.
If you have a large-RAM machine and want to exceed the above hard-coded limit, you will need
to edit ``plan.hpp`` and recompile.

Similar sanity checks are done on the numbers of nonuniform points, and it is
(barely) conceivable that you could want to
increase ``MAX_NU_PTS`` beyond its current value
of ``1e14`` in ``plan.hpp``, and recompile.

All internal memory allocations performed by FINUFFT are checked for success, and error code 11 (general internal allocation failure) will be returned in case of an error.
As a consequence, there should be no segmentation faults due to failing memory allocations; so if you observe one, it's very likely a genuine bug in the application.
