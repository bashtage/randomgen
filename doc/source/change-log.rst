.. _change-log:

Change Log
----------

v1.16.5
=======
- Fixed bugs in :func:`~randomgen.mtrand.RandomState.laplace`,
  :func:`~randomgen.mtrand.RandomState.gumbel`,
  :func:`~randomgen.mtrand.RandomState.logseries`,
  :func:`~randomgen.mtrand.RandomState.normal`,
  :func:`~randomgen.mtrand.RandomState.standard_normal`,
  :func:`~randomgen.mtrand.RandomState.standard_exponential`,
  :func:`~randomgen.mtrand.RandomState.exponential`, and
  :func:`~randomgen.mtrand.RandomState.logistic` that could result in ``nan``
  values in rare circumstances (about 1 in :math:`10^{53}` draws).
- Added keyword ``closed`` to :func:`~randomgen.generator.Generator.randint`
  which changes sampling from the half-open interval ``[low, high)`` to the closed
  interval ``[low, high]``.
- Fixed a bug in :func:`~randomgen.mtrand.RandomState.random_integers` that
  could lead to valid values being treated as invalid.

v1.16.4
=======
- Add a fast path for broadcasting :func:`~randomgen.generator.Generator.randint`
  when using ``uint64`` or ``int64``.
- Refactor PCG64 so that it does not rely on Cython conditional compilation.
- Add :func:`~randomgen.generator.Generator.brng` to access the basic RNG.
- Allow multidimensional arrays in :func:`~randomgen.generator.Generator.choice`.
- Speed-up :func:`~randomgen.generator.Generator.choice` when not replacing.
  The gains can be very large (1000x or more) when the input array is large but
  the sample size is small.
- Add parameter checks in :func:`~randomgen.generator.Generator.multinomial`.
- Fix an edge-case bug in :func:`~randomgen.generator.Generator.zipf`.
- Allow 0 for sample in :func:`~randomgen.generator.Generator.hypergeometric`.
- Add broadcasting to :func:`~randomgen.generator.Generator.multinomial` (see
  `NumPy issue 9710 <https://github.com/numpy/numpy/pull/9710>`_)

v1.16.3
=======
- Release fixing Python 2.7 issues

v1.16.2
=======
- Updated Xoroshiro120 to use AUthor's latest parameterization
- Closely synchronized with the version of randomgen being integrated
  into NumPy, including removing:

    - ``random_raw``, which have been moved to the individual bit generators
    - ``random_uintegers``, which can be replaced with
      :func:`~randomgen.generator.Generator.randint`.

- Added :class:`~randomgen.mtrand.RandomState` as a clone of NumPy's
  RandomState.
- Removed :class:`~randomgen.legacy.LegacyGenerator` since this is no
  longer needed
- Fixed many small bugs, including in cffi and ctype interfaces

v1.16.1
=======
- Synchronized with upstream changes.
- Fixed a bug in gamma generation if the shape parameters is 0.0.

v1.16.0
=======
- Fixed a bug that affected :class:`~randomgen.dsfmt.DSFMT` when calling
  :func:`~randomgen.dsfmt.DSFMT.jump` or :func:`~randomgen.dsfmt.DSFMT.seed`
  that failed to reset the buffer.  This resulted in upto 381 values from the
  previous state being used before the buffer was refilled at the new state.
- Fixed bugs in :class:`~randomgen.xoshiro512starstar.Xoshiro512StarStar`
  and :class:`~randomgen.xorshift1024.Xorshift1024` where the fallback
  entropy initialization used too few bytes. This bug is unlikely to be
  encountered since this path is only encountered if the system random
  number generator fails.
- Synchronized with upstream changes.

v1.15.1
=======
- Added Xoshiro256** and Xoshiro512**, the preferred generators of this class.
- Fixed bug in `jump` method of Random123 generators which did nto specify a default value.
- Added support for generating bounded uniform integers using Lemire's method.
- Synchronized with upstream changes, which requires moving the minimum supported NumPy to 1.13.

v1.15
=====
- Synced empty choice changes
- Synced upstream docstring changes
- Synced upstream changes in permutation
- Synced upstream doc fixes
- Added absolute_import to avoid import noise on Python 2.7
- Add legacy generator which allows NumPy replication
- Improve type handling of integers
- Switch to array-fillers for 0 parameter distribution to improve performance
- Small changes to build on manylinux
- Build wheels using multibuild
