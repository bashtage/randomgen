.. _change-log:

Change Log
----------

v2.1.0
======
- Fixed a bug in :class:`~randomgen.pcg64.LCG128Mix` that resulted in ``inc``
  not being correctly set when initialized without a user-provided ``inc``.
- Added the :class:`~randomgen.tyche.Tyche` PRNG of Neves and Araujo. Supports
  two variants. One is the original implementation in the 2012 paper. The
  second implementation matches the version in ``OpenRand``.
- Added the :class:`~randomgen.squares.Squares` PRNG of Widynski. Supports
  two variants. The default uses 5 rounds of the middle square algorithm and outputs
  a 64-bit value. If ``variant=32``, then 4 rounds are used but only 32 bits
  returned.
- Added the helper function :func:`~randomgen.squares.generate_keys` for
  :class:`~randomgen.squares.Squares`. This function can be used to pre-generate
  keys for use with :class:`~randomgen.squares.Squares`.
- Refactored the broadcasting helper functions out of ``randomgen.common``
  to ``randomgen.broadcast``. Tests have been added and some edge case bugs
  have been found and fixed.
- Improve test coverage.
- Additional code clean-up.

v2.0.0
======
- Final compatibility with NumPy 2
- Minimum NumPy is now 1.22.3.
- Removed ``"legacy"`` seeding in favor of using :class:`~numpy.random.SeedSequence`.
- Removed the vendored copy of ``SeedSequence``.
- Deprecated using the ``mode`` keyword argument to set the seed mode, since only ``SeedSequences`` are supported.
- Changed ``randomgen.common.BitGenerator`` to inherit from ``numpy.random.BitGenerator`` so that
  numpy will recognize these as ``BitGenerators``.
- Removed C distribution functions that are available in NumPy (see libnpyrandom)`.
- General code cleanup and modernization.

v1.26.1
=======
- Initial compatability with Cython 3 and NumPy 2

v1.26.0
=======
- Fixed a bug that affected the :meth:`randomgen.xoroshiro128.Xoroshiro128.jumped`
  method of :class:`randomgen.xoroshiro128.Xoroshiro128` where the ** version was
  swapped with the standard version.
- Fixed a bug where :class:`numpy.random.SeedSequence` was not copied when advancing
  generators using ``jumped``.
- Small compatibility fixes for change in NumPy.
- Changes the documentation theme to `sphinx-immaterial <https://jbms.github.io/sphinx-immaterial/>`_.
- Added builds for Python 3.11.
- Increased the minimum Python to 3.8.

v1.23.1
=======
- Registered the bit generators included in ``randomgen`` with NumPy
  so that NumPy :class:`~numpy.random.Generator` instances can be pickled
  and unpickled when using a ``randomstate`` bit generator.
- Changed the canonical name of the bit generators to be their fully qualified
  name. For example, :class:`~randomgen.pcg64.PCG64` is not named ``"randomgen.pcg64.PCG64"``
  instead of ``"PCG64"``.  This was done to avoid ambiguity with NumPy's supplied
  bit generators with the same name.

v1.23.0
=======
- Removed ``Generator`` and ``RandomState``.

v1.20.2
=======
- Fixed a bug in :class:`~randomgen.sfc.SFC64` the used the wrong value from the Weyl
  sequence. In the original implementation, the current value is added to the next random
  integer and then incremented. The buggy version was incrementing then adding, and so
  was shifted by one value. This sequence should be similarly random in appearance, but it
  does not match the original specification and so has been changed.
- Added ``mode="numpy"`` support to :class:`~randomgen.pcg64.PCG64`,
  :class:`~randomgen.mt19937.MT19937`, :class:`~randomgen.philox.Philox`, and
  :class:`~randomgen.sfc.SFC64`. When using this mode, the sequence generated is
  guaranteed to match the sequence produced using the NumPy implementations as long as
  a ``randomgen.seed_sequence.SeedSequence`` or :class:`numpy.random.SeedSequence`
  is used with the same initial seed values.
- Added :func:`~randomgen.generator.ExtendedGenerator.random` with support for
  ``dtype="longdouble"`` to produce extended precision random floats.

.. ipython::

   In [1]: import numpy as np

   In [2]: from randomgen import ExtendedGenerator, PCG64

   In [3]: eg = ExtendedGenerator(PCG64(20210501))

   In [4]: eg.random(5, dtype=np.longdouble)


v1.20.1
=======
- Fixed a bug that affects ``standard_gamma`` when
  used with ``out`` and a Fortran contiguous array.
- Added :func:`~randomgen.generator.ExtendedGenerator.multivariate_complex_normal`.
- Added :func:`~randomgen.generator.ExtendedGenerator.standard_wishart` and
  :func:`~randomgen.generator.ExtendedGenerator.wishart` variate generators.

v1.20.0
=======
- Sync upstream changes from NumPy
- Added typing information
- Corrected a buffer access in :class:`~randomgen.threefry.ThreeFry` and
  :class:`~randomgen.philox.Philox`.
- Fixed a bug in :class:`~randomgen.aes.AESCounter` that prevented a small
  number of counter values from being directly set.

v1.19.3
=======
- Future proofed setup against ``setuptools`` and ``distutils`` changes.
- Enhanced documentation for :class:`~randomgen.rdrand.RDRAND`.

v1.19.2
=======
- Corrected :class:`~randomgen.rdrand.RDRAND` to retry on failures with pause
  between retries. Add a parameter ``retry`` which allows the number of retries
  to be set. It defaults to the Intel recommended value of 10. Also sets an
  exception when the number of retries has been exhausted (very unlikely). See
  the :class:`~randomgen.rdrand.RDRAND` docstring with unique considerations
  when using :class:`~randomgen.rdrand.RDRAND` that do not occur with deterministic
  PRNGs.

v1.19.1
=======
- Added :class:`randomgen.romu.Romu` which is among the fastest available bit generators.
- Added :func:`~randomgen.sfc.SFC64.weyl_increments` to simplify generating increments for
  use in parallel applications of :class:`~randomgen.sfc.SFC64`.
- Completed * :ref:`quality-assurance` of all bit generators to at least 4TB.

v1.19.0
=======

- Tested all bit generators out to at least 1TB `using PractRand`_.
- Added :class:`randomgen.pcg64.PCG64DXSM` which is an alias for :class:`randomgen.pcg64.PCG64`
  with ``variant="dxsm"`` and ``mode="sequence"``. This is the 2.0 version of PCG64 and
  will likely become the default bit generator in NumPy in the near future.
- Added :class:`randomgen.efiix64.EFIIX64` which is both fast and high-quality.
- Added :class:`randomgen.sfc.SFC64` which supports generating streams using distinct
  Weyl constants.
- Added a :class:`randomgen.pcg64.LCG128Mix` which supports setting the LCG multiplier,
  changing the output function (including support for user-defined output functions) and
  pre- or post-state update generation.
- Added a :class:`randomgen.lxm.LXM` which generates variates using a mix of two simple,
  but flawed generators: an Xorshift and a 64-bit LCG. This has been
  proposed for including in `in Java`_.
- Added a :class:`randomgen.wrapper.UserBitGenerator` which allows bit generators to be written
  in Python or numba.
- Added :class:`randomgen.generator.ExtendedGenerator` which contains features not in :class:`numpy.random.Generator`.
- Added  support for the ``dxsm`` and ``dxsm-128`` variants of :class:`randomgen.pcg64.PCG64`. The
  ``dxsm`` variant is the official PCG 2.0 generator.
- Added support for broadcasting inputs in :class:`randomgen.generator.ExtendedGenerator.multivariate_normal`.
- Added support for the `++` variant of :class:`randomgen.xoroshiro128.Xoroshiro128`.
- Fixed a bug the produced incorrect results in :func:`~randomgen.mt19937.MT19937.jumped`.
- Fixed multiple bugs in ``Generator`` that were fixed in :class:`numpy.random.Generator`.

v1.18.0
=======
- ``choice`` pulled in upstream performance improvement that
  use a hash set when choosing without replacement and without user-provided probabilities.
- Added support for ``randomgen.seed_sequence.SeedSequence`` (and NumPy's :class:`~numpy.random.SeedSequence`).
- Fixed a bug that affected both ``randomgen.generator.Generator.randint``
  in ``Generator`` and ``randint``
  in  ``RandomState`` when ``high=2**32``.  This value is inbounds for
  a 32-bit unsigned closed interval generator, and so  should have been redirected to
  a 32-bit generator. It  was erroneously sent to the 64-bit path. The random values produced
  are fully random but inefficient. This fix breaks the stream in ``randomgen.generator.Generator``
  is the value for ``high`` is used. The fix restores ``RandomState`` to
  NumPy 1.16 compatibility.
  only affects the output if ``dtype`` is ``'int64'``
- This release brings many breaking changes.  Most of these have been
  implemented using ``DeprecationWarnings``. This has been done to
  bring ``randomgen`` in-line with the API changes of the version
  going into NumPy.
- Two changes that are more abrupt are:

  * The ``.generator`` method of the bit generators raise ``NotImplementedError``
  * The internal structures that is used in C have been renamed.
    The main rename is ``brng_t`` to ``bitgen_t``

- The other key changes are:

  * Rename ``RandomGenerator`` to ``Generator``.
  * Rename ``randint`` to ``integers``.
  * Rename ``random_integers`` to ``integers``.
  * Rename ``random_sample`` to ``random``.
  * Change ``jump`` which operated in-place to
    :meth:`~randomgen.xoshiro256.Xoshiro256.jumped` which
    returns a new ``BitGenerator``.
  * Rename Basic RNG to bit generator, which has been consistently applied
    across the docs and references
- Add the integer-based SIMD-based Fast Mersenne Twister (SFMT) generator
  :class:`~randomgen.sfmt.SFMT`.
- Add the 64-bit Mersenne Twister (MT64) generator :class:`~randomgen.mt64.MT64`.
- Renamed `Xoshiro256StarStar` to :class:`~randomgen.xoshiro256.Xoshiro256`
  and `Xoshiro512StarStar` to :class:`~randomgen.xoshiro512.Xoshiro512`

v1.17.0
=======
- This release was skipped

v1.16.6
=======
- Changed the default jump step size to phi times the period of the generator for
  :class:`~randomgen.pcg32.PCG32` and :class:`~randomgen.pcg64.PCG64`.
- Improved the performance of :class:`~randomgen.pcg64.PCG64` on Windows.
- Improved performance of :func:`~randomgen.dsfmt.DSFMT.jump` and
  :func:`~randomgen.dsfmt.DSFMT.jumped`.
- Improves backward compatibility of ``RandomState``


v1.16.5
=======
- Fixed bugs in ``laplace``, ``gumbel``, ``logseries``, ``normal``,
  ``standard_normal``, ``standard_exponential``, ``exponential``, and ``logistic``
  that could result in ``nan`` values in rare circumstances (about 1 in :math:`10^{53}` draws).
- Added keyword ``closed`` to ``randint``
  which changes sampling from the half-open interval ``[low, high)`` to the closed
  interval ``[low, high]``.
- Fixed a bug in ``random_integers`` that
  could lead to valid values being treated as invalid.

v1.16.4
=======
- Add a fast path for broadcasting ``randint``
  when using ``uint64`` or ``int64``.
- Refactor PCG64 so that it does not rely on Cython conditional compilation.
- Add ``brng`` to access the basic RNG.
- Allow multidimensional arrays in ``choice``.
- Speed-up ``choice`` when not replacing.
  The gains can be very large (1000x or more) when the input array is large but
  the sample size is small.
- Add parameter checks in ``multinomial``.
- Fix an edge-case bug in ``zipf``.
- Allow 0 for sample in ``hypergeometric``.
- Add broadcasting to ``multinomial`` (see
  `NumPy issue 9710 <https://github.com/numpy/numpy/pull/9710>`_)

v1.16.3
=======
- Release fixing Python 2.7 issues

v1.16.2
=======
- Updated Xoroshiro120 to use Author's latest parametrization
- Closely synchronized with the version of randomgen being integrated
  into NumPy, including removing:

  * ``random_raw``, which have been moved to the individual bit generators
  * ``random_uintegers``, which can be replaced with ``randint``.

- Added ``RandomState`` as a clone of NumPy's RandomState.
- Removed ``LegacyGenerator`` since this is no longer needed
- Fixed many small bugs, including in cffi and ctype interfaces

v1.16.1
=======
- Synchronized with upstream changes.
- Fixed a bug in gamma generation if the shape parameters is 0.0.

v1.16.0
=======
- Fixed a bug that affected :class:`~randomgen.dsfmt.DSFMT` when calling
  :func:`~randomgen.dsfmt.DSFMT.jump` or :func:`~randomgen.dsfmt.DSFMT.seed`
  that failed to reset the buffer.  This resulted in up to 381 values from the
  previous state being used before the buffer was refilled at the new state.
- Fixed bugs in :class:`~randomgen.xoshiro512.Xoshiro512`
  and :class:`~randomgen.xorshift1024.Xorshift1024` where the fallback
  entropy initialization used too few bytes. This bug is unlikely to be
  encountered since this path is only encountered if the system random
  number generator fails.
- Synchronized with upstream changes.

v1.15.1
=======
- Added Xoshiro256** and Xoshiro512**, the preferred generators of this class.
- Fixed bug in `jump` method of Random123 generators which did not specify a default value.
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

.. _in Java: https://openjdk.java.net/jeps/356
.. _using PractRand: http://pracrand.sourceforge.net/