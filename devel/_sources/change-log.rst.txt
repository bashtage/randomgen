.. _change-log:

Change Log
----------

.. container:: admonition danger

  .. raw:: html

      <p class="admonition-title"> Deprecated </p>

  :class:`~randomgen.generator.Generator` and :class:`~randomgen.mtrand.RandomState`
  are **DEPRECATED**. You should be using :class:`numpy.random.Generator` or
  :class:`numpy.random.RandomState` which are better maintained. These will be
  maintained until after NumPy 1.21 (or 2 releases after NumPy 1.19) for users who
  cannot update NumPy.

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
- Fixed multiple bugs in :class:`~randomgen.generator.Generator` that were fixed in :class:`numpy.random.Generator`.

v1.18.0
=======
- :meth:`~randomgen.generator.Generator.choice` pulled in upstream performance improvement that
  use a hash set when choosing without replacement and without user-provided probabilities.
- Added support for :class:`~randomgen.seed_sequence.SeedSequence` (and NumPy's ``SeedSequence``).
- Fixed a bug that affected both :class:`~randomgen.generator.Generator.randint`
  in :class:`~randomgen.generator.Generator` and :meth:`~randomgen.mtrand.RandomState.randint`
  in  :class:`~randomgen.mtrand.RandomState` when ``high=2**32``.  This value is inbounds for
  a 32-bit unsigned closed interval generator, and so  should have been redirected to
  a 32-bit generator. It  was erroneously sent to the 64-bit path. The random values produced
  are fully random but inefficient. This fix breaks the stream in :class:`~randomgen.generator.Generator
  is the value for ``high`` is used. The fix restores :class:`~randomgen.mtrand.RandomState` to
  NumPy 1.16 compatibility.
  only affects the output if ``dtype`` is ``'int64'``
- This release brings many breaking changes.  Most of these have been
  implemented using ``DeprecationWarnings``. This has been done to
  bring ``randomgen`` in-line with the API changes of the version
  going into NumPy.
- Two changes that are more abrupt are:

  * The ``.generator`` method of the bit generators raise
    ``NotImplementedError``
  * The internal structures that is used in C have been renamed.
    The main rename is ``brng_t`` to ``bitgen_t``

- The other key changes are:

  * Rename ``RandomGenerator`` to :class:`~randomgen.generator.Generator`.
  * Rename :meth:`~randomgen.generator.Generator.randint` to
    :meth:`~randomgen.generator.Generator.integers`.
  * Rename :meth:`~randomgen.generator.Generator.random_integers` to
    :meth:`~randomgen.generator.Generator.integers`.
  * Rename :meth:`~randomgen.generator.Generator.random_sample`
    to :meth:`~randomgen.generator.Generator.random`.
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
- Improves backward compatibility of :class:`~randomgen.mtrand.RandomState`


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
- Updated Xoroshiro120 to use Author's latest parametrization
- Closely synchronized with the version of randomgen being integrated
  into NumPy, including removing:

  * ``random_raw``, which have been moved to the individual bit generators
  * ``random_uintegers``, which can be replaced with
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