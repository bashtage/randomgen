.. _change-log:

Change Log
----------
v2.6.0
======

.. note::

   The version number now follows NumPy's, so there were no versions 2.4.0
   or 2.5.0.

.. warning::

   Some streams differ from randomgen 2.3.0 for the same seed and calls. Most
   of these changes fix output that was wrong; see
   :ref:`stream-changes-2-6-0` before upgrading code that must reproduce
   earlier results.

New features
~~~~~~~~~~~~
- Added the scrambled multiply-with-carry generators
  :class:`~randomgen.mwcx.MWC192X`, :class:`~randomgen.mwcx.MWC128X`,
  :class:`~randomgen.mwcx.MWC128X32`, and :class:`~randomgen.mwcx.MWC64X32`
  of Vigna. All support :meth:`~randomgen.mwcx.MWC192X.advance` and
  :meth:`~randomgen.mwcx.MWC192X.jumped`, which take time logarithmic in
  the number of steps or jumps, respectively, and accept negative values
  to move backward.
- Added support for free-threaded Python. The extensions are declared
  free-threading compatible, so importing randomgen does not re-enable the
  GIL. Every bit generator now holds its lock while it reads or sets its
  state, seeds, advances or jumps, so one bit generator can be shared
  between threads. The lock is now always a reentrant
  :class:`threading.RLock`.
- The SIMD switches ``use_aesni`` (:class:`~randomgen.aes.AESCounter`),
  ``use_avx2`` (:class:`~randomgen.blabla.BlaBla`), ``use_simd``
  (:class:`~randomgen.chacha.ChaCha`) and ``use_sse41``
  (:class:`~randomgen.speck128.SPECK128`) now apply to a single instance.
  They used to set a flag shared by the whole process, and creating or
  reseeding any instance turned SIMD back on. The output is the same with
  and without SIMD.
- Added ``advance_step(max_draws)`` to :class:`~randomgen.pcg64.PCG64`,
  :class:`~randomgen.pcg64.PCG64DXSM`, :class:`~randomgen.pcg64.LCG128Mix`
  and :class:`~randomgen.pcg32.PCG32`
  (:meth:`~randomgen.pcg64.PCG64.advance_step`). It returns a step for
  splitting one stream into several, where stream ``i`` is advanced by
  ``i * step`` and makes at most ``max_draws`` draws. The step is the odd
  integer closest to :math:`2^{XX}/\phi`, where :math:`\phi` is the golden
  ratio, for the smallest :math:`XX` with :math:`2^{XX}/\phi` larger than
  ``max_draws``, and is computed with integer arithmetic only. Because it is
  odd, the states of the streams differ as if at random, apart from their
  lowest few bits. Because it is at least ``max_draws``, ``n`` streams do not
  overlap as long as ``n * step`` is no larger than the period.

.. _stream-changes-2-6-0:

Changes to random streams
~~~~~~~~~~~~~~~~~~~~~~~~~
These changes alter values that randomgen 2.3.0 produced for the same seed
and calls.

- ``jumped`` in :class:`~randomgen.pcg64.PCG64`,
  :class:`~randomgen.pcg64.PCG64DXSM` and
  :class:`~randomgen.pcg64.LCG128Mix` now jumps by the odd integer closest to
  :math:`2^{128}/\phi`, where :math:`\phi` is the golden ratio, and
  :class:`~randomgen.pcg32.PCG32` by the odd integer closest to
  :math:`2^{64}/\phi`. The previous steps were even, so ``jumped(2**126)``
  (``jumped(2**63)`` for :class:`~randomgen.pcg32.PCG32`) returned the
  original state. ``jumped`` of :class:`~randomgen.pcg64.PCG64` with
  ``numpy_seed=True`` and of :class:`~randomgen.pcg64.PCG64DXSM` now matches
  NumPy (`#398 <https://github.com/bashtage/randomgen/issues/398>`_).
- :meth:`~randomgen.philox.Philox.advance` and
  :meth:`~randomgen.threefry.ThreeFry.advance` now default to
  ``counter=False``, so ``advance(delta)`` moves the stream by ``delta``
  draws. This change has been announced by a ``FutureWarning`` since
  randomgen 1.18. Pass ``counter=True`` to advance the counter, as the
  previous default did.
- Fixed :meth:`~randomgen.philox.Philox.advance` and
  :meth:`~randomgen.threefry.ThreeFry.advance`. With ``number=2`` and
  ``counter=False``, advancing a newly seeded generator by an even number
  of draws did not move it. With any configuration, steps that reach the
  highest word of the counter, which include every negative step, lost
  that word.
- Fixed :meth:`~randomgen.generator.ExtendedGenerator.wishart` with
  ``method`` ``"svd"``, ``"eigh"`` or ``"cholesky"`` when the factor of
  ``scale`` is not symmetric, which includes every non-diagonal ``scale``.
  The draws had the wrong distribution, with a mean different from
  ``df * scale``. Rank-deficient ``scale`` with ``"cholesky"`` is also fixed.
- :class:`~randomgen.philox.Philox` with ``numpy_seed=True`` now uses
  ``counter``, which was ignored, so it matches :class:`numpy.random.Philox`.
- Fixed :meth:`~randomgen.sfc.SFC64.weyl_increments`. The number of values
  with 30 to 35 non-zero bits overflowed, which includes the default
  ``max_bits=32``, so increments were drawn with the wrong probabilities
  when ``min_bits < max_bits``. A draw at the end of the range could also
  return a value with the wrong number of bits, or loop forever.
- Setting the state of :class:`~randomgen.squares.Squares` now restores the
  buffered 32-bit value (``has_uint32`` and ``uinteger``).
- Setting the state of :class:`~randomgen.pcg64.LCG128Mix` now updates all
  of its parameters (``multiplier``, ``dxsm_multiplier``, ``post`` and the
  output function), which a later ``seed`` used.
- :meth:`~randomgen.generator.ExtendedGenerator.multivariate_normal` and
  :meth:`~randomgen.generator.ExtendedGenerator.multivariate_complex_normal`
  with a size containing 0 now draw nothing and return an empty array.
  With ``size=1`` they return shape ``(1, N)``, as NumPy does, instead of
  ``(N,)``.
- ``random_raw(size, output=False)`` now advances by ``prod(size)`` values.
  It used to advance by ``sum(size)`` when ``size`` was a tuple.

Stricter validation
~~~~~~~~~~~~~~~~~~~
These calls used to run, often with wrong results, and now raise.

- Setting ``state`` with a different configuration, for example a
  :class:`~randomgen.pcg64.PCG64` state with another ``variant`` or a
  :class:`~randomgen.philox.Philox` state with another ``number`` or
  ``width``, raises ``ValueError``. Generators created from the bit
  generator kept using the old configuration. Create a new bit generator
  instead. Unpickling is not affected.
- Setting ``state`` with values out of range, such as a position or buffer
  index, the number of rounds, or an unknown :class:`~randomgen.pcg64.PCG64`
  variant, raises ``ValueError`` instead of using them. So does a
  :class:`~randomgen.squares.Squares` state with an invalid ``key``.
- An unknown ``method`` in
  :meth:`~randomgen.generator.ExtendedGenerator.multivariate_normal`,
  :meth:`~randomgen.generator.ExtendedGenerator.multivariate_complex_normal`
  or :meth:`~randomgen.generator.ExtendedGenerator.wishart` raises
  ``ValueError``. It used to be treated as ``"cholesky"``.
  :meth:`~randomgen.generator.ExtendedGenerator.multivariate_complex_normal`
  does not accept ``"factor"``.
- :meth:`~randomgen.generator.ExtendedGenerator.complex_normal` validates
  scalar parameters as it does arrays, and a ``size`` smaller than the
  broadcast shape of the parameters raises, as in
  :meth:`numpy.random.Generator.normal`. It used to return a partly filled
  array.
- :meth:`~randomgen.generator.ExtendedGenerator.standard_wishart` raises
  when ``df`` is not positive.
- An ``inc`` out of range in :class:`~randomgen.pcg32.PCG32` and
  :class:`~randomgen.pcg64.PCG64` raises ``ValueError``, and
  :class:`~randomgen.pcg64.PCG64` with ``numpy_seed=True`` rejects ``inc``,
  which it used to ignore.
- :class:`~randomgen.wrapper.UserBitGenerator` checks that the functions it
  is given are callable.

Removals
~~~~~~~~
- Removed the in-place ``jump`` method, deprecated since 1.18, from every
  bit generator. Use ``jumped``, which returns a new bit generator.
- Removed ``BitGenerator.generator``, which has only raised
  ``NotImplementedError`` since 1.18.
- Removed ``randomgen.legacy``, which could not be imported since the move
  to meson, the ``mtrand`` stubs, and the internal ``randomgen.compat``
  module.

Other fixes
~~~~~~~~~~~
- :meth:`~randomgen.generator.ExtendedGenerator.complex_normal` accepts
  ``gamma`` and ``relation`` arrays of different shapes that broadcast.
- :meth:`~randomgen.generator.ExtendedGenerator.wishart` accepts a list for
  ``scale`` and ``size``.
- ``PCG64(None, inc=...)`` uses ``inc``, which used to be replaced by a
  random increment.
- :func:`~randomgen.entropy.random_entropy` checks for errors on Windows,
  where it could return uninitialized memory, and retries short reads on
  other platforms. ``random_entropy(0)`` returns an empty array.
- :class:`~randomgen.rdrand.RDRAND` keeps ``retries`` in ``jumped``, no
  longer reads past its buffer after a failure, and raises pending errors
  without printing them.
- :func:`~randomgen.squares.generate_keys` and seeding
  :class:`~randomgen.squares.Squares` are thread safe.
- 64-bit values built from two 32-bit draws now use a defined order. The
  order matches what GCC, Clang and MSVC produced before.
- A failed aligned allocation raises ``MemoryError`` instead of crashing,
  and HC-128 no longer relies on unaligned pointer casts.
- The ``advance`` methods of the PCG generators warn against splitting one
  stream with a large power of two, and explain how to choose an odd step,
  which ``advance_step`` computes
  (`#450 <https://github.com/bashtage/randomgen/issues/450>`_).
- The documentation of ``random(dtype=np.longdouble)`` explains why it gives
  different values across platforms.

Build and packaging
~~~~~~~~~~~~~~~~~~~
- Building requires Cython 3.1 or later.
- NumPy 2.4.0 is excluded: it renamed a class in a way that stops randomgen
  from importing. NumPy 2.4.1 restored the name.
- Extensions are no longer all compiled with AVX2 (``-mavx2`` or
  ``/arch:AVX2``). SIMD instructions are only enabled in the functions that
  check the CPU first, so importing randomgen no longer crashes on x86-64
  CPUs without AVX2.
- The ``disable-sse2`` build option now takes effect, and the new
  ``mwcx-force-portable`` option forces the portable multiplication in the
  MWCX generators.
- Wheels no longer install the C sources, build helpers or examples, and
  the sdist no longer includes CI configuration or test results. The
  license is declared as ``NCSA OR BSD-3-Clause``, and the licenses of the
  included third-party code are part of the package metadata.
- Tested on Python 3.10 to 3.15, including free-threaded 3.14 and 3.15.

v2.3.0
======
- Added the :class:`~randomgen.blabla.BlaBla` PRNG which is based on the Blake 2b
  hash function.  :class:`~randomgen.blabla.BlaBla` is a counter-based PRNG like
  :class:`~randomgen.aes.AESCounter` and supports both :meth:`~randomgen.blabla.BlaBla.advance`
  and :meth:`~randomgen.blabla.BlaBla.jumped`.
- Removed ``mode`` from bit generator initialization. This argument has been deprecated since release 2.0.0.

.. warning::

    This change is backward incompatible. If ``mode`` is essential, you should continue to use legacy
    versions.

- Moved to the build system to meson which improves build time.
- Fixed a breaking change that affected the use of :func:`functools.partial` test suite in
  Python 3.14.

.. note::

   There was no version 2.2.0.

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
- Initial compatibility with Cython 3 and NumPy 2

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