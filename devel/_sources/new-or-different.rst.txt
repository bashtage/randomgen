.. _new-or-different:

What's New or Different
-----------------------

Differences from NumPy (1.17+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* :class:`~randomgen.wrapper.UserBitGenerator` allows bit generators to be
  written in Python (slow, suitable for experiments and testing) or numba
  (fast, similar speed to compiled C). See `the demonstration notebook`_ for
  examples.
* :class:`~randomgen.pcg64.PCG64` supports additional variants of PCG64, including
  the PCG4 2.0 variant (`"cm-dxsm"`).
* :class:`~randomgen.sfc.SFC64` supports optional Weyl sequence increments other
  than 1 which is the fixed increment in :class:`numpy.random.SFC64`.
* :func:`~randomgen.entropy.random_entropy` provides access to the system
  source of randomness that is used in cryptographic applications (e.g.,
  ``/dev/urandom`` on Unix).
* Support broadcasting when producing multivariate Gaussian values
  (:meth:`~randomgen.generator.ExtendedGenerator.multivariate_normal`)
* Simulate from the complex normal distribution
  (:meth:`~randomgen.generator.ExtendedGenerator.complex_normal`)
* Direct access to unsigned integers is provided by
  (:meth:`~randomgen.generator.ExtendedGenerator.uintegers`)
* A wider range of bit generators:

  * Chaotic mappings

    * :class:`~randomgen.jsf.JSF` (32 and 64-bit variants)
    * :class:`~randomgen.sfc.SFC64`

  * Cryptographic Cipher-based:

    * :class:`~randomgen.aes.AESCounter`
    * :class:`~randomgen.chacha.ChaCha`
    * :class:`~randomgen.hc128.HC128`
    * :class:`~randomgen.philox.Philox` (limited version in NumPy)
    * :class:`~randomgen.speck128.SPECK128`
    * :class:`~randomgen.threefry.ThreeFry`

  * Hardware-based:

    * :class:`~randomgen.rdrand.RDRAND`

  * Mersenne Twisters

    * :class:`~randomgen.dsfmt.DSFMT`
    * :class:`~randomgen.mt64.MT64`
    * :class:`~randomgen.mt19937.MT19937` (in NumPy)
    * :class:`~randomgen.sfmt.SFMT`

  * Permuted Congruential Generators

    * :class:`~randomgen.pcg32.PCG32`
    * :class:`~randomgen.pcg64.PCG64` (limited version in NumPy)
    * :class:`~randomgen.pcg64.LCG128Mix` (limited version in NumPy)

  * Shift/rotate based:

    * :class:`~randomgen.lxm.LXM`
    * :class:`~randomgen.xoroshiro128.Xoroshiro128`
    * :class:`~randomgen.xorshift1024.Xorshift1024`
    * :class:`~randomgen.xoshiro256.Xoshiro256`
    * :class:`~randomgen.xoshiro512.Xoshiro512`

.. container:: admonition danger

  .. raw:: html

      <p class="admonition-title"> Deprecated </p>

  ``Generator`` is **deprecated**. You should be using
  :class:`numpy.random.Generator`.

* randomgen's ``Generator`` continues to expose legacy
  methods ``random_sample``, ``randint``, ``random_integers``, ``rand``, ``randn``,
  and ``tomaxint``. **Note**: These should not be used, and their modern replacements are preferred:

  * ``random_sample``, ``rand` → ``random``
  * ``random_integers``, ``randint`` → ``integers``
  * ``randn`` → ``standard_normal``
  * ``tomaxint`` → ``integers`` with ``dtype`` set to ``int``

* randomgen's bit generators remain seedable and the convenience function
  ``seed` is exposed as part of``Generator``. Additionally, the convenience
  property ``state`` is available to get or set the state of the underlying bit generator.

* :func:`numpy.random.Generator.multivariate_hypergeometric` was added after
  ``Generator`` was merged into NumPy and will not be ported over.

* :func:`numpy.random.Generator.shuffle` and :func:`numpy.random.Generator.permutation`
  support ``axis`` keyword to operator along an axis other than 0.

* ``integers`` supports the keyword argument ``use_masked`` to switch between masked
  generation of bounded integers and Lemire's superior method.

Differences from NumPy before 1.17
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* The normal, exponential and gamma generators use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's default implementation in
  ``standard_normal``, ``standard_exponential`` or ``standard_gamma``.

* The Box-Muller used to produce NumPy's normals is no longer available.
* All bit generators functions to produce doubles, uint64s and
  uint32s via CTypes (:meth:`~randomgen.xoroshiro128.Xoroshiro128.ctypes`)
  and CFFI (:meth:`~randomgen.xoroshiro128.Xoroshiro128.cffi`).  This allows
  the bit generators to be used in numba or in other low-level applications
* The bit generators can be used in downstream projects via Cython.
* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double prevision uniform random variables for
  select core distributions

  * Uniforms (``random`` and ``rand``)
  * Normals (``standard_normal`` and ``randn``)
  * Standard Gammas (``standard_gamma``)
  * Standard Exponentials (``standard_exponential``)

* Optional ``out`` argument that allows existing arrays to be filled for
  select core distributions

  * Uniforms (``random``)
  * Normals (``standard_normal``)
  * Standard Gammas (``standard_gamma``)
  * Standard Exponentials (``standard_exponential``)

  This allows multithreading to fill large arrays in chunks using suitable
  PRNGs in parallel.


* ``integers`` supports broadcasting inputs.

* ``integers`` supports drawing from open (default, ``[low, high)``) or closed
  (``[low, high]``) intervals using the keyword argument
  ``endpoint``. Closed intervals are simpler to use when the
  distribution may include the maximum value of a given integer type.


* The closed interval is particularly helpful when using arrays since
  it avoids object-dtype arrays when sampling from the full range.


* Support for Lemire’s method of generating uniform integers on an
  arbitrary interval by setting ``use_masked=True`` in
  (``integers``).


* ``multinomial`` supports multidimensional values of ``n``


* ``choice``  is much faster when sampling small amounts from large arrays


* ``choice`` supports the ``axis`` keyword to work with multidimensional arrays.


* For changes since the previous release, see the :ref:`change-log`

.. _the demonstration notebook: custom-bit-generators.ipynb