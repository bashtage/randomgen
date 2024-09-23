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

* For changes since the previous release, see the :ref:`change-log`

.. _the demonstration notebook: custom-bit-generators.ipynb