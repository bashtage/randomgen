Bit Generators
--------------

The random values produced by :class:`numpy.random.Generator`
(and also  :class:`~randomgen.generator.Generator`)
are produced by a bit generator.  These bit generators do not directly provide
random numbers and only contains methods used for seeding, getting or
setting the state, jumping or advancing the state, and for accessing
low-level wrappers for consumption by code that can efficiently
access the functions provided, e.g., `numba <https://numba.pydata.org>`_.

Stable RNGs
===========
These RNGs will be included in future releases.


.. toctree::
   :maxdepth: 1

   AES Counter <aesctr>
   ChaCha <chacha>
   DSFMT <dsfmt>
   HC128 <hc128>
   JSF <jsf>
   LXM <lxm>
   MT19937 <mt19937>
   MT64 <mt64>
   PCG32 <pcg32>
   PCG64 <pcg64>
   Philox <philox>
   RDRAND <rdrand>
   SFC64 <sfc>
   SFMT <sfmt>
   SPECK128 <speck128>
   ThreeFry <threefry>
   XoroShiro128+/++ <xoroshiro128>
   Xoshiro256** <xoshiro256>
   Xoshiro512** <xoshiro512>

Experimental RNGs
=================

These RNGs are currently included for testing but are may not be
permanent.

.. toctree::
   :maxdepth: 1

   Xorshift1024*φ <xorshift1024>

User-defined Bit Generators
===========================

.. toctree::
   :maxdepth: 1

   userbitgenerator
