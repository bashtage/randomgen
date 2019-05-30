Bit Generators
--------------

.. warning::

   These for formerly called Basic Random Number Generators. They have been
   renamed Bit Generators for compatibility with the version that will ship
   with NumPy.

The random values produced by :class:`~randomgen.generator.Generator`
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

   DSFMT <dsfmt>
   MT19937 <mt19937>
   MT64 <mt64>
   PCG64 <pcg64>
   Philox <philox>
   SFMT <sfmt>
   ThreeFry <threefry>
   Xoshiro256** <xoshiro256>
   Xoshiro512** <xoshiro512>


Experimental RNGs
=================

These RNGs are currently included for testing but are may not be
permanent.

.. toctree::
   :maxdepth: 1

   XoroShiro128+ <xoroshiro128>
   Xorshift1024*φ <xorshift1024>
   PCG32 <pcg32>
   ThreeFry32 <threefry32>
