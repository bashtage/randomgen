:hero: Additional bit generators and distributions for NumPy's Generator

.. danger::

    ``Generator`` and ``RandomState`` has been removed from randomgen in 1.23. 
    randomgen has been substantially merged into NumPy as of 1.17. :ref:`evolution`
    details how randomgen has changed since it was incorporated into NumPy.

RandomGen
=========
This package contains additional bit generators for NumPy's
:class:`~numpy.random.Generator` and an :class:`~randomgen.generator.ExtendedGenerator`
exposing methods not in :class:`~numpy.random.Generator`.

Introduction
------------
randomgen supplies many bit generators that can be used with
:class:`numpy.random.Generator`.  It also supports additional variants
of the bit generators included in NumPy.

.. code-block:: python

  from numpy.random import Generator
  from randomgen import AESCounter
  rg = Generator(AESCounter())
  rg.random()

Seed information is directly passed to the bit generator.

.. code-block:: python

  rg = Generator(AESCounter(12345))
  rg.random()

History
-------

randomgen began as a project to modernize NumPy's :class:`~numpy.random.RandomState`.
It has succeed at this goal. Some of the components on randomgen were deemed too
exotic to include with NumPy and so have been retained in randomgen.  Most of these
are bit generators or extended features of bit generators included with NumPy. In addition
:class:`randomgen.generator.ExtendedGenerator` exposes some methods that are not included in
:class:`~numpy.random.Generator`.


What's New or Different
~~~~~~~~~~~~~~~~~~~~~~~

* An :class:`~randomgen.generator.ExtendedGenerator` containing methods not in :class:`numpy.random.Generator`
  that can be used alongside NumPy's :class:`~numpy.random.Generator`.
* Support for a wider variety of generators including high-quality
  cryptography-based generators (:class:`~randomgen.aes.AESCounter`,
  :class:`~randomgen.speck128.SPECK128`).
* Support for writing :class:`~randomgen.wrapper.UserBitGenerator` in Python (slow) or numba (fast)
  that can be used as an input for :class:`~numpy.random.Generator`.
* Extended configuration options in :class:`~randomgen.pcg64.PCG64` (see also :class:`~randomgen.pcg64.LCG128Mix`),
  :class:`~randomgen.sfc.SFC64`, and :class:`~randomgen.philox.Philox`.

See :ref:`new-or-different` for a complete list of improvements and
differences.

Parallel Generation
~~~~~~~~~~~~~~~~~~~

The included generators can be used in parallel, distributed applications in
one of five ways:

* :ref:`using-seed-sequence`
* :ref:`distinct-cryptographic-keys`
* :ref:`advancing`
* :ref:`jumping`
* :ref:`weyl-sequences`

Supported Generators
--------------------
The main innovation is the inclusion of a number of alternative pseudo-random number
generators, 'in addition' to the standard PRNG in NumPy.  The included PRNGs are:

* :class:`~randomgen.mt19937.MT19937` - The standard NumPy generator.  Produces identical results to NumPy
  using the same seed/state. Adds a jump function that advances the generator
  as-if 2**128 draws have been made (:meth:`~randomgen.mt19937.MT19937.jumped`).
  See `NumPy's documentation`_.
* :class:`~randomgen.dsfmt.DSFMT` and :class:`~randomgen.sfmt.SFMT` - SSE2 enabled versions of the MT19937 generator.  Theoretically
  the same, but with a different state and so it is not possible to produce a
  sequence identical to MT19937. :class:`~randomgen.dsfmt.DSFMT` supports ``jump`` and so can
  be used in parallel applications. See the `dSFMT authors' page`_.
* Xorshiro256** and Xorshiro512** - The most recently introduced XOR,
  shift, and rotate generator. Supports ``jump`` and so can be used in
  parallel applications. See the documentation for
  :meth:`~randomgen.xoshiro256.Xoshiro256.jumped` for details. More
  information about these PRNGs is available at the
  `xorshift, xoroshiro and xoshiro authors' page`_.
* :class:`~randomgen.pcg64.PCG64` - Fast generator that support many parallel streams and
  can be advanced by an arbitrary amount. See the documentation for
  :meth:`~randomgen.pcg64.PCG64.advance`. PCG-64 has a period of
  :math:`2^{128}`. See the `PCG author's page`_ for more details about
  this class of PRNG. :class:`~randomgen.pcg64.LCG128Mix` extends the
  basic PCG-64 generator to allow user-defined multipliers and output functions.
* ThreeFry and Philox - counter-based generators capable of being advanced an
  arbitrary number of steps or generating independent streams. See the
  `Random123`_ page for more details about this class of PRNG.
* Other cryptographic-based generators: :class:`~randomgen.aes.AESCounter`,
  :class:`~randomgen.speck128.SPECK128`, :class:`~randomgen.chacha.ChaCha`, and
  :class:`~randomgen.hc128.HC128`.
* XoroShiro128+/++ - Improved version of XorShift128+ with better performance
  and statistical quality. Like the XorShift generators, it can be jumped
  to produce multiple streams in parallel applications. See
  :meth:`~randomgen.xoroshiro128.Xoroshiro128.jumped` for details.
  More information about this PRNG is available at the
  `xorshift, xoroshiro and xoshiro authors' page`_.
* XorShift1024*φ - Fast generator based on the XSadd
  generator. Supports ``jump`` and so can be used in
  parallel applications. See the documentation for
  :meth:`~randomgen.xorshift1024.Xorshift1024.jumped` for details. More information
  about these PRNGs is available at the
  `xorshift, xoroshiro and xoshiro authors' page`_.

.. _`NumPy's documentation`: https://docs.scipy.org/doc/numpy/reference/routines.random.html
.. _`dSFMT authors' page`: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/
.. _`xorshift, xoroshiro and xoshiro authors' page`:  https://prng.di.unimi.it/
.. _`PCG author's page`: https://www.pcg-random.org/
.. _`Random123`: https://www.deshawresearch.com/resources_random123.html

Random Generator
----------------
.. toctree::
   :maxdepth: 1

   extended-generator
   new-or-different
   future

Bit Generators
--------------

.. toctree::
   :maxdepth: 3

   Bit Generators <bit_generators/index>
   Seed Sequences <seed_sequence>

New Features
------------
.. toctree::
   :maxdepth: 2

   Parallel Applications <parallel>
   Multithreaded Generation <multithreading>
   Quality Assurance <testing>
   Comparing Performance <performance>
   extending
   custom-bit-generators.ipynb
   Reading System Entropy <entropy>
   references

Removed Features
----------------
.. toctree::
   :maxdepth: 2

   Random Generation <generator>
   legacy


Changes
~~~~~~~
.. toctree::
   :maxdepth: 2

   evolution
   Change Log <change-log>

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
