RandomGen
=========
This package contains replacements for the NumPy
:class:`~numpy.random.RandomState` object that allows the core random number
generator be be changed.

Introduction
------------
RandomGen takes a different approach to producing random numbers from the
:class:`numpy.random.RandomState` object used in NumPy.  Random number
generation is separated into two components, a basic RNG and a random
generator.

The basic RNG has a limited set of responsibilities -- it manages the
underlying RNG state and provides functions to produce random doubles and
random unsigned 32- and 64-bit values. The basic random generator also handles
all seeding since this varies when using alternative basic RNGs.

The random generator (:class:`~randomgen.generator.RandomGenerator`) takes the
basic RNG-provided functions and transforms them into more useful
distributions, e.g., simulated normal random values. This structure allows
alternative basic RNGs to be used without code duplication.

The :class:`~randomgen.generator.RandomGenerator` is the user-facing object
that is nearly identical to :class:`~numpy.random.RandomState`. The canonical
method to initialize a generator passes a basic RNG --
:class:`~randomgen.mt19937.MT19937`, the underlying RNG in NumPy  -- as the
sole argument. Note that the basic RNG must be instantized.

.. ipython:: python

  from randomgen import RandomGenerator, MT19937
  rg = RandomGenerator(MT19937())
  rg.random_sample()

Seed information is directly passed to the basic RNG.

.. ipython:: python

  rg = RandomGenerator(MT19937(12345))
  rg.random_sample()

A shorthand method is also available which uses the
:meth:`~randomgen.mt19937.MT19937.generator` property from a basic RNG to
access an embedded random generator.

.. ipython:: python

  rg = MT19937(12345).generator
  rg.random_sample()

What's New or Different
~~~~~~~~~~~~~~~~~~~~~~~
.. warning::

  The Box-Muller method used to produce NumPy's normals is no longer available.
  It is not possible to exactly reproduce the random values produced from NumPy
  for the normal distribution or any other distribution that relies on the
  normal such as the gamma or student's t.

* The normal, exponential and gamma generators use 256-step Ziggurat
  methods which are 2-10 times faster than NumPy's Box-Muller or inverse CDF
  implementations.
* Optional ``dtype`` argument that accepts ``np.float32`` or ``np.float64``
  to produce either single or double prevision uniform random variables for
  select distributions
* Optional ``out`` argument that allows existing arrays to be filled for
  select distributions
* Simulate from the complex normal distribution
  (:meth:`~randomgen.generator.RandomGenerator.complex_normal`)
* :func:`~randomgen.entropy.random_entropy` provides access to the system
  source of randomness that is used in cryptographic applications (e.g.,
  ``/dev/urandom`` on Unix).
* All basic random generators functions to produce doubles, uint64s and
  uint32s via CTypes (:meth:`~randomgen.xoroshiro128.Xoroshiro128.ctypes`)
  and CFFI (:meth:`~randomgen.xoroshiro128.Xoroshiro128.cffi`).  This allows
  these basic RNGs to be used in numba.
* The basic random number generators can be used in downstream projects via
  Cython.

See :ref:`new-or-different` for a complete list of improvements and
differences.

Parallel Generation
~~~~~~~~~~~~~~~~~~~

The included generators can be used in parallel, distributed applications in
one of two ways:

* :ref:`independent-streams`
* :ref:`jump-and-advance`

Supported Generators
--------------------
The main innovation is the inclusion of a number of alternative pseudo-random number
generators, 'in addition' to the standard PRNG in NumPy.  The included PRNGs are:

* MT19937 - The standard NumPy generator.  Produces identical results to NumPy
  using the same seed/state. Adds a jump function that advances the generator
  as-if 2**128 draws have been made (:meth:`~randomgen.mt19937.MT19937.jump`).
  See `NumPy's documentation`_.
* dSFMT - SSE2 enabled versions of the MT19937 generator.  Theoretically
  the same, but with a different state and so it is not possible to produce a
  sequence identical to MT19937. Supports ``jump`` and so can
  be used in parallel applications. See the `dSFMT authors' page`_.
* XoroShiro128+ - Improved version of XorShift128+ with better performance
  and statistical quality. Like the XorShift generators, it can be jumped
  to produce multiple streams in parallel applications. See
  :meth:`~randomgen.xoroshiro128.Xoroshiro128.jump` for details.
  More information about this PRNG is available at the
  `xorshift and xoroshiro authors' page`_.
* XorShift1024*φ - Vast fast generator based on the XSadd
  generator. Supports ``jump`` and so can be used in
  parallel applications. See the documentation for
  :meth:`~randomgen.xorshift1024.Xorshift1024.jump` for details. More information
  about these PRNGs is available at the
  `xorshift and xoroshiro authors' page`_.
* PCG-64 - Fast generator that support many parallel streams and
  can be advanced by an arbitrary amount. See the documentation for
  :meth:`~randomgen.pcg64.PCG64.advance`. PCG-64 has a period of
  :math:`2^{128}`. See the `PCG author's page`_ for more details about
  this class of PRNG.
* ThreeFry and Philox - counter-based generators capable of being advanced an
  arbitrary number of steps or generating independent streams. See the
  `Random123`_ page for more details about this class of PRNG.

.. _`NumPy's documentation`: https://docs.scipy.org/doc/numpy/reference/routines.random.html
.. _`dSFMT authors' page`: http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/
.. _`xorshift and xoroshiro authors' page`:  http://xoroshiro.di.unimi.it/
.. _`PCG author's page`: http://www.pcg-random.org/
.. _`Random123`: https://www.deshawresearch.com/resources_random123.html

New Features
------------
.. toctree::
   :maxdepth: 2

   Parallel Applications <parallel>
   Multithreaded Generation <multithreading>
   new-or-different
   Reading System Entropy <entropy>
   Comparing Performance <performance>
   extending

Random Generator
----------------
.. toctree::
   :maxdepth: 1

   Random Generator <generator>

Basic Random Number Generators
------------------------------

.. toctree::
   :maxdepth: 3

   Basic Random Number Generators <brng/index>

Changes
~~~~~~~
.. toctree::
   :maxdepth: 2

   Change Log <change-log>

Indices and tables
~~~~~~~~~~~~~~~~~~

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`