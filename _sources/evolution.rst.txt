.. _evolution:

Evolution of randomgen
======================

Changes in 1.23
---------------
``Generator`` and ``RandomState`` were **removed** in 1.23.0.

Use :class:`numpy.random.Generator` if possible, or :class:`numpy.random.RandomState`
if you face legacy constraints.

Changes in 1.19
---------------

``Generator`` and ``RandomState`` have been
officially deprecated, and will warn with a ``FutureWarning`` about their removal. They will
also receive virtually no maintenance. It is now time to move to NumPy's :class:`numpy.random.Generator`
which has features not in ``Generator`` and is maintained more actively.

A few distributions that are not present in ``Generator`` have been moved
to :class:`~randomgen.generator.ExtendedGenerator`:

* :func:`~randomgen.generator.ExtendedGenerator.multivariate_normal`: which supports broadcasting
* :func:`~randomgen.generator.ExtendedGenerator.uintegers`: fast 32 and 64-bit uniform integers
* :func:`~randomgen.generator.ExtendedGenerator.complex_normal`: scalar complex normals

There are no plans to remove any of the bit generators, e.g., :class:`~randomgen.aes.AESCounter`,
:class:`~randomgen.threefry.ThreeFry`, or :class:`~randomgen.pcg64.PCG64`.

Changes between 1.16 and 1.18
-----------------------------

There are many changes between v1.16.x and v1.18.x. These reflect API
decision taken in conjunction with NumPy in preparation of the core
of ``randomgen`` being used as the preferred random number generator in
NumPy. These all issue ``DeprecationWarning`` except for ``BitGenerator.generator``
which raises ``NotImplementedError``. The C-API has also changed to reflect
the preferred naming the underlying Pseudo-RNGs, which are now known as
bit generators (or ``BitGenerator``).

The main changes are

* Rename ``RandomGenerator`` to ``Generator``.
* Rename ``randint`` to ``integers``.
* Rename ``random_integers`` to   ``integers``.
* Rename ``random_sample`` to ``random``.
* Change ``jump`` which operated in-place to ``jumped`` which returns a new ``BitGenerator``.
* Rename Basic RNG to bit generator, which impacts the API in multiple places where names
  like ``brng`` and ``basic_rng`` have been replaced by ``bitgen`` or ``bit_generator``.
* Support for ``randomgen.seed_sequence.SeedSequence`` (also support NumPy :class:`~numpy.random.SeedSequence` instances)
* Removed support for Python 2.7
