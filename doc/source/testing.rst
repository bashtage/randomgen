=================
Quality Assurance
=================

A values below are the maximum output size where a bit generator or sequence of bit generators
has passed PractRand_.ꭞ A -- indicates that configuration is not relevant. Failures are marked
with FAIL. Most bit generators were only tested in their default configuration.
Non-default configurations are indicated by listing the keyword arguments to the bit generator.

All bit generators have been tested using the same :class:`~numpy.random.SeedSequence`
initialized with the same 256-bits of entropy taken from random.org.

.. include:: test-results.txt

Notes
-----
ꭞ Testing is an on-going process. All generators will be tested to at east 4TB.

¹ Failed at 512GB.

² PCG64DXSM and PCG64(variant=dxsm) are identical and so the latter not separately reported.

³ SFC64(k=weyl) uses distinct Weyl increments that have 50% or fewer non-zero bits.

⁴ The Mersenne Twisters begin to fail at 256GB.  This is a known limitation of MT-family
generators. These should not be used in large studies except when backward compatibility
is required.

⁵ Identical output to the version included in NumPy 1.19.

.. _PractRand: http://pracrand.sourceforge.net/
