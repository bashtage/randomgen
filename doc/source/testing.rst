=================
Quality Assurance
=================

A values below are the maximum output size where a bit generator or sequence of bit generators
have been tested using PractRand_. A -- indicates that configuration is not relevant. Failures are marked
with FAIL. Most bit generators were only tested in their default configuration.
Non-default configurations are indicated by listing the keyword arguments to the bit generator.

All bit generators have been tested using the same :class:`~numpy.random.SeedSequence`
initialized with 256-bits of entropy taken from random.org.

.. note::

   Testing is a WIP and all configurations of the bit generators will be tested to
   at least 4TB.

+------------------------------+-----------+-----------------------+-----------------------+
| Method                       |           | Seed Sequence         | Jumped                |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| Streams                      |           | 4         | 8196      | 4         | 8196      |
+==============================+===========+===========+===========+===========+===========+
| AESCounter                   |       4TB |       4TB |       4TB |       4TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| ChaCha(rounds=20)            |       4TB |       4TB |       4TB |       4TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| ChaCha(rounds=8)             |       4TB |       4TB |       4TB |       4TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| DSFMT⁴                       |       4TB |     FAIL¹ |       4TB |     FAIL¹ |     FAIL¹ |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| EFIIX64                      |       1TB |       1TB |       1TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| HC128                        |       4TB |       4TB |       4TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| JSF                          |       4TB |       4TB |       4TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| JSF(seed_size=3)             |       4TB |       4TB |       4TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| LCG128Mix(output=upper)      |       4TB |       4TB |       4TB |       4TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| LXM                          |       4TB |       4TB |       4TB |       4TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| MT19937⁴,⁵                   |       4TB |     FAIL¹ |       4TB |     FAIL¹ |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| PCG64DXSM²                   |       1TB |       1TB |       4TB |       1TB |       1TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| PCG64(variant=dxsm-128)      |       1TB |       1TB |       1TB |       4TB |       1TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| PCG64⁵                       |       4TB |       4TB |       1TB |       4TB |       1TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| Philox⁵                      |       4TB |       4TB |       4TB |       4TB |       1TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| SFC64⁵                       |       1TB |       1TB |       4TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| SFC64(k=3394385948627484371) |       1TB |       1TB |       1TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| SFC64(k=Weyl)³               |       1TB |       1TB |       1TB |        -- |        -- |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| SFMT⁴                        |       1TB |     FAIL¹ |       1TB |     FAIL¹ |       1TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| SPECK128                     |       4TB |       1TB |       1TB |       1TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| ThreeFry                     |       4TB |       4TB |       1TB |       4TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| Xoshiro256                   |       1TB |       4TB |       4TB |       4TB |       1TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+
| Xoshiro512                   |       4TB |       1TB |       1TB |       1TB |       4TB |
+------------------------------+-----------+-----------+-----------+-----------+-----------+

Notes
-----
¹ Failed at 512GB.

² PCG64DXSM and PCG64(variant=dxsm) are identical and so the latter not separately reported.

³ SFC64(k=weyl) uses distinct Weyl increments that have 50% or fewer non-zero bits.

⁴ The Mersenne Twisters begin to fail at 256GB.  This is a known limitation of MT-family
generators. These should not be used in large studies except when backward compatibility
is required.

⁵ Identical output to the version included in NumPy 1.19.

.. _PractRand: http://pracrand.sourceforge.net/
