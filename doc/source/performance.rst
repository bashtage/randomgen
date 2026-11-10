Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is :class:`~randomgen.pcg64.PCG64DXSM`
although :class:`~randomgen.mwcx.MWC192X`, :class:`~randomgen.sfc.SFC64` and :class:`~randomgen.xoshiro256.Xoshiro256`
are all excellent alternatives. :class:`~randomgen.romu.Romu` is a newer generator that
is also very fast.

For very large scale
applications -- requiring 1,000+ streams --
:class:`~randomgen.mwcx.MWC192X` using ``advance``, 
:class:`~randomgen.pcg64.PCG64DXSM` combined with a :class:`~numpy.random.SeedSequence` and ``spawn``,
:class:`~randomgen.sfc.SFC64` initialized using distinct Weyl increments (``k``), or one of
the cryptography-based generators :class:`~randomgen.aes.AESCounter` (if you have hardware acceleration),
:class:`~randomgen.efiix64.EFIIX64`, :class:`~randomgen.speck128.SPECK128`,
:class:`~randomgen.philox.Philox`, or :class:`~randomgen.hc128.HC128` if you do not)
initialized with distinct keys are all excellent choices.

Unless you need backward compatibility, there are no good reasons to use any
of the Mersenne Twister PRNGS: :class:`~randomgen.mt19937.MT19937`, :class:`~randomgen.mt64.MT64`,
:class:`~randomgen.sfmt.SFMT`, and :class:`~randomgen.dsfmt.DSFMT`.

Timings
*******

The timings below are the time in ns to produce 1 random value from a
specific distribution. :class:`~randomgen.romu.Romu` and :class:`~randomgen.sfc.SFC64` are the fastest,
followed closely by  followed closely by  :class:`~randomgen.mwcx.MWC128X` and :class:`~randomgen.mwcx.MWC192X` (and related generators), :class:`~randomgen.xoshiro256.Xoshiro256`,
:class:`~randomgen.pcg64.PCG64DXSM`, :class:`~randomgen.jsf.JSF`,
and :class:`~randomgen.efiix64.EFIIX64`. The original
NumPy :class:`~randomgen.mt19937.MT19937` generator is slower since
it requires 2 32-bit values to equal the output of the faster generators.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma
   :widths: 30,10,10,10,10,10,10

   "Romu(variant=""trio"")",2.0,2.2,2.3,3.8,4.1,17.6
   SFC64,2.1,2.5,2.7,3.6,4.0,16.8
   MWC128X,2.5,2.5,2.2,3.9,4.3,17.5
   MWC192X,2.8,2.6,2.6,3.9,4.6,16.3
   Xoshiro256,2.3,2.8,3.1,3.9,4.1,17.2
   Romu,2.4,2.5,3.8,4.4,4.3,17.7
   MWC64X32,2.2,3.3,3.2,4.5,4.3,18.8
   PCG64DXSM,2.5,3.3,3.4,4.2,4.3,18.7
   MWC128X32,2.5,3.4,3.3,4.1,4.4,21.4
   JSF,2.9,3.3,3.4,4.8,5.1,19.6
   EFIIX64,2.7,4.1,3.5,4.4,4.3,22.7
   SFMT,3.0,3.2,3.2,5.3,5.4,19.8
   Squares,2.8,3.6,3.8,5.1,5.0,19.5
   PCG64,3.1,3.9,3.5,5.9,6.1,23.2
   LXM,3.3,4.0,4.0,5.6,5.7,22.1
   AESCounter,3.5,3.9,3.8,5.8,6.2,21.0
   DSFMT,3.1,4.8,2.9,6.7,6.6,19.5
   TycheOpenRand,3.0,5.1,4.7,5.7,5.4,20.6
   "PCG64(variant=""dxsm-128"")",3.1,5.4,3.9,6.2,6.2,19.5
   JSF32,3.0,6.1,5.4,6.6,6.5,24.8
   Xoshiro512,4.4,6.6,6.6,7.2,7.5,25.4
   MT19937,3.7,6.3,7.0,8.6,8.6,26.7
   Philox,5.0,5.9,6.9,8.7,7.9,24.0
   MT64,3.6,7.3,5.0,10.4,11.4,24.9
   Tyche,4.3,7.8,7.7,8.7,9.1,25.3
   "Philox(n=2, w=64)",5.2,7.2,8.2,9.2,8.8,28.6
   HC128,4.2,7.9,7.8,9.6,9.7,31.1
   SPECK128,6.1,10.3,10.7,12.4,13.3,32.9
   NumPy,3.8,7.2,7.6,17.7,22.6,44.5
   BlaBla,6.7,10.7,10.8,14.4,14.4,34.0
   ChaCha(rounds=8),7.1,14.0,14.4,14.5,14.6,42.6
   ThreeFry,10.2,14.9,15.2,16.6,16.4,45.7
   "Philox(n=4, w=32)",7.6,17.4,21.3,24.3,25.0,50.6
   ChaCha,11.1,19.6,21.6,20.9,21.3,55.7
   "ThreeFry(n=2, w=64)",14.2,24.0,23.9,26.3,26.2,63.7
   "ThreeFry(n=4, w=32)",15.3,25.4,26.2,27.7,27.8,68.5
   RDRAND,1618.4,1581.0,1595.3,1855.4,1966.9,3409.9

The next table presents the performance relative to NumPy's ``RandomState`` in
percentage. The overall performance is computed using a geometric mean.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
   :widths: 30,10,10,10,10,10,10,10

   "Romu(variant=""trio"")",192,326,337,469,555,253,334
   SFC64,180,286,278,493,570,265,320
   MWC128X,154,289,344,453,527,254,312
   MWC192X,137,276,292,455,496,273,297
   Xoshiro256,163,259,249,448,551,259,296
   Romu,158,288,203,404,529,251,281
   MWC64X32,172,218,235,394,520,237,274
   PCG64DXSM,152,222,225,423,529,238,272
   MWC128X32,150,214,229,428,511,208,263
   JSF,132,220,223,366,441,228,249
   EFIIX64,141,176,216,400,527,196,246
   SFMT,124,227,235,334,416,225,243
   Squares,136,202,200,347,447,228,241
   PCG64,122,184,220,302,370,192,217
   LXM,113,180,191,316,394,201,215
   AESCounter,109,188,202,303,362,213,214
   DSFMT,121,151,259,263,340,228,214
   TycheOpenRand,125,142,162,310,421,216,208
   "PCG64(variant=""dxsm-128"")",122,134,194,284,363,229,205
   JSF32,125,118,141,267,349,179,181
   Xoshiro512,86,110,115,244,300,175,155
   MT19937,102,115,109,207,261,167,150
   Philox,76,122,111,203,286,186,149
   MT64,105,100,151,170,197,179,146
   Tyche,87,93,99,203,249,176,139
   "Philox(n=2, w=64)",73,101,92,191,257,156,132
   HC128,89,92,98,185,232,143,130
   SPECK128,62,71,71,143,170,135,100
   BlaBla,57,68,71,123,157,131,94
   ChaCha(rounds=8),54,52,53,122,154,105,81
   ThreeFry,37,49,50,106,138,97,71
   "Philox(n=4, w=32)",50,42,36,73,90,88,59
   ChaCha,34,37,35,85,106,80,56
   "ThreeFry(n=2, w=64)",27,30,32,67,86,70,47
   "ThreeFry(n=4, w=32)",25,29,29,64,81,65,44
   RDRAND,0,0,0,1,1,1,1


.. note::

   All timings were taken using Linux on an Intel Cascade Lake (Family 6,
   Model 85, Stepping 7) running at 3.1GHz.
