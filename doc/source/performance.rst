Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is :class:`~randomgen.pcg64.PCG64DXSM`
although :class:`~randomgen.sfc.SFC64` and :class:`~randomgen.xoshiro256.Xoshiro256`
are both excellent alternatives. :class:`~randomgen.romu.Romu` is a newer generator that
is also very fast.

For very large scale
applications -- requiring 1,000+ streams --
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
specific distribution. :class:`~randomgen.sfc.SFC64` is the fastest,
followed closely by  :class:`~randomgen.xoshiro256.Xoshiro256`,
:class:`~randomgen.pcg64.PCG64DXSM`, :class:`~randomgen.jsf.JSF`,
and :class:`~randomgen.efiix64.EFIIX64`. The original
NumPy :class:`~randomgen.mt19937.MT19937` generator is slower since
it requires 2 32-bit values to equal the output of the faster generators.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma
   :widths: 30,10,10,10,10,10,10

   "Romu(variant=""trio"")",2.3,2.1,2.0,5.2,9.3,18.2
   SFC64,2.3,2.5,2.4,5.1,9.6,18.7
   Romu,2.5,2.5,2.4,5.0,9.5,18.8
   Xoshiro256,2.5,2.5,2.5,5.2,9.8,19.2
   PCG64DXSM,2.3,2.9,3.0,5.3,11.0,20.7
   JSF,2.4,3.0,3.0,5.8,10.2,20.0
   EFIIX64,2.5,3.0,3.0,5.4,10.5,20.4
   PCG64,2.3,3.1,3.1,5.9,11.3,21.4
   Xoshiro512,2.7,3.5,3.3,5.8,10.3,20.3
   SFMT,2.9,3.3,3.1,6.3,11.0,20.9
   LXM,2.6,3.5,3.5,6.3,11.3,21.9
   "PCG64(variant=""dxsm-128"")",2.8,3.4,3.5,6.1,12.5,23.1
   DSFMT,3.0,4.2,2.7,7.0,12.2,21.7
   MT64,2.8,4.0,4.2,6.9,12.8,23.7
   JSF32,3.0,4.3,4.3,6.9,11.2,22.9
   "Philox(n=2, w=64)",3.2,4.7,5.1,8.1,14.7,27.4
   Philox,4.0,5.9,6.1,8.8,13.6,27.0
   AESCounter,4.4,6.0,5.7,9.1,14.4,27.4
   MT19937,3.8,6.3,7.2,9.2,14.8,28.8
   "ThreeFry(n=2, w=64)",4.1,6.5,6.9,9.5,15.7,30.6
   NumPy,3.0,4.6,5.8,14.4,20.1,39.8
   HC128,4.1,7.2,7.2,10.5,16.6,31.5
   "Philox(n=4, w=32)",4.2,7.6,8.7,10.8,16.5,32.6
   SPECK128,5.4,8.1,9.7,11.4,17.0,33.4
   ThreeFry,5.9,9.1,9.3,12.0,16.7,34.9
   ChaCha(rounds=8),6.7,10.4,10.3,13.2,18.1,36.4
   ChaCha,9.7,16.6,16.4,19.6,24.4,49.0
   "ThreeFry(n=4, w=32)",9.0,16.8,17.5,20.7,24.3,54.1
   RDRAND,129.5,129.9,129.6,136.9,139.6,287.4

The next table presents the performance relative to NumPy's ``RandomState`` in
percentage. The overall performance is computed using a geometric mean.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
   :widths: 30,10,10,10,10,10,10,10

   "Romu(variant=""trio"")",128,219,282,278,215,219,217
   SFC64,129,189,244,283,210,213,205
   Romu,118,185,242,285,212,212,202
   Xoshiro256,119,185,229,279,205,208,198
   PCG64DXSM,128,163,190,271,182,192,183
   JSF,126,155,193,250,197,199,182
   EFIIX64,120,154,193,264,190,195,181
   PCG64,128,149,184,246,178,186,175
   Xoshiro512,111,131,176,249,195,196,170
   SFMT,104,143,185,228,183,190,167
   LXM,114,135,164,227,177,182,162
   "PCG64(variant=""dxsm-128"")",105,138,165,234,160,172,158
   DSFMT,100,110,213,206,165,184,156
   MT64,104,117,136,210,156,168,145
   JSF32,99,108,134,208,178,174,145
   "Philox(n=2, w=64)",91,98,114,177,136,145,124
   Philox,75,79,95,163,147,147,112
   AESCounter,67,77,102,158,139,145,109
   MT19937,78,74,80,156,135,138,105
   "ThreeFry(n=2, w=64)",72,71,84,152,128,130,101
   HC128,72,64,80,136,121,126,96
   "Philox(n=4, w=32)",71,61,66,134,122,122,91
   SPECK128,55,57,59,127,118,119,83
   ThreeFry,51,51,62,120,120,114,80
   ChaCha(rounds=8),45,45,56,109,111,109,73
   ChaCha,30,28,35,73,82,81,49
   "ThreeFry(n=4, w=32)",33,28,33,69,82,73,48
   RDRAND,2,4,4,10,14,14,7

.. note::

   All timings were taken using Linux on an Intel Cascade Lake (Family 6,
   Model 85, Stepping 7) running at 3.1GHz.
