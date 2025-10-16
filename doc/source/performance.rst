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

    "Romu(variant=""trio"")",1.9,2.6,2.4,3.9,7.3,18.4
    Romu,2.1,2.6,2.4,4.2,7.4,19.4
    SFC64,2.1,2.6,2.5,4.2,7.6,20.6
    Xoshiro256,2.3,2.8,2.7,4.4,8.0,22.1
    JSF,2.2,2.9,2.7,4.6,8.1,21.5
    SFMT,2.2,2.8,2.5,4.7,8.7,21.5
    PCG64DXSM,2.3,3.0,2.8,4.7,9.2,23.8
    PCG64,2.3,3.2,2.9,4.6,9.0,23.2
    Xoshiro512,2.6,3.0,2.9,5.3,8.9,21.9
    Squares,2.4,2.9,2.8,5.2,11.3,27.1
    AESCounter,2.8,2.9,3.0,5.3,10.4,24.4
    "PCG64(variant=""dxsm-128"")",2.5,3.4,3.2,4.8,10.5,25.5
    LXM,2.6,3.4,3.3,5.8,9.5,24.6
    MT64,2.6,3.3,3.2,5.5,10.4,26.5
    DSFMT,2.8,4.2,2.5,6.3,10.1,25.1
    JSF32,2.8,3.6,3.8,5.8,9.4,23.7
    MT19937,2.5,4.2,4.0,6.1,10.5,20.8
    EFIIX64,2.4,3.5,3.4,9.8,12.3,24.7
    Philox,3.1,4.0,4.5,6.1,10.6,28.1
    "Philox(n=2, w=64)",3.0,4.0,4.1,6.6,12.3,30.7
    "ThreeFry(n=2, w=64)",3.5,5.4,5.2,7.7,13.0,31.0
    TycheOpenRand,4.3,5.5,6.2,8.3,12.6,27.5
    ThreeFry,4.7,6.5,6.4,8.4,12.2,23.7
    HC128,3.7,5.9,5.9,8.2,13.2,34.8
    SPECK128,4.4,6.4,6.8,8.5,12.7,24.7
    "Philox(n=4, w=32)",3.8,6.5,6.5,9.2,13.5,32.8
    ChaCha(rounds=8),4.2,7.3,7.2,9.3,13.9,26.4
    Tyche,5.4,8.2,7.8,9.8,16.3,30.7
    NumPy,5.1,5.2,5.2,18.9,28.3,37.9
    BlaBla,7.6,12.4,12.7,14.8,17.9,36.6
    ChaCha,6.9,12.9,12.7,15.1,19.4,37.6
    "ThreeFry(n=4, w=32)",6.2,11.4,12.8,18.8,22.4,59.8
    RDRAND,882.5,886.7,883.7,912.2,903.3,1929.9

The next table presents the performance relative to NumPy's ``RandomState`` in
percentage. The overall performance is computed using a geometric mean.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
   :widths: 30,10,10,10,10,10,10,10

    Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
    "Romu(variant=""trio"")",267,198,217,481,389,206,276
    Romu,248,203,217,455,380,196,268
    SFC64,242,197,206,455,370,184,259
    Xoshiro256,219,185,195,432,353,171,243
    JSF,232,183,192,407,348,177,243
    SFMT,228,187,207,400,323,177,242
    PCG64DXSM,219,172,186,406,309,160,228
    PCG64,224,162,177,414,313,163,227
    Xoshiro512,195,173,177,358,319,173,221
    Squares,210,179,185,364,250,140,211
    AESCounter,181,179,173,354,273,155,209
    "PCG64(variant=""dxsm-128"")",201,154,165,391,270,149,208
    LXM,193,155,159,329,298,154,204
    MT64,200,157,162,343,272,143,202
    DSFMT,183,125,209,298,279,151,198
    JSF32,182,144,138,324,301,160,196
    MT19937,201,123,128,309,269,182,191
    EFIIX64,212,148,155,192,229,154,179
    Philox,163,131,116,311,266,135,174
    "Philox(n=2, w=64)",170,131,126,285,231,123,168
    "ThreeFry(n=2, w=64)",145,96,101,246,217,122,145
    TycheOpenRand,118,94,84,227,224,138,137
    ThreeFry,109,80,81,225,232,160,135
    HC128,137,88,89,229,214,109,134
    SPECK128,115,82,76,222,222,154,132
    "Philox(n=4, w=32)",134,80,80,206,209,116,127
    ChaCha(rounds=8),122,72,73,202,203,144,125
    Tyche,95,63,66,193,173,124,109
    BlaBla,67,42,41,128,158,104,79
    ChaCha,74,40,41,125,146,101,78
    "ThreeFry(n=4, w=32)",83,46,41,100,126,63,71
    RDRAND,1,1,1,2,3,2,1


.. note::

   All timings were taken using Linux on an Intel Cascade Lake (Family 6,
   Model 85, Stepping 7) running at 3.1GHz.
