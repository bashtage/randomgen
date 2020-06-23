Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is :class:`~randomgen.pcg64.PCG64DXSM`
although :class:`~randomgen.sfc64.SFC64` and :class:`~randomgen.xoshiro256.Xoshiro256`
are both excellent alternatives.

For very large scale
applications -- requiring 1,000+ streams,
:class:`~randomgen.pcg64.PCG64DXSM`, :class:`~randomgen.sfc64.SFC64`
using distinct Weyl increments (``k``), or one of the cryptography-based generators
:class:`~randomgen.aes.AESCounter` (if you have hardware acceleration),
:class:`~randomgen.effix64.EFFIC64`, :class:`~randomgen.speck128.SPECK128`,
:class:`~randomgen.philox.Philox`, or :class:`~randomgen.hc128.HC128` if you do not)
are all excellent choices.

Unless you need backward compatibilyt, there are no longer good reasons to any
of the Mersenne Twister PRNGS: :class:`~randomgen.mt19937.MT19937`, :class:`~randomgen.mt64.MT64`,
:class:`~randomgen.sfmt.SFMT`, and :class:`~randomgen.dsfmt.DSFMT`.

Timings
*******

The timings below are the time in ns to produce 1 random value from a
specific distribution. :class:`~randomgen.sfc64.SFC64` is the fastest,
followed closely by  :class:`~randomgen.xoshiro256.Xoshiro256`,
:class:`~randomgen.pcg64.PCG64DXSM`, :class:`~randomgen.jsf.JSF`,
and :class:`~randomgen.effix64.EFFIX64`. The original
NumPy :class:`~randomgen.mt19937.MT19937` generator is slower since
it requires 2 32-bit values to equal the output of the faster generators.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma
   :widths: 30,10,10,10,10,10,10

    SFC64,2.3,2.5,2.4,5.1,10.6,18.7
    Xoshiro256,2.5,2.5,2.5,5.2,9.7,19.1
    PCG64DXSM,2.3,2.8,3.0,5.3,11.0,20.7
    JSF,2.3,3.0,3.0,5.7,10.2,20.0
    EFIIX64,2.5,3.0,3.0,5.5,10.7,20.8
    PCG64,2.3,3.1,3.1,5.8,11.2,21.4
    Xoshiro512,2.6,3.5,3.3,5.8,10.3,20.3
    SFMT,2.9,3.3,3.1,6.3,10.9,20.8
    LXM,2.6,3.5,3.5,6.3,11.3,21.9
    PCG64(variant="dxsm-128"),2.8,3.3,3.5,6.1,12.5,24.3
    DSFMT,3.0,4.2,2.7,7.0,12.1,21.6
    MT64,2.8,4.0,4.2,6.8,12.8,23.7
    JSF32,3.0,4.3,4.3,6.9,11.2,22.9
    "Philox(n=2, w=64)",3.3,4.7,5.1,8.1,14.7,27.4
    Philox,3.9,5.9,6.0,8.8,13.6,26.9
    AESCounter,4.4,6.0,5.9,9.1,14.5,27.4
    MT19937,3.8,6.2,7.2,9.2,14.8,28.7
    "ThreeFry(n=2, w=64)",4.1,6.5,6.9,9.4,15.7,29.8
    NumPy,3.0,4.7,5.8,16.7,20.8,40.2
    HC128,4.1,7.2,7.2,10.5,16.6,31.6
    "Philox(n=4, w=32)",4.2,7.6,8.5,10.7,16.5,32.6
    SPECK128,5.4,8.1,9.6,11.3,17.0,33.3
    ThreeFry,5.8,9.1,9.3,12.0,16.7,34.9
    ChaCha(rounds=8),6.7,10.2,10.2,13.2,18.0,36.4
    ChaCha,9.6,16.4,16.4,19.6,24.3,48.9
    "ThreeFry(n=4, w=32)",9.0,16.3,17.6,20.7,24.4,53.4
    RDRAND,129.9,133.1,130.7,139.0,140.8,298.6

The next table presents the performance relative to NumPy's ``RandomState`` in
percentage. The overall performance is computed using a geometric mean.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
   :widths: 30,10,10,10,10,10,10,10

    SFC64,128,190,246,283,210,211,205
    Xoshiro256,120,186,229,280,206,207,198
    JSF,126,155,193,251,197,198,183
    EFIIX64,120,154,193,265,191,194,181
    PCG64DXSM,128,163,191,248,164,192,177
    PCG64,129,149,185,247,179,185,175
    Xoshiro512,111,132,177,251,195,195,171
    SFMT,104,142,186,229,183,190,167
    LXM,114,135,165,228,177,181,163
    PCG64(variant="dxsm-128"),105,139,165,235,161,172,158
    DSFMT,100,110,214,207,166,184,157
    JSF32,99,108,134,209,179,173,145
    MT64,105,117,137,211,157,167,145
    "Philox(n=2, w=64)",89,98,114,177,137,145,123
    Philox,75,79,95,165,148,147,112
    AESCounter,67,77,102,159,139,144,109
    MT19937,77,74,80,157,135,138,105
    "ThreeFry(n=2, w=64)",72,71,84,153,128,133,102
    HC128,72,65,80,137,121,125,96
    "Philox(n=4, w=32)",71,61,68,134,122,122,92
    SPECK128,55,57,60,127,118,119,83
    ThreeFry,51,53,63,120,120,114,81
    ChaCha(rounds=8),45,46,56,109,111,109,73
    ChaCha,31,28,35,74,83,81,50
    "ThreeFry(n=4, w=32)",33,28,33,70,83,76,49
    RDRAND,2,3,4,10,14,13,6

.. note::

   All timings were taken using Linux on an Intel Cascade Lake (Family 6,
   Model 85, Stepping 7) running at 3.1GHz.
