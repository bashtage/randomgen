Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is
:class:`~randomgen.pcg64.PCG64` with the keyword argument ``variant="cm-dxsm"`.
An excellent alternative is :class:`~randomgen.xoshiro256.Xoshiro256`
where the `jump` method is used to advance the state. For very large scale
applications -- requiring 1,000+ independent streams,
:class:`~randomgen.pcg64.PCG64` or one of the cryptographic-based generators
(:class:`~randomgen.aes.AESCounter` if you have hardware acceleration or
:class:`~randomgen.speck128.SPECK128`, :class:`~randomgen.philox.Philox`, or
:class:`~randomgen.threefry.ThreeFry` if you do not) are the best choices.

Timings
*******

The timings below are the time in ns to produce 1 random value from a
specific distribution. :class:`~randomgen.xoshiro256.Xoshiro256` is the
fastest, followed by :class:`~randomgen.jsf.JSF`,
:class:`~randomgen.sfmt.SFMT`, and :class:`~randomgen.pcg64.PCG64`. The original
:class:`~randomgen.mt19937.MT19937` generator is slower since it requires 2 32-bit values
to equal the output of the faster generators.

Integer performance has a similar ordering although `dSFMT` is slower since
it generates 53-bit floating point values rather than integer values. On the
other hand, it is very fast for uniforms, although slower than `xoroshiro128+`.

The pattern is similar for other, more complex generators. The normal
performance of NumPy's MT19937 is much lower than the other since it
uses the Box-Muller transformation rather than the Ziggurat generator. The
performance gap for Exponentials is also large due replacement of the use of the
log function to invert the CDF with a Ziggurat-based generator.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma
   :widths: 30,10,10,10,10,10,10

   Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma
   Xoshiro256,2.3,2.7,2.6,4.7,8.8,20.7
   JSF,2.3,3.1,3.0,5.4,9.1,20.4
   PCG64CMDXSM,2.5,3.0,3.1,4.9,10.2,21.2
   SFMT,2.8,3.0,3.0,5.2,9.8,20.9
   PCG64,2.3,3.2,3.1,5.0,10.2,21.8
   Xoshiro512,2.7,3.6,3.3,5.4,9.3,20.6
   PCG64DXSM,2.9,3.4,3.4,5.4,11.5,22.8
   LXM,2.6,4.1,3.5,6.3,11.1,22.6
   DSFMT,2.9,4.5,2.6,6.9,11.4,22.4
   MT64,2.9,4.0,3.8,6.3,12.0,23.7
   JSF32,3.0,4.4,4.3,6.8,10.8,23.2
   Philox2x64,3.4,5.0,5.1,7.8,14.1,27.6
   Philox,3.9,6.0,6.0,8.0,12.9,27.6
   AESCounter,4.1,6.2,6.4,8.7,13.9,28.2
   ThreeFry2x64,4.2,6.6,6.9,9.3,15.3,30.1
   MT19937,4.0,6.7,7.8,9.3,14.6,29.9
   HC128,4.1,7.3,7.5,9.8,15.8,32.2
   Philox4x32,4.3,7.7,8.5,10.4,15.8,33.5
   NumPy,3.0,5.0,5.7,20.1,25.2,40.2
   SPECK128,5.4,8.2,9.6,10.6,16.1,33.9
   ThreeFry,5.9,9.9,9.2,11.8,16.3,35.7
   ChaCha8,6.6,10.2,10.2,12.9,18.0,36.5
   ChaCha,9.7,16.4,16.3,19.2,24.2,49.0
   ThreeFry4x32,9.1,16.5,17.7,20.4,23.7,53.7
   RDRAND,131.3,131.5,131.2,138.1,139.9,293.9



The next table presents the performance relative to NumPy's 1.16 `RandomState` in
percentage. The overall performance is computed using a geometric mean.

.. csv-table::
   :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
   :widths: 30,10,10,10,10,10,10,10

   Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
   Xoshiro256,130,186,222,431,285,194,224
   JSF,128,163,192,373,275,198,208
   PCG64CMDXSM,120,168,188,407,247,190,204
   SFMT,108,164,194,389,258,193,201
   PCG64,128,155,183,402,246,184,201
   Xoshiro512,112,140,175,370,269,196,194
   PCG64DXSM,102,147,171,370,218,176,182
   LXM,116,123,164,318,227,178,176
   DSFMT,105,112,219,291,220,179,176
   MT64,105,125,153,321,210,170,168
   JSF32,98,113,133,294,232,174,161
   Philox2x64,87,101,114,257,178,146,137
   Philox,76,83,96,251,195,146,128
   AESCounter,74,80,90,231,182,143,121
   ThreeFry2x64,71,75,83,216,165,133,113
   MT19937,75,74,74,217,173,134,113
   HC128,72,68,77,205,159,125,107
   Philox4x32,70,65,68,193,159,120,102
   SPECK128,56,61,60,190,156,119,95
   ThreeFry,51,50,63,171,155,113,88
   ChaCha8,45,49,56,155,140,110,82
   ChaCha,31,30,35,105,104,82,56
   ThreeFry4x32,33,30,33,98,106,75,54
   RDRAND,2,4,4,15,18,14,7

.. note::

   All timings were taken using Linux on an Intel Cascade Lake (Family 6,
   Model 85, Stepping 7) running at 3.1GHz.

