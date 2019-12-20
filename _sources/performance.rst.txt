Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is
:class:`~randomgen.xoroshiro128.Xoroshiro128`. The recommended generator
for use in large-scale parallel applications is
:class:`~randomgen.xoshiro256.Xoshiro256`
where the `jump` method is used to advance the state. For very large scale
applications -- requiring 1,000+ independent streams,
:class:`~randomgen.pcg64.PCG64` or :class:`~randomgen.threefry.Philox` are
the best choices.

Timings
*******

The timings below are the time in ns to produce 1 random value from a
specific distribution. :class:`~randomgen.xoroshiro128.Xoroshiro128` is the
fastest, followed by :class:`~randomgen.xorshift1024.Xorshift1024` and
:class:`~randomgen.pcg64.PCG64`. The original :class:`~randomgen.mt19937.MT19937`
generator is much slower since it requires 2 32-bit values to equal the output
of the faster generators.

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

    Xoshiro256,3.2,4.2,3.8,4.6,8.3,26.2
    JSF,3.4,4.6,3.8,5.1,8.9,27.1
    SFMT,3.9,4.5,3.9,5.0,9.4,27.7
    PCG64,3.6,4.7,4.1,5.6,10.2,28.0
    Xoshiro512,3.8,5.5,4.4,5.6,9.4,28.4
    DSFMT,4.1,6.5,3.7,7.7,12.3,29.5
    JSF32,4.3,5.7,5.4,6.7,10.8,30.9
    MT64,4.5,5.8,5.3,7.0,12.3,32.4
    MT19937,4.2,7.0,7.8,8.5,13.7,36.0
    AESCounter,4.9,8.0,7.6,8.6,13.8,35.7
    Philox2x64,5.5,7.4,7.5,9.4,15.1,37.3
    Philox,5.7,8.1,8.5,8.8,13.7,35.9
    ChaCha8,6.0,10.2,9.7,11.6,16.6,41.5
    ThreeFry2x64,6.3,9.5,9.8,11.5,17.0,41.3
    ThreeFry,6.4,10.2,10.1,12.0,15.4,42.8
    HC128,6.7,10.1,10.0,11.6,17.4,42.2
    Philox4x32,5.9,11.0,11.0,12.4,17.3,43.1
    SPECK128,8.1,13.5,13.3,14.1,19.0,46.6
    NumPy,3.4,7.5,8.7,41.8,36.8,60.5
    ChaCha,9.7,17.5,16.8,19.1,23.9,56.5
    ThreeFry4x32,10.1,19.8,21.2,22.6,25.6,66.6
    RDRAND,38.5,40.6,40.4,42.3,43.8,108.6


The next table presents the performance relative to NumPy's 1.16 `RandomState` in
percentage. The overall performance is computed using a geometric mean.

.. csv-table::
    :header: Bit Gen,Uint32,Uint64,Uniform,Expon,Normal,Gamma,Overall
    :widths: 30,10,10,10,10,10,10,10

    Xoshiro256,104,181,226,904,445,231,271
    JSF,99,164,230,813,412,223,255
    SFMT,87,168,225,831,390,219,248
    PCG64,93,162,210,744,360,216,238
    Xoshiro512,88,137,196,741,390,213,229
    DSFMT,81,116,233,542,299,205,204
    JSF32,78,132,160,624,339,195,202
    MT64,75,130,165,596,299,187,194
    MT19937,80,107,111,493,269,168,166
    AESCounter,68,94,114,484,267,169,158
    Philox2x64,61,101,116,446,244,162,153
    Philox,59,93,102,476,269,168,152
    ChaCha8,56,74,90,359,221,146,127
    ThreeFry2x64,53,79,88,363,217,146,127
    ThreeFry,52,74,86,349,239,141,125
    HC128,50,74,87,360,212,143,123
    Philox4x32,57,69,79,336,212,140,121
    SPECK128,42,56,66,296,193,130,102
    ChaCha,35,43,52,219,154,107,81
    ThreeFry4x32,33,38,41,185,144,91,71
    RDRAND,9,19,21,99,84,56,34


.. note::

   All timings were taken using Linux on a i5-3570 processor.
