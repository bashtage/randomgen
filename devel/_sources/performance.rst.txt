Performance
-----------

.. py:module:: randomgen

Recommendation
**************
The recommended generator for single use is
:class:`~randomgen.xoroshiro128.Xoroshiro128`. The recommended generator
for use in large-scale parallel applications is
:class:`~randomgen.xoshiro256starstar.Xoshiro256StarStar`
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
performance gap for Exponentials is also large due to the cost of computing
the log function to invert the CDF.

.. csv-table::
    :header: ,Xoroshiro128,Xoshiro256**,Xorshift1024,PCG64,MT19937,Philox,ThreeFry,NumPy
    :widths: 14,14,14,14,14,14,14,14,14

    64-bit Unsigned Ints,11.9,13.6,14.9,16.0,18.0,22.0,25.9,42.0
    Uniforms,16.3,15.6,16.0,15.3,19.1,23.5,25.5,44.1
    32-bit Unsigned Ints,21.6,23.7,23.1,25.6,23.6,27.9,32.3,17.9
    Exponentials,21.2,22.4,23.8,21.5,26.7,30.8,33.0,115.3
    Normals,25.1,26.9,26.2,26.8,31.7,32.6,37.8,106.8
    Binomials,72.4,73.0,71.9,72.6,77.4,80.0,83.1,101.9
    Complex Normals,80.4,86.4,81.1,84.4,93.4,96.3,105.5,
    Laplaces,97.0,97.4,99.6,96.0,109.8,102.3,105.1,125.3
    Gammas,91.3,91.2,94.8,93.0,101.7,108.7,113.8,187.9
    Poissons,136.7,137.6,139.7,135.5,161.9,171.0,181.0,265.1


The next table presents the performance relative to `xoroshiro128+` in
percentage. The overall performance was computed using a geometric mean.

.. csv-table::
    :header: ,Xoroshiro128,Xoshiro256**,Xorshift1024,PCG64,MT19937,Philox,ThreeFry
    :widths: 14,14,14,14,14,14,14,14

    64-bit Unsigned Ints,353,309,283,263,233,191,162
    Uniforms,271,283,276,288,232,188,173
    32-bit Unsigned Ints,83,76,78,70,76,64,56
    Exponentials,544,514,485,536,432,375,350
    Normals,425,397,407,398,337,328,283
    Binomials,141,140,142,140,132,127,123
    Laplaces,129,129,126,131,114,123,119
    Gammas,206,206,198,202,185,173,165
    Poissons,194,193,190,196,164,155,146
    Overall,223,215,210,211,186,170,156

.. note::

   All timings were taken using Linux on a i5-3570 processor.
