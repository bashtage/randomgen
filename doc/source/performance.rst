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

============  ========  ========  =========  =======  ========  =======
Bit Gen         Uint32    Uint64    Uniform    Expon    Normal    Gamma
============  ========  ========  =========  =======  ========  =======
Xoshiro256         3.5       4.2        4        4.7       8.7     27.3
JSF                3.4       4.4        3.8      5.2       9       28
SFMT               3.9       4.4        3.8      5.1       9.6     27.5
PCG64              3.9       4.6        4.3      5.6      10.8     28.2
Xoshiro512         4         5.4        4.7      6.3       9.8     29.4
JSF32              4.4       5.6        5.4      6.7      10.8     31.4
DSFMT              4.2       6.6        3.7      8.1      12.4     29.9
MT64               4.6       5.8        5.3      7.1      12.3     32.8
MT19937            4.9       7.2        8        9.4      14.1     38.8
AESCounter         5.2       8.2        7.7      8.8      14.4     35.5
Philox2x64         5.5       7.4        7.6      9.7      15.3     37.8
Philox             5.8       7.8        8.8      9.2      14.6     38.3
ThreeFry2x64       6.4       9.7        9.8     11.7      16.7     40.6
HC128              6.6      10.3       10.3     11.7      17.4     42.9
Philox4x32         6        11.2       11.2     12.5      17.7     44.3
ThreeFry           6.5      10.3       10.8     13        16.9     48.9
SPECK128           8.3      13.5       13.5     14.2      19.4     47.2
NumPy              3.3       7.7        8.9     43        38.1     62.7
ChaCha             9.6      17.7       17.2     19.8      25.1     61.7
ThreeFry4x32      10.1      19.9       21.6     22.8      25.9     66.5
RDRAND            39.9      39.3       40.8     42.9      43.7    109.9
============  ========  ========  =========  =======  ========  =======


The next table presents the performance relative to NumPy's 1.16 `RandomState` in
percentage. The overall performance is computed using a geometric mean.

============  ========  ========  =========  =======  ========  =======  =========
Bit Gen         Uint32    Uint64    Uniform    Expon    Normal    Gamma    Overall
============  ========  ========  =========  =======  ========  =======  =========
Xoshiro256          94       183        225      907       440      230        266
JSF                 97       176        235      820       421      224        260
SFMT                86       174        233      851       398      228        254
PCG64               85       168        208      772       354      223        238
Xoshiro512          83       144        191      679       388      213        225
JSF32               76       137        165      643       353      200        207
DSFMT               79       117        240      533       307      210        206
MT64                73       134        168      604       311      191        197
MT19937             69       106        111      455       271      162        159
AESCounter          64        93        116      488       265      177        159
Philox2x64          60       104        117      444       248      166        154
Philox              58        99        101      466       260      164        150
ThreeFry2x64        52        80         91      368       228      155        130
HC128               51        75         87      367       219      146        125
Philox4x32          55        69         79      343       216      142        121
ThreeFry            51        75         83      330       225      128        120
SPECK128            40        57         66      302       197      133        103
ChaCha              35        44         52      217       151      102         80
ThreeFry4x32        33        39         41      188       147       94         72
RDRAND               8        20         22      100        87       57         35
============  ========  ========  =========  =======  ========  =======  =========

.. note::

   All timings were taken using Linux on a i5-3570 processor.
