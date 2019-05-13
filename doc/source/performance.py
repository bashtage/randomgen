from collections import OrderedDict
from timeit import repeat

import numpy as np
import pandas as pd

from randomgen import MT19937, DSFMT, ThreeFry, PCG64, Xoroshiro128, \
    Xorshift1024, Philox, Xoshiro256StarStar, Xoshiro512StarStar

NUMBER = 100
REPEAT = 10
SIZE = 10000
PRNGS = [DSFMT, MT19937, Philox, PCG64, ThreeFry, Xoroshiro128, Xorshift1024,
         Xoshiro256StarStar, Xoshiro512StarStar]

funcs = {'32-bit Unsigned Ints':
         f'integers(2**32, dtype="uint32", size={SIZE})',
         '64-bit Unsigned Ints':
         f'_bit_generator._benchmark({SIZE}, "uint64")',
         'Uniforms': f'_bit_generator._benchmark({SIZE}, "double")',
         'Complex Normals': f'complex_normal(size={SIZE})',
         'Normals': f'standard_normal(size={SIZE})',
         'Exponentials': f'standard_exponential(size={SIZE})',
         'Gammas': f'standard_gamma(3.0,size={SIZE})',
         'Binomials': f'binomial(9, .1, size={SIZE})',
         'Laplaces': f'laplace(size={SIZE})',
         'Poissons': f'poisson(3.0, size={SIZE})', }


setup = """
from randomgen import {prng}
rg = {prng}().generator
"""

test = "rg.{func}"
table = OrderedDict()
for prng in PRNGS:
    print(prng.__name__)
    print('-'*40)
    col = OrderedDict()
    for key in funcs:
        print(key)
        t = repeat(test.format(func=funcs[key]),
                   setup.format(prng=prng().__class__.__name__),
                   number=NUMBER, repeat=REPEAT)
        col[key] = 1000 * min(t)
    print('\n'*2)
    col = pd.Series(col)
    table[prng().__class__.__name__] = col

npfuncs = OrderedDict()
npfuncs.update(funcs)
del npfuncs['Complex Normals']
npfuncs.update({'64-bit Unsigned Ints':
                f'integers(2**64, dtype="uint64", size={SIZE})',
                'Uniforms': f'random_sample(size={SIZE})'})

setup = """
from numpy.random import RandomState
rg = RandomState()
"""
col = {}
for key in npfuncs:
    t = repeat(test.format(func=npfuncs[key]),
               setup.format(prng=prng().__class__.__name__),
               number=NUMBER, repeat=REPEAT)
    col[key] = 1000 * min(t)
table['NumPy'] = pd.Series(col)

table = pd.DataFrame(table)
table = table.reindex(table.mean(1).sort_values().index)
order = np.log(table).mean().sort_values().index
table = table.T
table = table.reindex(order)
table = table.T
table.pop('Xoshiro512StarStar')
table.pop('DSFMT')
table = 1000000 * table / (SIZE * NUMBER)
print(table.to_csv(float_format='%0.1f'))

rel = table.loc[:, ['NumPy']].values @ np.ones((1, table.shape[1])) / table
rel.pop('NumPy')
rel = rel.T
rel['Overall'] = np.exp(np.log(rel).mean(1))
rel *= 100
rel = np.round(rel)
rel = rel.T
print(rel.to_csv(float_format='%0d'))
