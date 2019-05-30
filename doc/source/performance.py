from collections import OrderedDict
from timeit import repeat

import numpy as np
import pandas as pd

from randomgen import MT19937, DSFMT, ThreeFry, PCG64, Philox, Xoshiro256, \
    Xoshiro512, MT64, SFMT

NUMBER = 100
REPEAT = 10
SIZE = 25000
PRNGS = [DSFMT, MT19937, MT64, Philox, PCG64, ThreeFry, SFMT,
         Xoshiro256, Xoshiro512]

funcs = OrderedDict()
funcs['32-bit Unsigned Int'] = f'integers(2**32, dtype="uint32", size={SIZE})'
funcs['64-bit Unsigned Int'] = f'integers(2**64, dtype="uint64", size={SIZE})'
funcs['Uniform'] = f'random(size={SIZE})'
funcs['Exponential'] = f'standard_exponential(size={SIZE})'
funcs['Normal'] = f'standard_normal(size={SIZE})'
funcs['Gamma'] = f'standard_gamma(3.0,size={SIZE})'
funcs['Complex Normal'] = f'complex_normal(size={SIZE})'
funcs['Binomial'] = f'binomial(9, .1, size={SIZE})'
funcs['Laplace'] = f'laplace(size={SIZE})'
funcs['Poisson'] = f'poisson(3.0, size={SIZE})'

setup = """
from randomgen import {prng}, Generator
rg = Generator({prng}())
"""

test = "rg.{func}"
table = OrderedDict()
for prng in PRNGS:
    print(prng.__name__)
    print('-' * 40)
    col = OrderedDict()
    for key in funcs:
        print(key)
        t = repeat(test.format(func=funcs[key]),
                   setup.format(prng=prng().__class__.__name__),
                   number=NUMBER, repeat=REPEAT)
        col[key] = 1000 * min(t)
    print('\n' * 2)
    col = pd.Series(col)
    table[prng().__class__.__name__] = col

npfuncs = OrderedDict()
npfuncs.update(funcs)
del npfuncs['Complex Normal']
npfuncs['Uniform'] = f'random_sample(size={SIZE})'
npfuncs['64-bit Unsigned Int'] = f'randint(2**64, dtype="uint64", size={SIZE})'
npfuncs['32-bit Unsigned Int'] = f'randint(2**32, dtype="uint32", size={SIZE})'


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
table = 1000000 * table / (SIZE * NUMBER)
print(table.to_csv(float_format='%0.1f'))

try:
    from tabulate import tabulate

    print(tabulate(table, headers='keys', tablefmt='psql'))
except ImportError:
    pass

rel = table.loc[:, ['NumPy']].values @ np.ones((1, table.shape[1])) / table
rel.pop('NumPy')
rel = rel.T
rel['Overall'] = np.exp(np.log(rel).mean(1))
rel *= 100
rel = np.round(rel)
rel = rel.T
print(rel.to_csv(float_format='%0d'))

try:
    from tabulate import tabulate

    print(tabulate(rel, headers='keys', tablefmt='psql'))
except ImportError:
    pass
