from collections import OrderedDict
from timeit import repeat

import numpy as np
import pandas as pd

from randomgen import (
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    LXM,
    MT64,
    MT19937,
    PCG64,
    PCG64DXSM,
    RDRAND,
    SFC64,
    SFMT,
    SPECK128,
    AESCounter,
    ChaCha,
    Philox,
    Romu,
    ThreeFry,
    Xoshiro256,
    Xoshiro512,
)


class ChaCha8(ChaCha):
    canonical_repr = "ChaCha(rounds=8)"

    def __init__(self, *args, **kwargs):
        if "rounds" in kwargs:
            del kwargs["rounds"]
        super().__init__(*args, rounds=8, **kwargs)


class JSF32(JSF):
    def __init__(self, *args, **kwargs):
        if "size" in kwargs:
            del kwargs["size"]
        super().__init__(*args, size=32, **kwargs)


class Philox4x32(Philox):
    canonical_repr = "Philox(n=4, w=32)"

    def __init__(self, *args, **kwargs):
        if "width" in kwargs:
            del kwargs["width"]
        super().__init__(*args, width=32, **kwargs)


class Philox2x64(Philox):
    canonical_repr = "Philox(n=2, w=64)"

    def __init__(self, *args, **kwargs):
        if "number" in kwargs:
            del kwargs["number"]
        super().__init__(*args, number=2, **kwargs)


class RomuTrio(Romu):
    canonical_repr = 'Romu(variant="trio")'

    def __init__(self, *args, **kwargs):
        if "variant" in kwargs:
            del kwargs["variant"]
        super().__init__(*args, variant="trio")


class ThreeFry4x32(ThreeFry):
    canonical_repr = "ThreeFry(n=4, w=32)"

    def __init__(self, *args, **kwargs):
        if "width" in kwargs:
            del kwargs["width"]
        super().__init__(*args, width=32, **kwargs)


class ThreeFry2x64(ThreeFry):
    canonical_repr = "ThreeFry(n=2, w=64)"

    def __init__(self, *args, **kwargs):
        if "number" in kwargs:
            del kwargs["number"]
        super().__init__(*args, number=2, **kwargs)


class PCG64DXSM128(PCG64):
    canonical_repr = 'PCG64(variant="dxsm-128")'

    def __init__(self, *args, **kwargs):
        if "variant" in kwargs:
            del kwargs["variant"]
        super().__init__(*args, variant="dxsm-128", **kwargs)


try:
    RDRAND()
    HAS_RDRND = True
except RuntimeError:
    HAS_RDRND = False

NUMBER = 100
REPEAT = 10
SIZE = 25000
PRNGS = [
    ChaCha8,
    JSF32,
    Philox4x32,
    ThreeFry2x64,
    ThreeFry4x32,
    Philox2x64,
    DSFMT,
    MT64,
    MT19937,
    PCG64,
    PCG64DXSM128,
    PCG64DXSM,
    LXM,
    SFMT,
    AESCounter,
    ChaCha,
    Philox,
    ThreeFry,
    Xoshiro256,
    Xoshiro512,
    JSF,
    Romu,
    RomuTrio,
    HC128,
    SPECK128,
    SFC64,
    EFIIX64,
]

if HAS_RDRND:
    PRNGS.append(RDRAND)


funcs = OrderedDict()
funcs["Uint32"] = f'integers(2**32, dtype="uint32", size={SIZE})'
funcs["Uint64"] = f'integers(2**64, dtype="uint64", size={SIZE})'
funcs["Uniform"] = f"random(size={SIZE})"
funcs["Expon"] = f"standard_exponential(size={SIZE})"
funcs["Normal"] = f"standard_normal(size={SIZE})"
funcs["Gamma"] = f"standard_gamma(3.0,size={SIZE})"

setup = """
from numpy.random import Generator
rg = Generator({prng}())
"""

test = "rg.{func}"
table = OrderedDict()
for prng in PRNGS:
    print(prng.__name__)
    print("-" * 40)
    col = OrderedDict()
    for key in funcs:
        print(key)
        t = repeat(
            test.format(func=funcs[key]),
            setup.format(prng=prng().__class__.__name__),
            number=NUMBER,
            repeat=REPEAT,
            globals=globals(),
        )
        col[key] = 1000 * min(t)
    print("\n" * 2)
    col = pd.Series(col)
    class_name = type(prng()).__name__
    if hasattr(prng, "canonical_repr"):
        class_name = prng.canonical_repr
    table[class_name] = col

npfuncs = OrderedDict()
npfuncs.update(funcs)
npfuncs["Uniform"] = f"random_sample(size={SIZE})"
npfuncs["Uint64"] = f'randint(2**64, dtype="uint64", size={SIZE})'
npfuncs["Uint32"] = f'randint(2**32, dtype="uint32", size={SIZE})'


setup = """
from numpy.random import RandomState
rg = RandomState()
"""
col = {}
for key in npfuncs:
    t = repeat(
        test.format(func=npfuncs[key]),
        setup.format(prng=prng().__class__.__name__),
        number=NUMBER,
        repeat=REPEAT,
    )
    col[key] = 1000 * min(t)
table["NumPy"] = pd.Series(col)
final = table

func_list = list(funcs.keys())
table = pd.DataFrame(final)
table = table.reindex(table.mean(1).sort_values().index)
order = np.log(table).mean().sort_values().index
table = table.T
table = table.reindex(order, axis=0)
table = table.reindex(func_list, axis=1)
table = 1000000 * table / (SIZE * NUMBER)
table.index.name = "Bit Gen"
print(table.to_csv(float_format="%0.1f"))

try:
    from tabulate import tabulate

    perf = table.applymap(lambda v: f"{v:0.1f}")
    print(tabulate(perf, headers="keys", tablefmt="rst"))
except ImportError:
    pass

table = table.T
rel = table.loc[:, ["NumPy"]].values @ np.ones((1, table.shape[1])) / table
rel.pop("NumPy")
rel = rel.T
rel["Overall"] = np.exp(np.log(rel).mean(1))
rel *= 100
rel = np.round(rel).astype(int)
rel.index.name = "Bit Gen"
print(rel.to_csv(float_format="%0d"))

try:
    from tabulate import tabulate

    rel_perf = rel.applymap(lambda v: f"{v:d}")
    print(tabulate(rel_perf, headers="keys", tablefmt="rst"))
except ImportError:
    pass
