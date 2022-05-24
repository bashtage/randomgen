from itertools import product
from typing import cast

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from packaging.version import parse
import pytest

from randomgen import Generator
import randomgen.common

try:
    from numpy.random import PCG64, Generator as NPGenerator

    pcg = PCG64()
    initial_state = pcg.state
    np_gen = NPGenerator(pcg)
    gen = Generator(cast(randomgen.common.BitGenerator, pcg))
except ImportError:
    from randomgen import PCG64  # type: ignore[misc]


NP_LT_1174 = parse(np.__version__) < parse("1.17.4")
NP_GTE_118 = parse(np.__version__) >= parse("1.18")
NP_GTE_120 = parse(np.__version__) >= parse("1.20")
NP_GTE_121 = parse(np.__version__) >= parse("1.21")

pytestmark = pytest.mark.skipif(NP_LT_1174, reason="Only test 1.17.4+")


def positive_param():
    base = Generator(PCG64())
    return [
        base.chisquare(10),
        base.chisquare(10, (5, 1, 3)),
        base.chisquare(10, (6, 5, 4, 3)),
    ]


def positive(num_args):
    args = list(product(*[positive_param() for _ in range(num_args)]))

    def param_generator():
        return args

    return param_generator


def int_prob():
    base = Generator(PCG64())
    return (
        [100, 0.5],
        [100, 0.5, (6, 5, 4, 3)],
        [base.integers(10, 100, size=(10, 2)), 0.3],
        [10, base.random((20, 2, 2))],
        [base.integers(10, 100, size=(5, 4, 3)), base.random(3)],
    )


def prob():
    base = Generator(PCG64())
    return (
        [0.5],
        [0.5, (6, 5, 4, 3)],
        [0.3],
        [base.random((20, 2, 2))],
        [base.random(3)],
    )


def length():
    return [(100,), (2500,)]


def input_0():
    return (tuple([]), (5,), ((5, 4, 3),))


def loc_scale():
    return positive(2)()


def above_1():
    return [(1 + val,) for val in positive_param()]


def triangular():
    out = product(*[positive_param() for _ in range(3)])
    out = [(lft, lft + mid, lft + mid + rgt) for lft, mid, rgt in out]
    return out


def uniform():
    low = positive_param()
    high = positive_param()
    scale = positive_param()
    out = []
    for lo, hi, sc in zip(low, high, scale):
        out.append((lo, lo + hi + sc))
        assert np.all(out[-1][1] >= out[-1][0])
    return out


def integers():
    dtypes = [
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ]
    base = Generator(PCG64())
    shape = tuple(base.integers(5, 10, size=2))
    configs = []

    for dt in dtypes:
        s1 = np.ones(shape, dtype=dt)
        s2 = np.ones((1,) + shape, dtype=dt)
        lo = np.iinfo(dt).min
        hi = np.iinfo(dt).max
        configs.extend(
            [
                (0, np.iinfo(dt).max, None, dt),
                (lo, hi // 2, None, dt),
                (lo, hi, (10, 2), dt),
                (lo // 2 * s1, hi // 2 * s2, None, dt),
            ]
        )
    return configs


def dirichlet():
    base = Generator(PCG64())
    probs = base.random(10)
    probs = probs / probs.sum()
    return [(probs,), (probs, (3, 4, 5))]


def hypergeometric():
    base = Generator(PCG64())
    good = [10, base.integers(10, 100, size=(3, 4))]
    bad = [10, base.integers(10, 100, size=(1, 4))]
    out = []
    for g, b in product(good, bad):
        nsample = g + b // 2
        if isinstance(nsample, int):
            nsample = max(nsample, 1)
        else:
            nsample.flat[nsample.flat < 1] = 1
        out.append((g, b, nsample))
    return out


def multinomial():
    base = Generator(PCG64())
    probs = base.random(10)
    probs /= probs.sum()
    return (10, probs), (base.integers(10, 100, size=(3, 4)), probs)


distributions = {
    "beta": positive(2),
    "binomial": int_prob,
    "bytes": length,
    "chisquare": positive(1),
    "dirichlet": dirichlet,
    "exponential": positive(1),
    "f": positive(2),
    "gamma": positive(2),
    "gumbel": positive(2),
    "laplace": loc_scale,
    "logistic": loc_scale,
    "lognormal": loc_scale,
    "logseries": prob,
    "multinomial": multinomial,
    "multivariate_normal": "",
    "negative_binomial": int_prob,
    "noncentral_chisquare": positive(2),
    "noncentral_f": positive(3),
    "normal": loc_scale,
    "pareto": positive(1),
    "poisson": positive(1),
    "power": positive(1),
    "random": input_0,
    "standard_cauchy": input_0,
    "standard_exponential": input_0,
    "standard_gamma": positive(1),
    "standard_normal": input_0,
    "standard_t": positive(1),
    "triangular": triangular,
    "uniform": uniform,
    "vonmises": loc_scale,
    "wald": positive(2),
    "weibull": positive(1),
    "zipf": above_1,
}


if not NP_GTE_121:
    distributions.update({"geometric": prob, "rayleigh": positive(1)})

tests = []
ids = []
for key in distributions:
    if not distributions[key]:
        continue
    params = distributions[key]()
    for i, param in enumerate(params):
        tests.append((key, param))
        ids.append(key + "-config-{0}".format(i))


@pytest.mark.parametrize("distribution, args", tests, ids=ids)
def test_equivalence(distribution, args):
    np_gen.bit_generator.state = initial_state
    np_rvs = getattr(np_gen, distribution)
    rvs = getattr(gen, distribution)
    expected = np_rvs(*args)

    gen.bit_generator.state = initial_state
    result = rvs(*args)
    if isinstance(result, (np.ndarray, float)):
        dtype = getattr(result, "dtype", None)
        if isinstance(result, float) or dtype in (np.float32, np.float64):
            assert_allclose(result, expected)
        else:
            assert_array_equal(result, expected)
    else:
        assert result == expected


def test_shuffle():
    np_gen.bit_generator.state = initial_state
    expected = np.arange(100)
    np_gen.shuffle(expected)

    gen.bit_generator.state = initial_state
    result = np.arange(100)
    gen.shuffle(result)
    assert_array_equal(result, expected)


def test_permutation():
    np_gen.bit_generator.state = initial_state
    expected = np_gen.permutation(100)

    gen.bit_generator.state = initial_state
    result = gen.permutation(100)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("replace", [True, False])
def test_choice_with_p(replace):
    x = np.arange(100)
    np_gen.bit_generator.state = initial_state
    p = (x + 1) / (x + 1).sum()
    expected = np_gen.choice(x, size=10, replace=replace, p=p)

    gen.bit_generator.state = initial_state
    result = gen.choice(x, size=10, replace=replace, p=p)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("replace", [True, False])
def test_choice(replace):
    np_gen.bit_generator.state = initial_state
    x = np.arange(100)
    expected = np_gen.choice(x, size=10, replace=replace)

    gen.bit_generator.state = initial_state
    result = gen.choice(x, size=10, replace=replace)
    assert_array_equal(result, expected)


configs = integers()


@pytest.mark.skipif(NP_LT_1174, reason="Changes to lemire generators")
@pytest.mark.parametrize("args", configs)
def test_integers(args):
    np_gen.bit_generator.state = initial_state
    expected = np_gen.integers(*args)

    gen.bit_generator.state = initial_state
    result = gen.integers(*args, use_masked=False)
    assert_array_equal(result, expected)


@pytest.mark.parametrize("args", hypergeometric())
def test_hypergeometric(args):
    np_gen.bit_generator.state = initial_state
    expected = np_gen.hypergeometric(*args)

    gen.bit_generator.state = initial_state
    result = gen.hypergeometric(*args)
    assert_allclose(result, expected)


def test_missing():
    KNOWN_SPECIAL_CASES = [
        "bit_generator",
        "choice",
        "hypergeometric",
        "integers",
        "permutation",
        "shuffle",
    ]
    if NP_GTE_121:
        KNOWN_SPECIAL_CASES += ["geometric", "rayleigh"]
    missing = [
        f
        for f in dir(np_gen)
        if not f.startswith("_")
        and f not in distributions
        and f not in KNOWN_SPECIAL_CASES
    ]
    missing_funcs = []
    if NP_GTE_118:
        missing_funcs += ["multivariate_hypergeometric"]
    if NP_GTE_120:
        missing_funcs += ["permuted"]
    assert missing == missing_funcs
