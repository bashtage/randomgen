#!python
# cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True

from typing import Any, Callable


cdef class RandomState:
    """
    RandomState(bit_generator=None)

    RandomState has been removed in the 1.23 release.

    Use ``numpy.random.Generator`` or ``numpy.random.RandomState``
    if backward compataibility to older versions of NumPy is required.

    See Also
    --------
    numpy.random.Generator
    numpy.random.default_rng
    numpy.random.RandomState
    randomgen.generator.ExtendedGenerator
    """


def _removed(name: str) -> Callable[[Any, ...], None]:
    def f(*args, **kwargs):
        raise NotImplementedError(
            f"{name} has been removed. Use NumPy's Generator"
        )
    return f


beta = _removed("beta")
binomial = _removed("binomial")
bytes = _removed("bytes")
chisquare = _removed("chisquare")
choice = _removed("choice")
dirichlet = _removed("dirichlet")
exponential = _removed("exponential")
f = _removed("f")
gamma = _removed("gamma")
get_state = _removed("get_state")
geometric = _removed("geometric")
gumbel = _removed("gumbel")
hypergeometric = _removed("hypergeometric")
laplace = _removed("laplace")
logistic = _removed("logistic")
lognormal = _removed("lognormal")
logseries = _removed("logseries")
multinomial = _removed("multinomial")
multivariate_normal = _removed("multivariate_normal")
negative_binomial = _removed("negative_binomial")
noncentral_chisquare = _removed("noncentral_chisquare")
noncentral_f = _removed("noncentral_f")
normal = _removed("normal")
pareto = _removed("pareto")
permutation = _removed("permutation")
poisson = _removed("poisson")
power = _removed("power")
rand = _removed("rand")
randint = _removed("randint")
randn = _removed("randn")
random = _removed("random_sample")
random_integers = _removed("random_integers")
random_sample = _removed("random_sample")
rayleigh = _removed("rayleigh")
seed = _removed("seed")
set_state = _removed("set_state")
shuffle = _removed("shuffle")
standard_cauchy = _removed("standard_cauchy")
standard_exponential = _removed("standard_exponential")
standard_gamma = _removed("standard_gamma")
standard_normal = _removed("standard_normal")
standard_t = _removed("standard_t")
triangular = _removed("triangular")
uniform = _removed("uniform")
vonmises = _removed("vonmises")
wald = _removed("wald")
weibull = _removed("weibull")
zipf = _removed("zipf")
sample = _removed("sample")
ranf = _removed("ranf")


__all__ = [
    "beta",
    "binomial",
    "bytes",
    "chisquare",
    "choice",
    "dirichlet",
    "exponential",
    "f",
    "gamma",
    "geometric",
    "get_state",
    "gumbel",
    "hypergeometric",
    "laplace",
    "logistic",
    "lognormal",
    "logseries",
    "multinomial",
    "multivariate_normal",
    "negative_binomial",
    "noncentral_chisquare",
    "noncentral_f",
    "normal",
    "pareto",
    "permutation",
    "poisson",
    "power",
    "rand",
    "randint",
    "randn",
    "random_integers",
    "random_sample",
    "ranf",
    "rayleigh",
    "sample",
    "seed",
    "set_state",
    "shuffle",
    "standard_cauchy",
    "standard_exponential",
    "standard_gamma",
    "standard_normal",
    "standard_t",
    "triangular",
    "uniform",
    "vonmises",
    "wald",
    "weibull",
    "zipf",
    "RandomState",
]
