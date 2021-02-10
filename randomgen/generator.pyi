from threading import Lock
from typing import Any, Dict, Literal, Optional, Sequence, Union

import numpy as np
from numpy import ndarray

from randomgen.common import BitGenerator
from randomgen.typing import Size

class Generator:
    _bit_generator: BitGenerator
    lock: Lock
    _poisson_lam_max: int
    def __init__(self, bit_generator: Optional[BitGenerator] = None) -> None: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    def seed(self, *args, **kwargs): ...
    @property
    def state(self) -> Dict[str, Any]: ...
    @state.setter
    def state(self, value: Dict[str, Any]) -> None: ...
    def uintegers(
        self, size: Size = None, bits: Literal[32, 64] = 64
    ) -> Union[int, ndarray]: ...
    def random_uintegers(
        self, size: Size = None, bits: Literal[32, 64] = 64
    ) -> Union[int, ndarray]: ...
    def random_sample(self, *args, **kwargs) -> Union[float, ndarray]: ...
    def random(
        self, size: Size = None, dtype=np.float64, out: ndarray = None
    ) -> Union[float, ndarray]: ...
    def beta(self, a, b, size: Size = None) -> Union[float, ndarray]: ...
    def exponential(self, scale=1.0, size: Size = None) -> Union[float, ndarray]: ...
    def standard_exponential(
        self, size: Size = None, dtype=np.float64, method=u"zig", out: ndarray = None
    ) -> Union[float, ndarray]: ...
    def tomaxint(self, size: Size = None) -> Union[int, ndarray]: ...
    def randint(self, *args, **kwargs) -> Union[int, ndarray]: ...
    def integers(
        self,
        low,
        high=None,
        size: Size = None,
        dtype=np.int64,
        use_masked=None,
        endpoint=False,
        closed=None,
    ) -> Union[int, ndarray]: ...
    def bytes(self, length) -> ndarray: ...
    def choice(
        self, a, size: Size = None, replace=True, p=None, axis=0, shuffle=True
    ): ...
    def uniform(
        self, low=0.0, high=1.0, size: Size = None
    ) -> Union[float, ndarray]: ...
    def rand(self, *args, dtype=np.float64) -> Union[float, ndarray]: ...
    def randn(self, *args, dtype=np.float64) -> Union[float, ndarray]: ...
    def random_integers(
        self, low, high=None, size: Size = None
    ) -> Union[int, ndarray]: ...
    # Complicated, continuous distributions:...
    def standard_normal(
        self, size: Size = None, dtype=np.float64, out: ndarray = None
    ) -> Union[float, ndarray]: ...
    def normal(
        self, loc=0.0, scale=1.0, size: Size = None
    ) -> Union[float, ndarray]: ...
    def standard_gamma(
        self, shape, size: Size = None, dtype=np.float64, out: ndarray = None
    ) -> Union[float, ndarray]: ...
    def gamma(self, shape, scale=1.0, size: Size = None) -> Union[float, ndarray]: ...
    def f(self, dfnum, dfden, size: Size = None) -> Union[float, ndarray]: ...
    def noncentral_f(
        self, dfnum, dfden, nonc, size: Size = None
    ) -> Union[float, ndarray]: ...
    def chisquare(self, df, size: Size = None) -> Union[float, ndarray]: ...
    def noncentral_chisquare(
        self, df, nonc, size: Size = None
    ) -> Union[float, ndarray]: ...
    def standard_cauchy(self, size: Size = None) -> Union[float, ndarray]: ...
    def standard_t(self, df, size: Size = None) -> Union[float, ndarray]: ...
    def vonmises(self, mu, kappa, size: Size = None) -> Union[float, ndarray]: ...
    def pareto(self, a, size: Size = None) -> Union[float, ndarray]: ...
    def weibull(self, a, size: Size = None) -> Union[float, ndarray]: ...
    def power(self, a, size: Size = None) -> Union[float, ndarray]: ...
    def laplace(
        self, loc=0.0, scale=1.0, size: Size = None
    ) -> Union[float, ndarray]: ...
    def gumbel(
        self, loc=0.0, scale=1.0, size: Size = None
    ) -> Union[float, ndarray]: ...
    def logistic(
        self, loc=0.0, scale=1.0, size: Size = None
    ) -> Union[float, ndarray]: ...
    def lognormal(
        self, mean=0.0, sigma=1.0, size: Size = None
    ) -> Union[float, ndarray]: ...
    def rayleigh(self, scale=1.0, size: Size = None) -> Union[float, ndarray]: ...
    def wald(self, mean, scale, size: Size = None) -> Union[float, ndarray]: ...
    def triangular(
        self, left, mode, right, size: Size = None
    ) -> Union[float, ndarray]: ...
    # Complicated, discrete distributions:
    def binomial(self, n, p, size: Size = None) -> Union[int, ndarray]: ...
    def negative_binomial(self, n, p, size: Size = None) -> Union[int, ndarray]: ...
    def poisson(self, lam=1.0, size: Size = None) -> Union[int, ndarray]: ...
    def zipf(self, a, size: Size = None) -> Union[int, ndarray]: ...
    def geometric(self, p, size: Size = None) -> Union[int, ndarray]: ...
    def hypergeometric(
        self, ngood, nbad, nsample, size: Size = None
    ) -> Union[int, ndarray]: ...
    def logseries(self, p, size: Size = None) -> Union[int, ndarray]: ...
    # Multivariate distributions:
    def multivariate_normal(
        self,
        mean,
        cov,
        size: Size = None,
        check_valid="warn",
        tol=1e-8,
        *,
        method="svd"
    ) -> ndarray: ...
    def multinomial(self, n, pvals, size: Size = None) -> ndarray: ...
    def dirichlet(self, alpha, size: Size = None) -> ndarray: ...
    # Shuffling and permutations:
    def shuffle(self, x: Sequence[Any]) -> None: ...
    def permutation(self, x: Sequence[Any]) -> None: ...
    def complex_normal(
        self, loc=0.0, gamma=1.0, relation=0.0, size: Size = None
    ) -> Union[complex, ndarray]: ...

class ExtendedGenerator:
    _bit_generator: BitGenerator
    lock: Lock
    _generator: Generator
    def __init__(self, bit_generator: Optional[BitGenerator] = None) -> None: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    @property
    def state(self) -> Dict[str, Any]: ...
    @state.setter
    def state(self, value: Dict[str, Any]) -> None: ...
    def uintegers(
        self, size: Size = None, bits: Literal[32, 64] = 64
    ) -> Union[int, ndarray]: ...
    # Multivariate distributions:
    def multivariate_normal(
        self,
        mean,
        cov,
        size: Size = None,
        check_valid="warn",
        tol=1e-8,
        *,
        method="svd"
    ) -> ndarray: ...
    def complex_normal(
        self, loc=0.0, gamma=1.0, relation=0.0, size: Size = None
    ) -> Union[complex, ndarray]: ...

_random_generator: Generator

beta = _random_generator.beta
binomial = _random_generator.binomial
bytes = _random_generator.bytes
chisquare = _random_generator.chisquare
choice = _random_generator.choice
complex_normal = _random_generator.complex_normal
dirichlet = _random_generator.dirichlet
exponential = _random_generator.exponential
f = _random_generator.f
gamma = _random_generator.gamma
geometric = _random_generator.geometric
gumbel = _random_generator.gumbel
hypergeometric = _random_generator.hypergeometric
integers = _random_generator.integers
laplace = _random_generator.laplace
logistic = _random_generator.logistic
lognormal = _random_generator.lognormal
logseries = _random_generator.logseries
multinomial = _random_generator.multinomial
multivariate_normal = _random_generator.multivariate_normal
negative_binomial = _random_generator.negative_binomial
noncentral_chisquare = _random_generator.noncentral_chisquare
noncentral_f = _random_generator.noncentral_f
normal = _random_generator.normal
pareto = _random_generator.pareto
permutation = _random_generator.permutation
poisson = _random_generator.poisson
power = _random_generator.power
rand = _random_generator.rand
randint = _random_generator.randint
randn = _random_generator.randn
random_integers = _random_generator.random_integers
random_sample = _random_generator.random_sample
random = _random_generator.random
rayleigh = _random_generator.rayleigh
shuffle = _random_generator.shuffle
standard_cauchy = _random_generator.standard_cauchy
standard_exponential = _random_generator.standard_exponential
standard_gamma = _random_generator.standard_gamma
standard_normal = _random_generator.standard_normal
standard_t = _random_generator.standard_t
tomaxint = _random_generator.tomaxint
triangular = _random_generator.triangular
uniform = _random_generator.uniform
vonmises = _random_generator.vonmises
wald = _random_generator.wald
weibull = _random_generator.weibull
zipf = _random_generator.zipf
