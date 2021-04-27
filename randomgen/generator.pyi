from threading import Lock
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union, overload

import numpy as np
from numpy import ndarray

from randomgen.common import BitGenerator
from randomgen.typing import RequiredSize, Size

class Generator:
    _bit_generator: BitGenerator
    lock: Lock
    _poisson_lam_max: int
    def __init__(self, bit_generator: Optional[BitGenerator] = ...) -> None: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    def seed(self, *args: Any, **kwargs: Any) -> None: ...
    @property
    def state(self) -> Dict[str, Any]: ...
    @state.setter
    def state(self, value: Dict[str, Any]) -> None: ...
    def uintegers(
        self, size: Size = ..., bits: Literal[32, 64] = ...
    ) -> Union[int, ndarray]: ...
    def random_uintegers(
        self, size: Size = ..., bits: Literal[32, 64] = ...
    ) -> Union[int, ndarray]: ...
    def random_sample(
        self, *args: Tuple[int, ...], **kwargs: Dict[str, Tuple[int, ...]]
    ) -> Union[float, ndarray]: ...
    def random(
        self, size: Size = ..., dtype: str = ..., out: ndarray = ...
    ) -> Union[float, ndarray]: ...
    def beta(
        self, a: Union[float, ndarray], b: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def exponential(
        self, scale: Optional[Union[float, ndarray]] = ..., size: Size = ...
    ) -> Union[float, ndarray]: ...
    def standard_exponential(
        self, size: Size = ..., dtype: str = ..., method: str = ..., out: ndarray = ...
    ) -> Union[float, ndarray]: ...
    def tomaxint(self, size: Size = ...) -> Union[int, ndarray]: ...
    def randint(
        self,
        *args: Tuple[Union[int, Tuple[int, ...]], ...],
        **kwargs: Dict[str, Union[int, Tuple[int, ...]]]
    ) -> Union[int, ndarray]: ...
    def integers(
        self,
        low: Union[int, ndarray],
        high: Optional[Union[int, ndarray]] = ...,
        size: Size = ...,
        dtype: str = ...,
        use_masked: Optional[bool] = ...,
        endpoint: bool = ...,
        closed: bool = ...,
    ) -> Union[int, ndarray]: ...
    def bytes(self, length: int) -> ndarray: ...
    def choice(
        self,
        a: Union[int, Sequence[Any]],
        size: Size = ...,
        replace: bool = ...,
        p: Optional[ndarray] = ...,
        axis: int = ...,
        shuffle: bool = ...,
    ) -> Sequence[Any]: ...
    def uniform(
        self,
        low: Optional[Union[float, ndarray]] = ...,
        high: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def rand(
        self, *args: Tuple[int, ...], dtype: str = ...
    ) -> Union[float, ndarray]: ...
    def randn(
        self, *args: Tuple[int, ...], dtype: str = ...
    ) -> Union[float, ndarray]: ...
    def random_integers(
        self,
        low: Union[int, ndarray],
        high: Optional[Union[int, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[int, ndarray]: ...
    # Complicated, continuous distributions:...
    def standard_normal(
        self, size: Size = ..., dtype: str = ..., out: ndarray = ...
    ) -> Union[float, ndarray]: ...
    def normal(
        self,
        loc: Optional[Union[float, ndarray]] = ...,
        scale: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def standard_gamma(
        self,
        shape: Union[float, ndarray],
        size: Size = ...,
        dtype: str = ...,
        out: ndarray = ...,
    ) -> Union[float, ndarray]: ...
    def gamma(
        self,
        shape: Union[float, ndarray],
        scale: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def f(
        self,
        dfnum: Union[float, ndarray],
        dfden: Union[float, ndarray],
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def noncentral_f(
        self,
        dfnum: Union[float, ndarray],
        dfden: Union[float, ndarray],
        nonc: Union[float, ndarray],
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def chisquare(
        self, df: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def noncentral_chisquare(
        self, df: Union[float, ndarray], nonc: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def standard_cauchy(self, size: Size = ...) -> Union[float, ndarray]: ...
    def standard_t(
        self, df: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def vonmises(
        self, mu: Union[float, ndarray], kappa: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def pareto(
        self, a: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def weibull(
        self, a: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def power(
        self, a: Union[float, ndarray], size: Size = ...
    ) -> Union[float, ndarray]: ...
    def laplace(
        self,
        loc: Optional[Union[float, ndarray]] = ...,
        scale: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def gumbel(
        self,
        loc: Optional[Union[float, ndarray]] = ...,
        scale: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def logistic(
        self,
        loc: Optional[Union[float, ndarray]] = ...,
        scale: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def lognormal(
        self,
        mean: Optional[Union[float, ndarray]] = ...,
        sigma: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def rayleigh(
        self, scale: Optional[Union[float, ndarray]] = ..., size: Size = ...
    ) -> Union[float, ndarray]: ...
    def wald(
        self,
        mean: Union[float, ndarray],
        scale: Union[float, ndarray],
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    def triangular(
        self,
        left: Union[float, ndarray],
        mode: Union[float, ndarray],
        right: Union[float, ndarray],
        size: Size = ...,
    ) -> Union[float, ndarray]: ...
    # Complicated, discrete distributions:
    def binomial(
        self, n: Union[int, ndarray], p: Union[float, ndarray], size: Size = ...
    ) -> Union[int, ndarray]: ...
    def negative_binomial(
        self, n: Union[int, ndarray], p: Union[float, ndarray], size: Size = ...
    ) -> Union[int, ndarray]: ...
    def poisson(
        self, lam: Optional[Union[float, ndarray]] = ..., size: Size = ...
    ) -> Union[int, ndarray]: ...
    def zipf(
        self, a: Union[float, ndarray], size: Size = ...
    ) -> Union[int, ndarray]: ...
    def geometric(
        self, p: Union[float, ndarray], size: Size = ...
    ) -> Union[int, ndarray]: ...
    def hypergeometric(
        self,
        ngood: Union[int, ndarray],
        nbad: Union[int, ndarray],
        nsample: Union[int, ndarray],
        size: Size = ...,
    ) -> Union[int, ndarray]: ...
    def logseries(
        self, p: Union[float, ndarray], size: Size = ...
    ) -> Union[int, ndarray]: ...
    # Multivariate distributions:
    def multivariate_normal(
        self,
        mean: ndarray,
        cov: ndarray,
        size: Size = ...,
        check_valid: str = ...,
        tol: float = ...,
        *,
        method: str = ...
    ) -> ndarray: ...
    def multinomial(
        self, n: Union[int, ndarray], pvals: Union[float, ndarray], size: Size = ...
    ) -> ndarray: ...
    def dirichlet(self, alpha: ndarray, size: Size = ...) -> ndarray: ...
    # Shuffling and permutations:
    def shuffle(self, x: Sequence[Any]) -> None: ...
    def permutation(self, x: Sequence[Any]) -> None: ...
    def complex_normal(
        self,
        loc: Optional[Union[float, ndarray]] = ...,
        gamma: Optional[Union[float, ndarray]] = ...,
        relation: Optional[Union[float, ndarray]] = ...,
        size: Size = ...,
    ) -> Union[complex, ndarray]: ...

class ExtendedGenerator:
    _bit_generator: BitGenerator
    lock: Lock
    _generator: Generator
    def __init__(self, bit_generator: Optional[BitGenerator] = ...) -> None: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    @property
    def state(self) -> Dict[str, Any]: ...
    @state.setter
    def state(self, value: Dict[str, Any]) -> None: ...
    @overload
    def uintegers(self, size: None, bits: Literal[32, 64] = ...) -> int: ...
    @overload
    def uintegers(self, size: RequiredSize, bits: Literal[32, 64] = ...) -> ndarray: ...
    @overload
    def random(self) -> float: ...  # type: ignore[misc]
    @overload
    def random(self, size: None = ...) -> float: ...  # type: ignore[misc]
    @overload
    def random(
        self, size: Size = ..., dtype: str = ..., out: Optional[ndarray] = ...
    ) -> ndarray: ...
    # Multivariate distributions:
    def multivariate_normal(
        self,
        mean: ndarray,
        cov: ndarray,
        size: Size = ...,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        *,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...
    ) -> ndarray: ...
    @overload
    def complex_normal(  # type: ignore[misc]
        self,
        loc: complex = ...,
        gamma: complex = ...,
        relation: complex = ...,
    ) -> complex: ...
    @overload
    def complex_normal(  # type: ignore[misc]
        self,
        loc: complex = ...,
        gamma: complex = ...,
        relation: complex = ...,
        size: RequiredSize = ...,
    ) -> ndarray: ...
    @overload
    def complex_normal(
        self,
        loc: Union[complex, ndarray] = ...,
        gamma: Union[complex, ndarray] = ...,
        relation: Union[complex, ndarray] = ...,
        size: Size = ...,
    ) -> ndarray: ...
    def standard_wishart(
        self, df: int, dim: int, size: Size = ..., *, rescale: bool = ...
    ) -> ndarray: ...
    def wishart(
        self,
        df: Union[int, ndarray],
        scale: ndarray,
        size: Size = ...,
        *,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        rank: Optional[int] = ...,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...
    ) -> ndarray: ...
    def multivariate_complex_normal(
        self,
        loc: ndarray,
        gamma: Optional[ndarray] = ...,
        relation: Optional[ndarray] = ...,
        size: Size = ...,
        *,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...
    ) -> ndarray: ...

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
