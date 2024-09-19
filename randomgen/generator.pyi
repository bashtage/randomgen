from threading import Lock
from typing import Any, Literal, overload

from numpy import ndarray

from randomgen.common import BitGenerator
from randomgen.typing import RequiredSize, Size

class Generator:
    def __init__(self, bit_generator: BitGenerator | None = ...) -> None: ...

class ExtendedGenerator:
    _bit_generator: BitGenerator
    lock: Lock
    _generator: Generator
    def __init__(self, bit_generator: BitGenerator | None = ...) -> None: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    @property
    def state(self) -> dict[str, Any]: ...
    @state.setter
    def state(self, value: dict[str, Any]) -> None: ...
    @overload
    def uintegers(self, size: None, bits: Literal[32, 64] = ...) -> int: ...
    @overload
    def uintegers(self, size: RequiredSize, bits: Literal[32, 64] = ...) -> ndarray: ...
    @overload
    def random(self) -> float: ...
    @overload
    def random(self, size: None) -> float: ...
    @overload
    def random(
        self, size: RequiredSize, dtype: str = ..., out: ndarray | None = ...
    ) -> ndarray: ...
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
    def complex_normal(self, loc: complex) -> complex: ...
    @overload
    def complex_normal(self, loc: complex, gamma: complex) -> complex: ...
    @overload
    def complex_normal(self, loc: complex, *, relation: complex) -> complex: ...
    @overload
    def complex_normal(self, *, gamma: complex, relation: complex) -> complex: ...
    @overload
    def complex_normal(
        self, loc: complex, gamma: complex, relation: complex
    ) -> complex: ...
    @overload
    def complex_normal(self, loc: ndarray) -> ndarray: ...
    @overload
    def complex_normal(self, *, gamma: ndarray) -> ndarray: ...
    @overload
    def complex_normal(self, loc: complex | ndarray, gamma: ndarray) -> ndarray: ...
    @overload
    def complex_normal(
        self, *, gamma: ndarray, relation: complex | ndarray
    ) -> ndarray: ...
    @overload
    def complex_normal(
        self, loc: complex | ndarray, *, relation: ndarray
    ) -> ndarray: ...
    @overload
    def complex_normal(
        self, loc: complex | ndarray, gamma: complex | ndarray, *, relation: ndarray
    ) -> ndarray: ...
    @overload
    def complex_normal(self, *, relation: ndarray) -> ndarray: ...
    @overload
    def complex_normal(
        self,
        loc: complex | ndarray,
        gamma: complex | ndarray,
        relation: complex | ndarray,
        size: RequiredSize,
    ) -> ndarray: ...
    def standard_wishart(
        self, df: int, dim: int, size: Size = ..., *, rescale: bool = ...
    ) -> ndarray: ...
    def wishart(
        self,
        df: int | ndarray,
        scale: ndarray,
        size: Size = ...,
        *,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        rank: int | None = ...,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...
    ) -> ndarray: ...
    def multivariate_complex_normal(
        self,
        loc: ndarray,
        gamma: ndarray | None = ...,
        relation: ndarray | None = ...,
        size: Size = ...,
        *,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...
    ) -> ndarray: ...

def _raises_not_implemented(*args: Any, **kwargs: Any) -> None: ...

beta = _raises_not_implemented
binomial = _raises_not_implemented
bytes = _raises_not_implemented
chisquare = _raises_not_implemented
choice = _raises_not_implemented
complex_normal = _raises_not_implemented
dirichlet = _raises_not_implemented
exponential = _raises_not_implemented
f = _raises_not_implemented
gamma = _raises_not_implemented
geometric = _raises_not_implemented
gumbel = _raises_not_implemented
hypergeometric = _raises_not_implemented
integers = _raises_not_implemented
laplace = _raises_not_implemented
logistic = _raises_not_implemented
lognormal = _raises_not_implemented
logseries = _raises_not_implemented
multinomial = _raises_not_implemented
multivariate_normal = _raises_not_implemented
negative_binomial = _raises_not_implemented
noncentral_chisquare = _raises_not_implemented
noncentral_f = _raises_not_implemented
normal = _raises_not_implemented
pareto = _raises_not_implemented
permutation = _raises_not_implemented
poisson = _raises_not_implemented
power = _raises_not_implemented
rand = _raises_not_implemented
randint = _raises_not_implemented
randn = _raises_not_implemented
random_integers = _raises_not_implemented
random_sample = _raises_not_implemented
random = _raises_not_implemented
rayleigh = _raises_not_implemented
shuffle = _raises_not_implemented
standard_cauchy = _raises_not_implemented
standard_exponential = _raises_not_implemented
standard_gamma = _raises_not_implemented
standard_normal = _raises_not_implemented
standard_t = _raises_not_implemented
tomaxint = _raises_not_implemented
triangular = _raises_not_implemented
uniform = _raises_not_implemented
vonmises = _raises_not_implemented
wald = _raises_not_implemented
weibull = _raises_not_implemented
zipf = _raises_not_implemented
