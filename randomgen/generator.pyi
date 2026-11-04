from typing import Any, Literal, overload

from numpy import ndarray
from numpy.typing import DTypeLike

from randomgen.common import BitGenerator
from randomgen.typing import RequiredSize, Size

class ExtendedGenerator:
    _bit_generator: BitGenerator
    def __init__(self, bit_generator: BitGenerator | None = ...) -> None: ...
    @property
    def bit_generator(self) -> BitGenerator: ...
    @property
    def state(self) -> dict[str, Any]: ...
    @state.setter
    def state(self, value: dict[str, Any]) -> None: ...
    @overload
    def uintegers(self, size: None = ..., bits: Literal[32, 64] = ...) -> int: ...
    @overload
    def uintegers(self, size: RequiredSize, bits: Literal[32, 64] = ...) -> ndarray: ...
    @overload
    def random(
        self,
        size: None = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> float: ...
    @overload
    def random(
        self,
        size: RequiredSize,
        dtype: DTypeLike = ...,
        out: ndarray | None = ...,
    ) -> ndarray: ...
    @overload
    def random(
        self,
        size: None = ...,
        dtype: DTypeLike = ...,
        *,
        out: ndarray,
    ) -> ndarray: ...
    def multivariate_normal(
        self,
        mean: ndarray,
        cov: ndarray,
        size: Size = ...,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        *,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...,
    ) -> ndarray: ...
    @overload
    def complex_normal(
        self,
        loc: complex = ...,
        gamma: complex = ...,
        relation: complex = ...,
        size: None = ...,
    ) -> complex: ...
    @overload
    def complex_normal(
        self,
        loc: complex | ndarray = ...,
        gamma: complex | ndarray = ...,
        relation: complex | ndarray = ...,
        size: Size = ...,
    ) -> ndarray: ...
    def standard_wishart(
        self,
        df: int,
        dim: int,
        size: Size = ...,
        rescale: bool = ...,
    ) -> ndarray: ...
    def wishart(
        self,
        df: int | ndarray,
        scale: ndarray | None = ...,
        size: Size = ...,
        *,
        check_valid: Literal["raise", "ignore", "warn"] = ...,
        tol: float = ...,
        rank: int | None = ...,
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...,
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
        method: Literal["svd", "eigh", "cholesky", "factor"] = ...,
    ) -> ndarray: ...
