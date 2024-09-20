from collections.abc import Sequence

from numpy import ndarray

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class ChaCha(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        counter: int | Sequence[int] | None = ...,
        key: int | Sequence[int] | None = ...,
        rounds: int = ...,
        mode: SeedMode | None = ...
    ) -> None: ...
    @property
    def use_simd(self) -> bool: ...
    @use_simd.setter
    def use_simd(self, value: bool) -> None: ...
    def seed(
        self,
        seed: IntegerSequenceSeed | None = ...,
        counter: int | Sequence[int] | None = ...,
        key: int | Sequence[int] | None = ...,
    ) -> None: ...
    @property
    def state(self) -> dict[str, str | dict[str, ndarray | int]]: ...
    @state.setter
    def state(self, value: dict[str, str | dict[str, ndarray | int]]) -> None: ...
