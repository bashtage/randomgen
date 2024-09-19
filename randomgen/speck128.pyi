import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

SPECK_MAX_ROUNDS: int

class SPECK128(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        counter: int | np.ndarray | None = ...,
        key: int | np.ndarray | None = ...,
        rounds: int = ...,
        mode: SeedMode | None = ...
    ) -> None: ...
    def seed(
        self,
        seed: IntegerSequenceSeed | None = ...,
        counter: int | np.ndarray | None = ...,
        key: int | np.ndarray | None = ...,
    ) -> None: ...
    @property
    def use_sse41(self) -> bool: ...
    @use_sse41.setter
    def use_sse41(self, value: bool) -> None: ...
    @property
    def state(
        self,
    ) -> dict[str, str | int | dict[str, int | np.ndarray]]: ...
    @state.setter
    def state(
        self, value: dict[str, str | int | dict[str, int | np.ndarray]]
    ) -> None: ...
    def jump(self, iter: int = ...) -> SPECK128: ...
    def jumped(self, iter: int = ...) -> SPECK128: ...
    def advance(self, delta: int) -> SPECK128: ...
