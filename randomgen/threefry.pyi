import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class ThreeFry(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        counter: int | np.ndarray | None = ...,
        key: int | np.ndarray | None = ...,
        number: int = ...,
        width: int = ...,
        mode: SeedMode | None = ...
    ) -> None: ...
    def seed(
        self,
        seed: IntegerSequenceSeed | None = ...,
        counter: int | np.ndarray | None = ...,
        key: int | np.ndarray | None = ...,
    ) -> None: ...
    @property
    def state(
        self,
    ) -> dict[str, str | int | np.ndarray | dict[str, np.ndarray]]: ...
    @state.setter
    def state(
        self, value: dict[str, str | int | np.ndarray | dict[str, np.ndarray]]
    ) -> None: ...
    def jump(self, iter: int = ...) -> ThreeFry: ...
    def jumped(self, iter: int = ...) -> ThreeFry: ...
    def advance(self, delta: int, counter: bool | None = ...) -> ThreeFry: ...
