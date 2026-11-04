from typing import Literal

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

def generate_keys(
    seed: IntegerSequenceSeed | None = ...,
    n: int = ...,
    unique: bool = ...,
) -> np.ndarray: ...

class Squares(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        counter: int | None = ...,
        key: int | None = ...,
        variant: Literal[32, 64] = ...,
    ) -> None: ...
    def seed(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        counter: int | None = ...,
        key: int | None = ...,
    ) -> None: ...
    @property
    def state(self) -> dict[str, str | int | dict[str, int]]: ...
    @state.setter
    def state(self, value: dict[str, str | int | dict[str, int]]) -> None: ...
    def advance(self, delta: int) -> Squares: ...
    def jumped(self, iter: int = ...) -> Squares: ...
