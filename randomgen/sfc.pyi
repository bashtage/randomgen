from typing import Literal

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class SFC64(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        w: int = ...,
        k: int = ...,
        *,
        mode: Literal["sequence", "numpy"] | None
    ) -> None: ...
    def weyl_increments(
        self, n: int, max_bits: int = ..., min_bits: int | None = ...
    ) -> np.ndarray: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    @property
    def state(self) -> dict[str, str | int | dict[str, int]]: ...
    @state.setter
    def state(self, value: dict[str, str | int | dict[str, int]]) -> None: ...
