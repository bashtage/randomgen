from typing import Literal

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class Philox(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        counter: int | np.ndarray | None = ...,
        key: int | np.ndarray | None = ...,
        number: int = ...,
        width: int = ...,
        mode: Literal["sequence", "numpy"] | None = ...
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
    def jump(self, iter: int = ...) -> Philox: ...
    def jumped(self, iter: int = ...) -> Philox: ...
    def advance(self, delta: int, counter: bool | None = ...) -> Philox: ...
