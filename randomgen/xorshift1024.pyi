import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class Xorshift1024(BitGenerator):
    def __init__(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    def jumped(self, iter: int = ...) -> Xorshift1024: ...
    @property
    def state(
        self,
    ) -> dict[str, str | dict[str, int | np.ndarray] | int]: ...
    @state.setter
    def state(
        self,
        value: dict[str, str | dict[str, int | np.ndarray] | int],
    ) -> None: ...
