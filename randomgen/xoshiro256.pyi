import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class Xoshiro256(BitGenerator):
    def __init__(
        self, seed: IntegerSequenceSeed | None = ..., *, mode: SeedMode = ...
    ) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    def jump(self, iter: int = ...) -> Xoshiro256: ...
    def jumped(self, iter: int = ...) -> Xoshiro256: ...
    @property
    def state(self) -> dict[str, str | np.ndarray | int]: ...
    @state.setter
    def state(self, value: dict[str, str | np.ndarray | int]) -> None: ...
