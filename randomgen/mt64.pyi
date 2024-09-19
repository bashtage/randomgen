import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class MT64(BitGenerator):
    def __init__(
        self, seed: IntegerSequenceSeed | None = ..., *, mode: SeedMode | None = ...
    ) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    @property
    def state(
        self,
    ) -> dict[str, str | int | dict[str, int | np.ndarray]]: ...
    @state.setter
    def state(
        self, value: dict[str, str | int | dict[str, int | np.ndarray]]
    ) -> None: ...
