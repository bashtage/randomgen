import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class EFIIX64(BitGenerator):
    def __init__(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    @property
    def state(
        self,
    ) -> dict[str, str | int | dict[str, int | np.ndarray]]: ...
    @state.setter
    def state(
        self, value: dict[str, str | int | dict[str, int | np.ndarray]]
    ) -> None: ...
