from typing import Literal

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class MT19937(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        mode: Literal["numpy", "sequence"] | None = ...
    ) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    def jump(self, jumps: int = ...) -> MT19937: ...
    def jumped(self, jumps: int = ...) -> MT19937: ...
    @property
    def state(self) -> dict[str, str | dict[str, int | np.ndarray]]: ...
    @state.setter
    def state(
        self,
        value: (
            tuple[str, np.ndarray, int] | dict[str, str | dict[str, int | np.ndarray]]
        ),
    ) -> None: ...
    def _jump_tester(self) -> MT19937: ...
