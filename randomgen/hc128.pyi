from collections.abc import Sequence

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class HC128(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        key: int | Sequence[int] | None = ...,
        mode: SeedMode = ...
    ) -> None: ...
    def seed(
        self,
        seed: IntegerSequenceSeed | None = ...,
        key: int | Sequence[int] | None = ...,
    ) -> None: ...
    @property
    def state(self) -> dict[str, str | dict[str, int | np.ndarray]]: ...
    @state.setter
    def state(self, value: dict[str, str | dict[str, int | np.ndarray]]) -> None: ...
