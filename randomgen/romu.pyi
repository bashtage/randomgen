from typing import Literal

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class Romu(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        variant: Literal["trio", "quad"] = ...,
    ) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    @property
    def state(self) -> dict[str, str | int | dict[str, int]]: ...
    @state.setter
    def state(self, value: dict[str, str | int | dict[str, int]]) -> None: ...
