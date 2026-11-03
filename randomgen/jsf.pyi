from typing import ClassVar

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class JSF(BitGenerator):
    parameters: ClassVar[dict[int, list[dict[str, int]]]]
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        seed_size: int = ...,
        size: int = ...,
        p: int | None = ...,
        q: int | None = ...,
        r: int | None = ...,
    ) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    @property
    def state(self) -> dict[str, str | int | dict[str, int]]: ...
    @state.setter
    def state(self, value: dict[str, str | int | dict[str, int]]) -> None: ...
