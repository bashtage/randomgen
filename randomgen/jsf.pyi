from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class JSF(BitGenerator):
    def __init__(
        self,
        seed: IntegerSequenceSeed | None = ...,
        *,
        seed_size: int = ...,
        size: int = ...,
        p: int | None = ...,
        q: int | None = ...,
        r: int | None = ...,
        mode: SeedMode | None = ...
    ) -> None: ...
    def seed(self, seed: IntegerSequenceSeed | None = ...) -> None: ...
    @property
    def state(self) -> dict[str, str | int | dict[str, int]]: ...
    @state.setter
    def state(self, value: dict[str, str | int | dict[str, int]]) -> None: ...
