from typing import Any, NamedTuple

from numpy import ndarray

from randomgen.seed_sequence import SeedSequence
from randomgen.typing import IntegerSequenceSeed

class Interface(NamedTuple):
    state_address: int
    state: int
    next_uint64: int
    next_uint32: int
    next_double: int
    bit_generator: int

class BitGenerator:
    seed_seq: SeedSequence
    def __init__(
        self,
        seed: IntegerSequenceSeed | None,
        *,
        numpy_seed: bool = ...,
    ) -> None: ...
    @property
    def state(self) -> dict[str, Any]: ...
    @state.setter
    def state(self, value: dict[str, Any]) -> None: ...
    def random_raw(
        self,
        size: int | None = ...,
        output: bool = ...,
    ) -> int | ndarray | None: ...
    def _benchmark(self, cnt: int, method: str = ...) -> None: ...
    @property
    def ctypes(self) -> Any: ...
    @property
    def cffi(self) -> Any: ...
