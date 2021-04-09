from typing import Any, NamedTuple, Optional, Union

from numpy import ndarray

from randomgen.seed_sequence import SeedSequence
from randomgen.typing import IntegerSequenceSeed, SeedMode

class interface(NamedTuple):
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
        seed: Union[IntegerSequenceSeed] = ...,
        mode: Optional[SeedMode] = ...,
    ): ...
    def random_raw(
        self, size: Optional[int] = ..., output: bool = ...
    ) -> Union[None, int, ndarray]: ...
    def _benchmark(self, cnt: int, method: str = "uint64") -> None: ...
    @property
    def ctypes(self) -> Any: ...
    @property
    def cffi(self) -> Any: ...
