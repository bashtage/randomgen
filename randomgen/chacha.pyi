from typing import Dict, Optional, Sequence, Union

from numpy import ndarray

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class ChaCha(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        *,
        counter: Optional[Union[int, Sequence[int]]] = ...,
        key: Optional[Union[int, Sequence[int]]] = ...,
        rounds: int = ...,
        mode: Optional[SeedMode] = ...
    ) -> None: ...
    @property
    def use_simd(self) -> bool: ...
    @use_simd.setter
    def use_simd(self, value: bool) -> None: ...
    def seed(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        counter: Optional[Union[int, Sequence[int]]] = ...,
        key: Optional[Union[int, Sequence[int]]] = ...,
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, Dict[str, Union[ndarray, int]]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, Dict[str, Union[ndarray, int]]]]
    ) -> None: ...
    def jump(self, iter: int = ...) -> ChaCha: ...
    def jumped(self, iter: int = ...) -> ChaCha: ...
    def advance(self, delta: int) -> ChaCha: ...
