from typing import Dict, Optional, Sequence, Union

from numpy import ndarray
from numpy.random import SeedSequence

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class AESCounter(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        *,
        counter: Optional[Union[int, Sequence[int]]] = ...,
        key: Optional[Union[int, Sequence[int]]] = ...,
        mode: Optional[SeedMode] = ...
    ) -> None: ...
    @property
    def use_aesni(self) -> bool: ...
    @use_aesni.setter
    def use_aesni(self, value: bool) -> None: ...
    def seed(
        self,
        seed: Union[int, SeedSequence] = ...,
        counter: Optional[Union[int, Sequence[int]]] = ...,
        key: Optional[Union[int, Sequence[int]]] = ...,
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, Dict[str, Union[int, ndarray]], int]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, Dict[str, Union[int, ndarray]], int]]
    ) -> None: ...
    def jump(self, iter: int = ...) -> AESCounter: ...
    def jumped(self, iter: int = ...) -> AESCounter: ...
    def advance(self, delta: int) -> AESCounter: ...
