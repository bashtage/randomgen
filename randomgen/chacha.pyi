from typing import Dict, Optional, Sequence, Union

from numpy import ndarray

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class ChaCha(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        *,
        counter: Optional[Union[int, Sequence[int]]] = None,
        key: Optional[Union[int, Sequence[int]]] = None,
        rounds: int = 20,
        mode: Optional[SeedMode] = None,
    ) -> None: ...
    @property
    def use_simd(self) -> bool: ...
    @use_simd.setter
    def use_simd(self, value: bool) -> None: ...
    def seed(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        counter: Optional[Union[int, Sequence[int]]] = None,
        key: Optional[Union[int, Sequence[int]]] = None,
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, Dict[str, Union[ndarray, int]]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, Dict[str, Union[ndarray, int]]]]
    ) -> None: ...
    def jump(self, iter: int = 1) -> ChaCha: ...
    def jumped(self, iter: int = 1) -> ChaCha: ...
    def advance(self, delta: int) -> ChaCha: ...
