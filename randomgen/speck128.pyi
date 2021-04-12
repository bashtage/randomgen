from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

SPECK_MAX_ROUNDS: int

class SPECK128(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        *,
        counter: Optional[Union[int, np.ndarray]] = ...,
        key: Optional[Union[int, np.ndarray]] = ...,
        rounds: int = ...,
        mode: Optional[SeedMode] = ...
    ) -> None: ...
    def seed(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        counter: Optional[Union[int, np.ndarray]] = ...,
        key: Optional[Union[int, np.ndarray]] = ...,
    ) -> None: ...
    @property
    def use_sse41(self) -> bool: ...
    @use_sse41.setter
    def use_sse41(self, value: bool) -> None: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, int, Dict[str, Union[int, np.ndarray]]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, int, Dict[str, Union[int, np.ndarray]]]]
    ) -> None: ...
    def jump(self, iter: int = ...) -> SPECK128: ...
    def jumped(self, iter: int = ...) -> SPECK128: ...
    def advance(self, delta: int) -> SPECK128: ...
