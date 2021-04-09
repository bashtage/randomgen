from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class ThreeFry(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        *,
        counter: Optional[Union[int, np.ndarray]] = ...,
        key: Optional[Union[int, np.ndarray]] = ...,
        number: int = ...,
        width: int = ...,
        mode: Optional[SeedMode] = ...
    ) -> None: ...
    def seed(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        counter: Optional[Union[int, np.ndarray]] = ...,
        key: Optional[Union[int, np.ndarray]] = ...,
    ) -> None: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, int, np.ndarray, Dict[str, np.ndarray]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, int, np.ndarray, Dict[str, np.ndarray]]]
    ) -> None: ...
    def jump(self, iter: int = ...) -> ThreeFry: ...
    def jumped(self, iter: int = ...) -> ThreeFry: ...
    def advance(self, delta: int, counter: Optional[bool] = ...) -> ThreeFry: ...
