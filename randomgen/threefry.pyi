from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class ThreeFry(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        *,
        counter: Optional[Union[int, np.ndarray]] = None,
        key: Optional[Union[int, np.ndarray]] = None,
        number: int = 4,
        width: int = 64,
        mode: Optional[SeedMode] = None
    ) -> None: ...
    def seed(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        counter: Optional[Union[int, np.ndarray]] = None,
        key: Optional[Union[int, np.ndarray]] = None,
    ) -> None: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, int, np.ndarray, Dict[str, np.ndarray]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, int, np.ndarray, Dict[str, np.ndarray]]]
    ) -> None: ...
    def jump(self, iter: int = 1) -> ThreeFry: ...
    def jumped(self, iter: int = 1) -> ThreeFry: ...
    def advance(self, delta: int, counter: Optional[bool] = None) -> ThreeFry: ...
