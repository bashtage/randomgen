from typing import Dict, Literal, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class SFC64(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        w: int = ...,
        k: int = ...,
        *,
        mode: Optional[Literal["sequence", "numpy"]]
    ) -> None: ...
    def weyl_increments(
        self, n: int, max_bits: int = ..., min_bits: Optional[int] = ...
    ) -> np.ndarray: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = ...) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, int, Dict[str, int]]]) -> None: ...
