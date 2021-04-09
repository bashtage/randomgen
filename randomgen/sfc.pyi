from typing import Dict, Optional, Union, Literal

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class SFC64(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        w: int = 1,
        k: int = 1,
        *,
        mode: Optional[Literal["sequence", "numpy"]]
    ) -> None: ...
    def weyl_increments(
        self, n: int, max_bits: int = 32, min_bits: Optional[int] = None
    ) -> np.ndarray: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = None) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, int, Dict[str, int]]]) -> None: ...
