from typing import Dict, Optional, Sequence, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

DSFMTState = Dict[str, Union[str, int, np.ndarray, Dict[str, Union[int, np.ndarray]]]]

class DSFMT(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        *,
        mode: Optional[SeedMode] = None,
    ) -> None: ...
    def seed(self, seed: Union[int, Sequence[int]] = None) -> None: ...
    def jump(self, iter: int = 1) -> DSFMT: ...
    def jumped(self, iter: int = 1) -> DSFMT: ...
    @property
    def state(
        self,
    ) -> DSFMTState: ...
    @state.setter
    def state(
        self,
        value: DSFMTState,
    ) -> None: ...