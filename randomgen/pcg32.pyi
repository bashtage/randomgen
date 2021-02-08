from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class PCG32(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        inc: int = 0,
        *,
        mode: Optional[SeedMode] = None
    ) -> None: ...
    def seed(
        self, seed: Optional[IntegerSequenceSeed] = None, inc: int = 0
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, Dict[str, int]]]) -> None: ...
    def advance(self, delta: int) -> None: ...
    def jump(self, iter: int = 1) -> PCG32: ...
    def jumped(self, iter: int = 1) -> PCG32: ...
