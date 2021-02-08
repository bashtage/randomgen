from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class Xorshift1024(BitGenerator):
    def __init__(
        self, seed: Optional[IntegerSequenceSeed] = None, *, mode: SeedMode = None
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = None) -> None: ...
    def jump(self, iter: int = 1) -> Xorshift1024: ...
    def jumped(self, iter: int = 1) -> Xorshift1024: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, Dict[str, Union[np.ndarray, int]], int]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, Dict[str, Union[np.ndarray, int]], int]]
    ) -> None: ...
