from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class SFMT(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        *,
        mode: Optional[SeedMode] = None
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = None) -> None: ...
    def jump(self, iter: int = 1) -> SFMT: ...
    def jumped(self, iter: int = 1) -> SFMT: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, int, np.ndarray, Dict[str, Union[int, np.ndarray]]]]: ...
    @state.setter
    def state(
        self,
        value: Dict[
            str, Union[str, int, np.ndarray, Dict[str, Union[int, np.ndarray]]]
        ],
    ) -> None: ...
