from typing import Dict, Optional, Tuple, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class MT19937(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        *,
        mode: Optional[SeedMode] = None
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = None) -> None: ...
    def jump(self, jumps: int = 1) -> MT19937: ...
    def jumped(self, jumps: int = 1) -> MT19937: ...
    @property
    def state(self) -> Dict[str, Union[str, Dict[str, Union[int, np.ndarray]]]]: ...
    @state.setter
    def state(
        self,
        value: Union[
            Tuple[str, np.ndarray, int],
            Dict[str, Union[str, Dict[str, Union[int, np.ndarray]]]],
        ],
    ) -> None: ...
