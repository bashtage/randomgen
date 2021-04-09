from typing import Dict, Optional, Tuple, Union, Literal

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class MT19937(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        *,
        mode: Optional[Literal["numpy", "sequence", "legacy"]] = ...
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = ...) -> None: ...
    def jump(self, jumps: int = ...) -> MT19937: ...
    def jumped(self, jumps: int = ...) -> MT19937: ...
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
