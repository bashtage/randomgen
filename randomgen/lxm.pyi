from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class LXM(BitGenerator):
    def __init__(
        self, seed: Optional[IntegerSequenceSeed] = None, *, b: int = 3037000493
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = None) -> None: ...
    def jump(self, iter: int = 1) -> LXM: ...
    def jumped(self, iter: int = 1) -> LXM: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, int, Dict[str, Union[int, np.ndarray]]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, int, Dict[str, Union[int, np.ndarray]]]]
    ) -> None: ...
