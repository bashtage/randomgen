from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class EFIIX64(BitGenerator):
    def __init__(self, seed: Optional[IntegerSequenceSeed] = ...) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = ...) -> None: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[str, int, Dict[str, Union[np.ndarray, int]]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, int, Dict[str, Union[np.ndarray, int]]]]
    ) -> None: ...
