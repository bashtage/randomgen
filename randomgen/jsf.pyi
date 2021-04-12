from typing import Dict, Optional, Union

import numpy as np

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, SeedMode

class JSF(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        *,
        seed_size: int = ...,
        size: int = ...,
        p: Optional[int] = ...,
        q: Optional[int] = ...,
        r: Optional[int] = ...,
        mode: Optional[SeedMode] = ...
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = ...) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, int, Dict[str, int]]]) -> None: ...
