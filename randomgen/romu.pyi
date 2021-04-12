from typing import Dict, Literal, Optional, Union

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed

class Romu(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = ...,
        variant: Literal["trio", "quad"] = ...,
    ) -> None: ...
    def seed(self, seed: Optional[IntegerSequenceSeed] = ...) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, int, Dict[str, int]]]) -> None: ...
