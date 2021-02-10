import numpy as np

DEFAULT_MULTIPLIER: int
DEFAULT_DXSM_MULTIPLIER: int
from typing import Dict, Optional, Union

from randomgen.common import BitGenerator
from randomgen.typing import IntegerSequenceSeed, Literal, SeedMode

class PCG64(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        inc: Optional[int] = -999999,
        *,
        variant: Literal[
            "xsl-rr", "1.0", 1, "dxsm", "cm-dxsm", 2, "2.0", "dxsm-128"
        ] = "xsl-rr",
        mode: Optional[SeedMode] = None
    ) -> None: ...
    def seed(
        self, seed: Optional[IntegerSequenceSeed] = None, inc: Optional[int] = -999999
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, int, Dict[str, int]]]) -> None: ...
    def advance(self, delta: int) -> PCG64: ...
    def jump(self, iter: int = 1) -> PCG64: ...
    def jumped(self, iter: int = 1) -> PCG64: ...

class LCG128Mix(BitGenerator):
    def __init__(
        self,
        seed: Optional[IntegerSequenceSeed] = None,
        inc: Optional[int] = None,
        *,
        multiplier: int = 47026247687942121848144207491837523525,
        output: Union[str, int] = "xsl-rr",
        dxsm_multiplier: int = 15750249268501108917,
        post: bool = True
    ) -> None: ...
    def seed(
        self, seed: Optional[IntegerSequenceSeed] = None, inc: Optional[int] = None
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, Union[bool, int, str]]]]: ...
    @state.setter
    def state(
        self, value: Dict[str, Union[str, int, Dict[str, Union[bool, int, str]]]]
    ) -> None: ...
    def advance(self, delta: int) -> LCG128Mix: ...
    def jumped(self, iter: int = 1) -> LCG128Mix: ...

class PCG64DXSM(PCG64):
    def __init__(
        self, seed: Optional[IntegerSequenceSeed] = None, inc: Optional[int] = None
    ): ...
    @property
    def state(self) -> Dict[str, Union[str, int, Dict[str, int]]]: ...
    @state.setter
    def state(self, value: Dict[str, Union[str, int, Dict[str, int]]]) -> None: ...
    def jumped(self, iter: int = 1) -> PCG64DXSM: ...
    def jump(self, iter: int = 1) -> PCG64DXSM: ...
