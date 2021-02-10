from ctypes import CFUNCTYPE, c_void_p
from typing import Any, Callable, Literal, Optional

from numba.core.ccallback import CFunc

from randomgen.common import BitGenerator

class UserBitGenerator(BitGenerator):
    def __init__(
        self,
        next_raw: Optional[Callable[[int], int]],
        bits: Literal[32, 64] = 64,
        next_64: Optional[Callable[[int], int]] = None,
        next_32: Optional[Callable[[int], int]] = None,
        next_double: Optional[Callable[[int], float]] = None,
        state: Optional[int] = None,
        state_getter: Optional[Callable[[], Any]] = None,
        state_setter: Optional[Callable[[Any], None]] = None,
    ): ...
    @property
    def state(self) -> Any: ...
    @state.setter
    def state(self, value: Any) -> None: ...
    @classmethod
    def from_cfunc(
        cls,
        next_raw: CFunc,
        next_64: CFunc,
        next_32: CFunc,
        next_double: CFunc,
        state: int,
        state_getter: Optional[Callable[[], Any]] = None,
        state_setter: Optional[Callable[[Any], None]] = None,
    ) -> UserBitGenerator: ...
    @classmethod
    def from_ctypes(
        cls,
        next_raw: Any,
        next_64: Any,
        next_32: Any,
        next_double: Any,
        state: c_void_p,
        state_getter: Optional[Callable[[], Any]] = None,
        state_setter: Optional[Callable[[Any], None]] = None,
    ) -> UserBitGenerator: ...
