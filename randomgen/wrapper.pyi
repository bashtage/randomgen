from collections.abc import Callable
from ctypes import c_void_p
from typing import Any, Literal

from numba.core.ccallback import CFunc

from randomgen.common import BitGenerator

class UserBitGenerator(BitGenerator):
    def __init__(
        self,
        next_raw: Callable[[int], int] | None,
        bits: Literal[32, 64] = ...,
        next_64: Callable[[int], int] | None = ...,
        next_32: Callable[[int], int] | None = ...,
        next_double: Callable[[int], float] | None = ...,
        state: int | None = ...,
        state_getter: Callable[[], Any] | None = ...,
        state_setter: Callable[[Any], None] | None = ...,
    ) -> None: ...
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
        state_getter: Callable[[], Any] | None = ...,
        state_setter: Callable[[Any], None] | None = ...,
    ) -> UserBitGenerator: ...
    @classmethod
    def from_ctypes(
        cls,
        next_raw: Any,
        next_64: Any,
        next_32: Any,
        next_double: Any,
        state: c_void_p,
        state_getter: Callable[[], Any] | None = ...,
        state_setter: Callable[[Any], None] | None = ...,
    ) -> UserBitGenerator: ...
