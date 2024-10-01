from collections.abc import Sequence
from typing import Literal

import numpy as np

def view_little_endian_shim(arr: np.ndarray, dtype: str) -> np.ndarray: ...
def int_to_array_shim(
    value: int, name: str, bits: int, uint_size: Literal[32, 64]
) -> np.ndarray: ...
def byteswap_little_endian_shim(arr: np.ndarray) -> np.ndarray: ...
def object_to_int_shim(
    val: int | np.ndarray | Sequence[int],
    bits: int,
    name: str,
    default_bits: Literal[32, 64] = ...,
    allowed_sizes: tuple[int] | tuple[int, int] = ...,
) -> int: ...
