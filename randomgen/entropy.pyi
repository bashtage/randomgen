from typing import Literal

import numpy as np

def seed_by_array(seed: int | np.ndarray, n: int) -> np.ndarray: ...
def random_entropy(
    size: int | tuple[int, ...] = ...,
    source: Literal["system", "fallback", "auto"] = ...,
) -> int | np.ndarray: ...
