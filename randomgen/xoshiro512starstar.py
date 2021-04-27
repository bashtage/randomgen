from __future__ import annotations

from typing import Optional
import warnings

from randomgen.typing import IntegerSequenceSeed, SeedMode
from randomgen.xoshiro512 import Xoshiro512


def Xoshiro512StarStar(
    seed: Optional[IntegerSequenceSeed] = None, *, mode: SeedMode = "sequence"
) -> Xoshiro512:
    """
    This is a deprecation shim.  Use Xoshiro512
    """
    warnings.warn("Xoshiro512StarStar has been renamed Xoshiro512", DeprecationWarning)
    return Xoshiro512(seed, mode=mode)
