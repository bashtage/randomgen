from typing import Optional
import warnings

from randomgen.typing import IntegerSequenceSeed, SeedMode
from randomgen.xoshiro256 import Xoshiro256


def Xoshiro256StarStar(
    seed: Optional[IntegerSequenceSeed] = None, *, mode: SeedMode = None
) -> Xoshiro256:
    """
    This is a deprecation shim.  Use Xoshiro256
    """
    warnings.warn("Xoshiro256StarStar has been renamed Xoshiro256", DeprecationWarning)
    return Xoshiro256(seed, mode=mode)
