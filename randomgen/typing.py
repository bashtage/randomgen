from typing import Literal, Optional, Union
from collections.abc import Sequence

from randomgen.seed_sequence import SeedSequence

SeedMode = Literal["sequence", "legacy"]

__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

RequiredSize = Union[int, Sequence[int]]
Size = Optional[RequiredSize]
