from collections.abc import Sequence
from typing import Literal, Optional, Union

from randomgen.seed_sequence import SeedSequence

SeedMode = Literal["sequence"]

__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

RequiredSize = int | Sequence[int]
Size = Optional[RequiredSize]
