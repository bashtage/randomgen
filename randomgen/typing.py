from typing import Literal, Optional, Sequence, Union

from randomgen.seed_sequence import SeedSequence

SeedMode = Literal["sequence", "legacy"]

__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

RequiredSize = Union[int, Sequence[int]]
Size = Optional[RequiredSize]
