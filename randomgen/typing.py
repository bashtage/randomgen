from collections.abc import Sequence
from typing import Optional, Union

from randomgen.seed_sequence import SeedSequence

__all__ = ["IntegerSequenceSeed", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

RequiredSize = int | Sequence[int]
Size = Optional[RequiredSize]
