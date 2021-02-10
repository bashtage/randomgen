from typing import Literal, Optional, Sequence, Tuple, Union

from randomgen.seed_sequence import SeedSequence

__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

SeedMode = Literal["sequence", "legacy"]
Size = Optional[Union[int, Tuple[int, ...]]]
