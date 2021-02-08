from typing import Literal, Sequence, Union

from randomgen.seed_sequence import SeedSequence

__all__ = ["IntegerSequenceSeed", "SeedMode"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

SeedMode = Literal["sequence", "legacy"]
