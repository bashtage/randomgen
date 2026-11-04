from collections.abc import Sequence

from randomgen.seed_sequence import SeedSequence

__all__ = ["IntegerSequenceSeed", "Size"]

IntegerSequenceSeed = int | Sequence[int] | SeedSequence

RequiredSize = int | Sequence[int]
Size = RequiredSize | None
