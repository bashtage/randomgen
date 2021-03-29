from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

if TYPE_CHECKING:
    from typing import Literal

    SeedMode = Literal["sequence", "legacy"]
else:
    SeedMode = Any

from randomgen.seed_sequence import SeedSequence

__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

RequiredSize = Union[int, Sequence[int]]
Size = Optional[RequiredSize]
