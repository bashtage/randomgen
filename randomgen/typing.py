from typing import TYPE_CHECKING, Any, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    from typing import Literal

    SeedMode = Literal["sequence", "legacy"]
else:
    SeedMode = Any

from randomgen.seed_sequence import SeedSequence

__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]


Size = Optional[Union[int, Tuple[int, ...]]]
