import sys
from typing import TYPE_CHECKING, Optional, Sequence, Union

from randomgen.seed_sequence import SeedSequence

if sys.version_info >= (3, 8):
    from typing import Literal
elif TYPE_CHECKING:
    from typing_extensions import Literal
else:

    class _Literal:
        def __getitem__(self, item):
            pass

    Literal = _Literal()

SeedMode = Literal["sequence", "legacy"]


__all__ = ["IntegerSequenceSeed", "SeedMode", "Size"]

IntegerSequenceSeed = Union[int, Sequence[int], SeedSequence]

RequiredSize = Union[int, Sequence[int]]
Size = Optional[RequiredSize]
