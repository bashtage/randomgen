from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Sequence, Type, Union

from numpy import uint32, uint64, unsignedinteger

DEFAULT_POOL_SIZE: int

class ISeedSequence(metaclass=ABCMeta):
    @abstractmethod
    def generate_state(
        self, n_words: int, dtype: Type[unsignedinteger[Any]] = uint32
    ) -> Sequence[int]: ...

class ISpawnableSeedSequence(ISeedSequence):
    @abstractmethod
    def spawn(self, n_children: int) -> List["SeedSequence"]: ...

class SeedSequence(ISpawnableSeedSequence):
    def __init__(
        self,
        entropy: Optional[Union[int, Sequence[int]]] = None,
        *,
        spawn_key: Sequence[int] = (),
        pool_size: int = DEFAULT_POOL_SIZE,
        n_children_spawned: int = 0
    ) -> None: ...
    @property
    def state(self) -> Dict[str, Union[int, Sequence[int]]]: ...
    def generate_state(
        self, n_words: int, dtype: Type[unsignedinteger[Any]] = uint32
    ) -> Sequence[int]: ...
    def spawn(self, n_children: int) -> List[SeedSequence]: ...

class SeedlessSeedSequence(ISeedSequence):
    def generate_state(
        self, n_words: int, dtype: Type[unsignedinteger[Any]] = uint32
    ) -> Sequence[int]: ...
    def spawn(self, n_children: int) -> List[SeedlessSeedSequence]: ...
