from abc import ABCMeta, abstractmethod
from collections.abc import Sequence
from typing import Any

from numpy import ndarray, unsignedinteger

class ISeedSequence(metaclass=ABCMeta):
    @abstractmethod
    def generate_state(
        self,
        n_words: int,
        dtype: type[unsignedinteger[Any]] = ...,
    ) -> Sequence[int]: ...

class ISpawnableSeedSequence(ISeedSequence, metaclass=ABCMeta):
    @abstractmethod
    def spawn(self, n_children: int) -> list[SeedSequence]: ...

class SeedSequence(ISpawnableSeedSequence):
    @property
    def entropy(self) -> int | Sequence[int] | None: ...
    @property
    def spawn_key(self) -> tuple[int, ...]: ...
    @property
    def pool_size(self) -> int: ...
    @property
    def pool(self) -> ndarray: ...
    @property
    def n_children_spawned(self) -> int: ...
    def __init__(
        self,
        entropy: int | Sequence[int] | None = ...,
        *,
        spawn_key: Sequence[int] = ...,
        pool_size: int = ...,
        n_children_spawned: int = ...,
    ) -> None: ...
    @property
    def state(self) -> dict[str, int | Sequence[int]]: ...
    def generate_state(
        self,
        n_words: int,
        dtype: type[unsignedinteger[Any]] = ...,
    ) -> Sequence[int]: ...
    def spawn(self, n_children: int) -> list[SeedSequence]: ...

class SeedlessSeedSequence(ISeedSequence):
    def generate_state(
        self,
        n_words: int,
        dtype: type[unsignedinteger[Any]] = ...,
    ) -> Sequence[int]: ...
    def spawn(self, n_children: int) -> list[SeedlessSeedSequence]: ...
