try:
    from numpy.random._bit_generator import (
        ISeedSequence,
        ISpawnableSeedSequence,
        SeedlessSeedSequence,
        SeedSequence,
    )
except (ImportError, AttributeError):
    try:
        from numpy.random.bit_generator import (
            ISeedSequence,
            ISpawnableSeedSequence,
            SeedlessSeedSequence,
            SeedSequence,
        )
    except (ImportError, AttributeError):
        from randomgen._seed_sequence import (
            ISeedSequence,
            ISpawnableSeedSequence,
            SeedlessSeedSequence,
            SeedSequence,
        )

__all__ = [
    "SeedSequence",
    "SeedlessSeedSequence",
    "ISeedSequence",
    "ISpawnableSeedSequence",
]
