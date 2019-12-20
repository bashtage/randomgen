try:
    from numpy.random._bit_generator import (SeedSequence,
                                             SeedlessSeedSequence,
                                             ISeedSequence,
                                             ISpawnableSeedSequence)
except (ImportError, AttributeError):
    try:
        from numpy.random.bit_generator import (SeedSequence,
                                                SeedlessSeedSequence,
                                                ISeedSequence,
                                                ISpawnableSeedSequence)
    except (ImportError, AttributeError):
        from randomgen._seed_sequence import (SeedSequence,
                                              SeedlessSeedSequence,
                                              ISeedSequence,
                                              ISpawnableSeedSequence)

__all__ = ["SeedSequence", "SeedlessSeedSequence", "ISeedSequence",
           "ISpawnableSeedSequence"]
