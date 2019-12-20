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
        from randomgen._seed_sequence import (SeedSequence,  # noqa: F401
                                              SeedlessSeedSequence,
                                              ISeedSequence,
                                              ISpawnableSeedSequence)
