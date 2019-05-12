from randomgen.mtrand import RandomState
from randomgen.pcg32 import PCG32
from randomgen.pcg64 import PCG64
from randomgen.philox import Philox
from randomgen.threefry import ThreeFry
from randomgen.threefry32 import ThreeFry32
from randomgen.xoroshiro128 import Xoroshiro128
from randomgen.xorshift1024 import Xorshift1024
from randomgen.xoshiro256starstar import Xoshiro256StarStar
from randomgen.xoshiro512starstar import Xoshiro512StarStar

from .dsfmt import DSFMT
from .generator import Generator
from .mt19937 import MT19937

BitGeneratorS = {'MT19937': MT19937,
             'DSFMT': DSFMT,
             'PCG32': PCG32,
             'PCG64': PCG64,
             'Philox': Philox,
             'ThreeFry': ThreeFry,
             'ThreeFry32': ThreeFry32,
             'Xorshift1024': Xorshift1024,
             'Xoroshiro128': Xoroshiro128,
             'Xoshiro256StarStar': Xoshiro256StarStar,
             'Xoshiro512StarStar': Xoshiro512StarStar,
             }


def __generator_ctor(bit_generator_name='mt19937'):
    """
    Pickling helper function that returns a Generator object

    Parameters
    ----------
    bit_generator_name: str
        String containing the core BitGenerator

    Returns
    -------
    rg: Generator
        Generator using the named core BitGenerator
    """
    try:
        bit_generator_name = bit_generator_name.decode('ascii')
    except AttributeError:
        pass
    if bit_generator_name in BitGeneratorS:
        bit_generator = BitGeneratorS[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')

    return Generator(bit_generator())


def __bit_generator_ctor(bit_generator_name='mt19937'):
    """
    Pickling helper function that returns a basic RNG object

    Parameters
    ----------
    bit_generator_name: str
        String containing the name of the Basic RNG

    Returns
    -------
    bit_generator: BitGenerator
        Bit generator instance
    """
    try:
        bit_generator_name = bit_generator_name.decode('ascii')
    except AttributeError:
        pass
    if bit_generator_name in BitGeneratorS:
        bit_generator = BitGeneratorS[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')

    return bit_generator()


def __randomstate_ctor(bit_generator_name='mt19937'):
    """
    Pickling helper function that returns a legacy RandomState-like object

    Parameters
    ----------
    bit_generator_name: str
        String containing the core BitGenerator

    Returns
    -------
    rs: RandomState
        Legacy RandomState using the named core BitGenerator
    """
    try:
        bit_generator_name = bit_generator_name.decode('ascii')
    except AttributeError:
        pass
    if bit_generator_name in BitGeneratorS:
        bit_generator = BitGeneratorS[bit_generator_name]
    else:
        raise ValueError(str(bit_generator_name) + ' is not a known BitGenerator module.')

    return RandomState(bit_generator())
