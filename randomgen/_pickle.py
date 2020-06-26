import warnings

from randomgen.aes import AESCounter
from randomgen.chacha import ChaCha
from randomgen.dsfmt import DSFMT
from randomgen.efiix64 import EFIIX64
from randomgen.generator import ExtendedGenerator, Generator
from randomgen.hc128 import HC128
from randomgen.jsf import JSF
from randomgen.lxm import LXM
from randomgen.mt64 import MT64
from randomgen.mt19937 import MT19937
from randomgen.mtrand import RandomState
from randomgen.pcg32 import PCG32
from randomgen.pcg64 import PCG64, PCG64DXSM, LCG128Mix
from randomgen.philox import Philox
from randomgen.rdrand import RDRAND
from randomgen.romu import Romu
from randomgen.sfc import SFC64
from randomgen.sfmt import SFMT
from randomgen.speck128 import SPECK128
from randomgen.threefry import ThreeFry
from randomgen.xoroshiro128 import Xoroshiro128
from randomgen.xorshift1024 import Xorshift1024
from randomgen.xoshiro256 import Xoshiro256
from randomgen.xoshiro512 import Xoshiro512

BitGenerators = {
    "AESCounter": AESCounter,
    "ChaCha": ChaCha,
    "LCG128Mix": LCG128Mix,
    "DSFMT": DSFMT,
    "EFIIX64": EFIIX64,
    "HC128": HC128,
    "JSF": JSF,
    "LXM": LXM,
    "MT19937": MT19937,
    "MT64": MT64,
    "PCG32": PCG32,
    "PCG64": PCG64,
    "PCG64DXSM": PCG64DXSM,
    "Philox": Philox,
    "Romu": Romu,
    "ThreeFry": ThreeFry,
    "Xorshift1024": Xorshift1024,
    "Xoroshiro128": Xoroshiro128,
    "Xoshiro256": Xoshiro256,
    "Xoshiro512": Xoshiro512,
    "SPECK128": SPECK128,
    "SFC64": SFC64,
    "SFMT": SFMT,
    "RDRAND": RDRAND,
}


def _get_bitgenerator(bit_generator_name):
    """
    Bit generator look-up with user-friendly errors
    """
    if bit_generator_name in BitGenerators:
        bit_generator = BitGenerators[bit_generator_name]
    else:
        raise ValueError(
            str(bit_generator_name) + " is not a known BitGenerator module."
        )
    return bit_generator


def __generator_ctor(bit_generator_name="MT19937"):
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
        bit_generator_name = bit_generator_name.decode("ascii")
    except AttributeError:
        pass
    bit_generator = _get_bitgenerator(bit_generator_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bit_gen = bit_generator()
    return Generator(bit_gen)


def __extended_generator_ctor(bit_generator_name="MT19937"):
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
        bit_generator_name = bit_generator_name.decode("ascii")
    except AttributeError:
        pass
    bit_generator = _get_bitgenerator(bit_generator_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bit_gen = bit_generator()
    return ExtendedGenerator(bit_gen)


def __bit_generator_ctor(bit_generator_name="MT19937"):
    """
    Pickling helper function that returns a bit generator object

    Parameters
    ----------
    bit_generator_name: str
        String containing the name of the bit generator

    Returns
    -------
    bit_generator: BitGenerator
        Bit generator instance
    """
    try:
        bit_generator_name = bit_generator_name.decode("ascii")
    except AttributeError:
        pass
    bit_generator = _get_bitgenerator(bit_generator_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bit_gen = bit_generator()
    return bit_gen


def __randomstate_ctor(bit_generator_name="MT19937"):
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
        bit_generator_name = bit_generator_name.decode("ascii")
    except AttributeError:
        pass
    bit_generator = _get_bitgenerator(bit_generator_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bit_gen = bit_generator()
    return RandomState(bit_gen)
