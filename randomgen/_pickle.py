from __future__ import annotations

import warnings

from randomgen.aes import AESCounter
from randomgen.chacha import ChaCha
from randomgen.common import BitGenerator
from randomgen.dsfmt import DSFMT
from randomgen.efiix64 import EFIIX64
from randomgen.generator import ExtendedGenerator
from randomgen.hc128 import HC128
from randomgen.jsf import JSF
from randomgen.lxm import LXM
from randomgen.mt64 import MT64
from randomgen.mt19937 import MT19937
from randomgen.pcg32 import PCG32
from randomgen.pcg64 import PCG64, PCG64DXSM, LCG128Mix
from randomgen.philox import Philox
from randomgen.rdrand import RDRAND
from randomgen.romu import Romu
from randomgen.sfc import SFC64
from randomgen.sfmt import SFMT
from randomgen.speck128 import SPECK128
from randomgen.squares import Squares
from randomgen.threefry import ThreeFry
from randomgen.tyche import Tyche
from randomgen.xoroshiro128 import Xoroshiro128
from randomgen.xorshift1024 import Xorshift1024
from randomgen.xoshiro256 import Xoshiro256
from randomgen.xoshiro512 import Xoshiro512

BitGenerators: dict[str, type[BitGenerator]] = {
    "AESCounter": AESCounter,
    "ChaCha": ChaCha,
    "DSFMT": DSFMT,
    "EFIIX64": EFIIX64,
    "HC128": HC128,
    "JSF": JSF,
    "LCG128Mix": LCG128Mix,
    "LXM": LXM,
    "MT19937": MT19937,
    "MT64": MT64,
    "PCG32": PCG32,
    "PCG64": PCG64,
    "PCG64DXSM": PCG64DXSM,
    "Philox": Philox,
    "RDRAND": RDRAND,
    "Romu": Romu,
    "SFC64": SFC64,
    "SFMT": SFMT,
    "SPECK128": SPECK128,
    "Squares": Squares,
    "ThreeFry": ThreeFry,
    "Tyche": Tyche,
    "Xoroshiro128": Xoroshiro128,
    "Xorshift1024": Xorshift1024,
    "Xoshiro256": Xoshiro256,
    "Xoshiro512": Xoshiro512,
}

# Assign the fully qualified name for future proofness
for value in list(BitGenerators.values()):
    BitGenerators[f"{value.__module__}.{value.__name__}"] = value


def _get_bitgenerator(bit_generator_name: str) -> type[BitGenerator]:
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


def _decode(name: str | bytes) -> str:
    if isinstance(name, str):
        return name
    assert isinstance(name, bytes)
    return name.decode("ascii")


def __extended_generator_ctor(
    bit_generator_name: str | bytes = "MT19937",
) -> ExtendedGenerator:
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
    bit_generator_name = _decode(bit_generator_name)
    assert isinstance(bit_generator_name, str)
    bit_generator = _get_bitgenerator(bit_generator_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bit_gen = bit_generator()
    return ExtendedGenerator(bit_gen)


def __bit_generator_ctor(bit_generator_name: str | bytes = "MT19937") -> BitGenerator:
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
    bit_generator_name = _decode(bit_generator_name)
    assert isinstance(bit_generator_name, str)
    bit_generator = _get_bitgenerator(bit_generator_name)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        bit_gen = bit_generator()
    return bit_gen
