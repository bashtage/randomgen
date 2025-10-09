from numpy.random._pickle import BitGenerators

from randomgen.aes import AESCounter
from randomgen.chacha import ChaCha
from randomgen.dsfmt import DSFMT
from randomgen.efiix64 import EFIIX64
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
from randomgen.wrapper import UserBitGenerator
from randomgen.xoroshiro128 import Xoroshiro128
from randomgen.xorshift1024 import Xorshift1024
from randomgen.xoshiro256 import Xoshiro256
from randomgen.xoshiro512 import Xoshiro512

bit_generators = [
    AESCounter,
    ChaCha,
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    LXM,
    MT19937,
    MT64,
    PCG32,
    PCG64,
    PCG64DXSM,
    LCG128Mix,
    Philox,
    RDRAND,
    Romu,
    SFC64,
    SFMT,
    SPECK128,
    Squares,
    ThreeFry,
    Tyche,
    UserBitGenerator,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
]

for bitgen in bit_generators:
    key = f"{bitgen.__name__}"
    if key not in BitGenerators:
        BitGenerators[key] = bitgen
    full_key = f"{bitgen.__module__}.{bitgen.__name__}"
    BitGenerators[full_key] = bitgen

__all__ = ["BitGenerators"]
