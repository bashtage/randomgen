from __future__ import annotations

import os
import sys

from randomgen._register import BitGenerators
from randomgen.aes import AESCounter
from randomgen.blabla import BlaBla
from randomgen.chacha import ChaCha
from randomgen.dsfmt import DSFMT
from randomgen.efiix64 import EFIIX64
from randomgen.entropy import random_entropy
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
from randomgen.seed_sequence import SeedlessSeedSequence, SeedSequence
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

from ._version import version as __version__, version_tuple as __version_info__

PKG = os.path.join(os.path.dirname(__file__))


__all__ = [
    "DSFMT",
    "EFIIX64",
    "HC128",
    "JSF",
    "LXM",
    "MT64",
    "MT19937",
    "PCG32",
    "PCG64",
    "PCG64DXSM",
    "RDRAND",
    "SFC64",
    "SFMT",
    "SPECK128",
    "AESCounter",
    "BitGenerators",
    "BlaBla",
    "ChaCha",
    "ExtendedGenerator",
    "LCG128Mix",
    "Philox",
    "Romu",
    "SeedSequence",
    "SeedlessSeedSequence",
    "Squares",
    "ThreeFry",
    "Tyche",
    "UserBitGenerator",
    "Xoroshiro128",
    "Xorshift1024",
    "Xoshiro256",
    "Xoshiro512",
    "__version__",
    "__version_info__",
    "random_entropy",
]


def test(extra_args: str | list[str] | None = None, exit=True) -> None:  # noqa: PT028
    try:
        import pytest  # noqa: PLC0415
    except ImportError as err:
        raise ImportError("Need pytest>=5.0.1 to run tests") from err
    cmd = ["--skip-slow"]
    if extra_args:
        if not isinstance(extra_args, list):
            extra_args = [extra_args]
        assert isinstance(extra_args, list)
        cmd = extra_args
    cmd += [PKG]
    joined = " ".join(cmd)
    print(f"running: pytest {joined}")
    result = pytest.main(cmd)
    if exit:
        sys.exit(result)
    return result
