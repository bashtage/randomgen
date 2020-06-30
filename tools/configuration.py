from collections import defaultdict

import jinja2

from randomgen import (
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    LXM,
    MT19937,
    PCG64,
    SFC64,
    SFMT,
    SPECK128,
    AESCounter,
    ChaCha,
    LCG128Mix,
    Philox,
    Romu,
    ThreeFry,
    Xoshiro256,
    Xoshiro512,
)

ALL_BIT_GENS = [
    AESCounter,
    ChaCha,
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    LXM,
    PCG64,
    LCG128Mix,
    MT19937,
    Philox,
    SFC64,
    SFMT,
    SPECK128,
    ThreeFry,
    Xoshiro256,
    Xoshiro512,
    Romu,
]
JUMPABLE = [bg for bg in ALL_BIT_GENS if hasattr(bg, "jumped")]

SPECIALS = {
    ChaCha: {"rounds": [8, 20]},
    JSF: {"seed_size": [1, 3]},
    SFC64: {"k": [1, 3394385948627484371, "weyl"]},
    LCG128Mix: {"output": ["upper"]},
    PCG64: {"variant": ["dxsm", "dxsm-128", "xsl-rr"]},
    Romu: {"variant": ["quad", "trio"]},
}
OUTPUT = defaultdict(lambda: 64)
OUTPUT.update({MT19937: 32, DSFMT: 32})
with open("templates/configuration.jinja") as tmpl:
    TEMPLATE = jinja2.Template(tmpl.read())

DSFMT_WRAPPER = """\

class Wrapper32:
    def __init__(self, seed, **kwargs):
        if isinstance(seed, rg.DSFMT):
            self._bit_gen = seed
        else:
            self._bit_gen = rg.DSFMT(seed)

    def random_raw(self, n=None):
        return self._bit_gen.random_raw(n).astype("u4")

    def jumped(self):
        return Wrapper32(self._bit_gen.jumped())

rg.Wrapper32 = Wrapper32
"""
# Specials
# SFC64
DEFAULT_ENTOPY = (
    86316980830225721106033794313786972513572058861498566720023788662568817403978
)
