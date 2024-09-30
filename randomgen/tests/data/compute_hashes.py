import hashlib
import itertools
import sys
import warnings

import numpy as np

from randomgen import (
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    LXM,
    MT64,
    MT19937,
    PCG32,
    PCG64,
    PCG64DXSM,
    SFC64,
    SFMT,
    SPECK128,
    AESCounter,
    ChaCha,
    LCG128Mix,
    Philox,
    Romu,
    Squares,
    ThreeFry,
    Tyche,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
)

warnings.filterwarnings("error", "The default value of inc")

try:
    from numpy.random import SeedSequence
except ImportError:
    from randomgen import SeedSequence


BIG_ENDIAN = sys.byteorder == "big"
SEED = np.array([4150402464, 4255976904, 1890890823, 1210894602], dtype=np.uint32)


def seed_seq():
    return SeedSequence(SEED)


UINT32 = 4083447352
UINT64 = 10998731014437268874
UINT128 = 332357023390495415282511736547623555666
UINT128_2 = 81642057924713167229618535206793462869
UINT256 = 34421456474825608341182049191702964259322525714408668276568880805363737678418
UINTS = {32: UINT32, 64: UINT64, 128: UINT128, 256: UINT256}


def fix_random_123(name, kwargs):
    if name not in ("Philox", "ThreeFry"):
        return kwargs
    width = kwargs.get("width", 64)
    number = kwargs.get("number", 4)
    nw = number * width
    if "counter" in kwargs:
        kwargs["counter"] = UINTS[nw]
    if "key" in kwargs:
        kwargs["key"] = UINTS[nw // 2]
    return kwargs


def int_to_array(int_val):
    arr_data = []
    while int_val > 0:
        arr_data.append(int_val % (2**24))
        int_val >>= 64
    return np.array(arr_data, dtype=np.uint64)


def hash_state(state, hasher=None, exclude_keys=()):
    if hasher is None:
        hasher = hashlib.sha256()
    for key, value in state.items():
        if isinstance(value, int):
            value = int_to_array(value)

        if isinstance(value, dict):
            hash_state(value, hasher, exclude_keys)
        elif isinstance(value, np.ndarray):
            if BIG_ENDIAN:
                value = value.byteswap()
            if key not in exclude_keys:
                # Skip excluded keys
                hasher.update(value.data)
        elif not isinstance(value, str):
            raise NotImplementedError(str(value))
    return hasher


def expand_kwargs(kwargs: dict):
    initial = {k: v for k, v in kwargs.items() if not isinstance(v, list)}
    initial_key = tuple(k for k in initial.keys())
    remaining = {k: v for k, v in kwargs.items() if k not in initial}
    expanded = [initial]
    expanded_keys = [initial_key]
    for key in remaining:
        new_expanded = []
        new_expanded_keys = []
        for a, b in itertools.product(expanded, kwargs[key]):
            a_copy = a.copy()
            a_copy[key] = b
            new_expanded.append(a_copy)
        for a, b in itertools.product(expanded_keys, kwargs[key]):
            new_expanded_keys.append(a + (key, b))
        expanded = new_expanded
        expanded_keys = new_expanded_keys
    return [(e, k) for e, k in zip(expanded, expanded_keys)]


configs = {
    "ChaCha": {
        "seed": seed_seq(),
        "counter": UINT128,
        "key": UINT128_2,
        "BLOCKED": (("seed", "key"),),
        "REQUIRED": ("seed", "key"),
    },
    "Xoroshiro128": {"seed": seed_seq(), "plusplus": [True, False]},
    "Xorshift1024": {"seed": seed_seq()},
    "Xoshiro256": {"seed": seed_seq()},
    "Xoshiro512": {"seed": seed_seq()},
    "EFIIX64": {"seed": seed_seq()},
    "LXM": {"seed": seed_seq()},
    "MT64": {"seed": seed_seq()},
    "MT19937": {"seed": seed_seq()},
    "DSFMT": {"seed": seed_seq(), "EXCLUDE_KEYS": ("buffered_uniforms",)},
    "SFC64": {"seed": seed_seq(), "k": [1, UINT64 | np.uint64(0x1)]},
    "SFMT": {"seed": seed_seq()},
    "SPECK128": {
        "seed": seed_seq(),
        "counter": UINT128,
        "key": UINT256,
        "BLOCKED": (("seed", "key"),),
        "REQUIRED": ("seed", "key"),
    },
    "PCG32": {"seed": seed_seq(), "inc": UINT64},
    "PCG64": {
        "seed": seed_seq(),
        "inc": [None, UINT128],
        "variant": ["xsl-rr", "dxsm-128", "dxsm"],
    },
    "PCG64DXSM": {"seed": seed_seq()},
    "AESCounter": {
        "seed": seed_seq(),
        "key": UINT128,
        "counter": UINT128_2,
        "BLOCKED": (("seed", "key"),),
        "REQUIRED": ("seed", "key"),
    },
    "HC128": {"seed": seed_seq()},
    "JSF": {"seed": seed_seq(), "seed_size": [1, 2, 3], "size": [32, 64]},
    "Philox": {
        "seed": seed_seq(),
        "number": [2, 4],
        "width": [32, 64],
        "counter": UINT256,
        "key": UINT128,
        "BLOCKED": (("seed", "key"),),
        "REQUIRED": ("seed", "key"),
    },
    "Romu": {"seed": seed_seq(), "variant": ["quad", "trio"]},
    "Squares": {
        "seed": seed_seq(),
        "key": 0xF9DB92E5A801E679,
        "variant": [32, 64],
        "BLOCKED": (("seed", "key"),),
        "REQUIRED": ("seed", "key"),
    },
    "ThreeFry": {
        "seed": seed_seq(),
        "number": [2, 4],
        "width": [32, 64],
        "counter": UINT256,
        "key": UINT128,
        "BLOCKED": (("seed", "key"),),
        "REQUIRED": ("seed", "key"),
    },
    "Tyche": {"seed": seed_seq(), "original": [True, False]},
    "LCG128Mix": {
        "seed": seed_seq(),
        "inc": [0, None],
        "output": ["xsl-rr", "dxsm"],
        "post": [True, False],
        "dxsm_multiplier": [0x9E3779B97F4A7C15, 0xDA942042E4DD58B5],
        "multiplier": [
            0x278F2419A4D3A5F7C2280069635487FD,
            0x2360ED051FC65DA44385DF649FCCF645,
        ],
    },
}

BIT_GEN = {
    "ChaCha": ChaCha,
    "Xoroshiro128": Xoroshiro128,
    "Xorshift1024": Xorshift1024,
    "Xoshiro256": Xoshiro256,
    "Xoshiro512": Xoshiro512,
    "EFIIX64": EFIIX64,
    "LXM": LXM,
    "MT64": MT64,
    "MT19937": MT19937,
    "DSFMT": DSFMT,
    "SFMT": SFMT,
    "Romu": Romu,
    "SPECK128": SPECK128,
    "PCG32": PCG32,
    "PCG64": PCG64,
    "PCG64DXSM": PCG64DXSM,
    "JSF": JSF,
    "SFC64": SFC64,
    "AESCounter": AESCounter,
    "HC128": HC128,
    "Philox": Philox,
    "ThreeFry": ThreeFry,
    "LCG128Mix": LCG128Mix,
    "Tyche": Tyche,
    "Squares": Squares,
}

final_configurations = {}
for gen in configs:
    args = configs[gen]
    bg = BIT_GEN[gen]
    blocked = args.get("BLOCKED", ())
    required = args.get("REQUIRED", ("seed",))
    exclude_keys = args.get("EXCLUDE_KEYS", ())
    keys = [k for k in args if k not in ("BLOCKED", "REQUIRED", "EXCLUDE_KEYS")]

    for i in range(1, len(keys) + 1):
        for comb in itertools.combinations(keys, i):
            found = False
            for block in blocked:
                found = found or set(block).issubset(set(list(comb)))
            if found:
                continue
            diff = set(list(required)).difference(set(list(comb)))
            if len(diff) == len(required):
                continue
            kwargs = {key: args[key] for key in comb}
            for ex_kwargs, ex_key in expand_kwargs(kwargs):
                if "seed" in ex_kwargs:
                    ex_kwargs["seed"] = seed_seq()
                ex_kwargs = fix_random_123(gen, ex_kwargs)
                if bg == PCG64 and "inc" not in kwargs:
                    # Special case exclude this variant to reduce noise
                    continue
                final_key = (gen,) + ex_key
                bit_gen = bg(**ex_kwargs)
                final_configurations[final_key] = bit_gen


def hash_configuration(bit_gen):
    state_hash = hash_state(bit_gen.state, exclude_keys=exclude_keys).hexdigest()
    randoms = bit_gen.random_raw(1000000)
    if BIG_ENDIAN:
        randoms = randoms.byteswap()
    return {
        "random_values": hashlib.sha256(randoms.data).hexdigest(),
        "initial_state_hash": state_hash,
        "final_state_hash": hash_state(bit_gen.state).hexdigest(),
    }
