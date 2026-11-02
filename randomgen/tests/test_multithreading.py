from functools import partial
import os
import random
import subprocess
import sys
import sysconfig
import threading

import numpy as np
import pytest

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
    RDRAND,
    SFC64,
    SFMT,
    SPECK128,
    AESCounter,
    BlaBla,
    ChaCha,
    ExtendedGenerator,
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
from randomgen.squares import generate_keys

try:
    RDRAND()
    HAS_RDRAND = True
except RuntimeError:
    HAS_RDRAND = False

BIT_GENERATORS = [
    AESCounter,
    BlaBla,
    ChaCha,
    DSFMT,
    EFIIX64,
    HC128,
    JSF,
    partial(JSF, size=32),
    LXM,
    MT19937,
    MT64,
    PCG32,
    PCG64,
    partial(PCG64, variant="xsl-rr"),
    LCG128Mix,
    Philox,
    partial(Philox, number=2, width=32),
    Romu,
    SFC64,
    SFMT,
    SPECK128,
    Squares,
    partial(Squares, variant=32),
    ThreeFry,
    partial(ThreeFry, number=2, width=32),
    Tyche,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
]


def _name(bit_gen):
    if isinstance(bit_gen, partial):
        kwargs = ", ".join(f"{k}={v}" for k, v in bit_gen.keywords.items())
        return f"{bit_gen.func.__name__}({kwargs})"
    return bit_gen.__name__


IDS = [_name(bg) for bg in BIT_GENERATORS]


def run_threaded(func, n_threads=8, timeout=120):
    barrier = threading.Barrier(n_threads)
    errors = []

    def target(i):
        try:
            barrier.wait()
            func(i)
        except BaseException as exc:  # pragma: no cover
            errors.append(exc)

    threads = [
        threading.Thread(target=target, args=(i,), daemon=True)
        for i in range(n_threads)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "Deadlock or timeout"
    if errors:
        raise errors[0]


def assert_state_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    else:
        assert actual == expected


def _advance(bit_gen, delta):
    if isinstance(bit_gen, (Philox, ThreeFry)):
        bit_gen.advance(delta, counter=False)
    else:
        bit_gen.advance(delta)


@pytest.mark.parametrize("bit_gen", BIT_GENERATORS, ids=IDS)
def test_lock_is_reentrant(bit_gen):
    bg = bit_gen(12345)
    with bg.lock:
        assert bg.lock.acquire(blocking=False)
        bg.lock.release()
        # These all acquire the lock internally
        bg.state = bg.state
        bg.seed(12345)
        if hasattr(bg, "advance"):
            _advance(bg, 10)


@pytest.mark.parametrize("bit_gen", BIT_GENERATORS, ids=IDS)
def test_parallel_seed_state(bit_gen):
    seeds = [12345, 67890]
    bg = bit_gen(seeds[0])
    expected = []
    for seed in seeds:
        bg.seed(seed)
        expected.append(bg.state)

    def func(i):
        rnd = random.Random(i)
        for _ in range(50):
            if rnd.randrange(2):
                bg.seed(rnd.choice(seeds))
            else:
                state = bg.state
                matched = 0
                for exp in expected:
                    try:
                        assert_state_equal(state, exp)
                        matched += 1
                    except AssertionError:
                        pass
                assert matched == 1, "state read while seeding was in progress"

    run_threaded(func)


@pytest.mark.parametrize("bit_gen", BIT_GENERATORS, ids=IDS)
def test_parallel_state_access(bit_gen):
    bg = bit_gen(12345)
    gen = np.random.Generator(bg)
    eg = ExtendedGenerator(bg)

    def func(i):
        rnd = random.Random(i)
        for _ in range(50):
            branch = rnd.randrange(6)
            if branch == 0:
                bg.state = bg.state
            elif branch == 1:
                gen.random(10)
            elif branch == 2:
                bg.random_raw(5)
            elif branch == 3:
                eg.complex_normal()
            elif branch == 4 and hasattr(bg, "advance"):
                _advance(bg, 10)
            elif branch == 4 and hasattr(bg, "jumped"):
                bg.jumped()
            else:
                _ = bg.state

    run_threaded(func)


def test_parallel_squares_keys():
    expected = generate_keys(12345, 100)
    assert np.all(expected & np.uint64(1))

    def func(i):
        for _ in range(20):
            np.testing.assert_array_equal(generate_keys(12345, 100), expected)
            assert Squares(12345).state == Squares(12345).state

    run_threaded(func)


@pytest.mark.skipif(not HAS_RDRAND, reason="RDRAND is not available")
def test_parallel_rdrand():
    bg = RDRAND()
    gen = np.random.Generator(bg)
    with bg.lock:
        assert bg.lock.acquire(blocking=False)
        bg.lock.release()
        _ = bg.state

    def func(i):
        rnd = random.Random(i)
        for _ in range(50):
            branch = rnd.randrange(3)
            if branch == 0:
                gen.random(10)
            elif branch == 1:
                bg.random_raw(5)
            else:
                assert bg.success

    run_threaded(func)


@pytest.mark.skipif(
    not sysconfig.get_config_var("Py_GIL_DISABLED"),
    reason="Requires a free-threaded build of Python",
)
def test_gil_not_reenabled():
    code = """\
import importlib, importlib.machinery, pathlib, sys, warnings
warnings.simplefilter("error", RuntimeWarning)
import randomgen
suffixes = [
    s for s in importlib.machinery.EXTENSION_SUFFIXES if s not in (".so", ".pyd")
]
root = pathlib.Path(randomgen.__file__).parent
n_ext = 0
for path in root.rglob("*"):
    if any(path.name.endswith(s) for s in suffixes):
        parts = path.relative_to(root.parent).parts
        name = ".".join(parts[:-1] + (path.name.split(".")[0],))
        importlib.import_module(name)
        n_ext += 1
assert n_ext > 20
assert not sys._is_gil_enabled()
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
