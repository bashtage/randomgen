from __future__ import annotations

from collections import defaultdict
from typing import NamedTuple

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from randomgen.pcg64 import PCG64
from randomgen.tests._shims import ShimGenerator


class Config(NamedTuple):
    a: float | np.ndarray | None
    b: float | np.ndarray | None
    c: float | np.ndarray | None
    size: tuple[int, ...] | None
    out: np.ndarray | None
    generator: ShimGenerator


CONFIGS = defaultdict(list)


def all_scalar(*args):
    return all([(arg is None or np.isscalar(arg)) for arg in args])


def get_broadcastable_size(a, b, c):
    if c is not None:
        return (a + b + c).shape
    if b is not None:
        return (a + b).shape
    if a is not None:
        return a.shape


def count_params(a, b, c):
    return sum((v is not None) for v in (a, b, c))


for a in (None, 0.5, 0.5 * np.ones((1, 2)), 0.5 * np.ones((3, 2))):
    for b in (
        None,
        0.5,
        0.5 * np.ones((1, 2)),
        0.5 * np.ones((3, 1)),
        0.5 * np.ones((3, 2)),
    ):
        if a is None and b is not None:
            continue
        for c in (
            None,
            1.5,
            1.5 * np.ones((1, 2)),
            1.5 * np.ones((3, 1)),
            1.5 * np.ones((3, 2)),
        ):
            if b is None and c is not None:
                continue
            for size in (True, False):
                for out in (True, False):
                    if size:
                        if all_scalar(a, b, c):
                            _size = (7, 5)
                        else:
                            _size = get_broadcastable_size(a, b, c)
                    else:
                        _size = None
                    if out:
                        if size:
                            _out = np.empty(_size)
                        elif all_scalar(a, b, c):
                            _out = np.empty((11, 7))
                        else:
                            _out = np.empty(get_broadcastable_size(a, b, c))
                    else:
                        _out = None
                    print(_size, _out.shape if isinstance(_out, np.ndarray) else _out)
                    generator = ShimGenerator(PCG64())
                    CONFIGS[count_params(a, b, c)].append(
                        Config(a, b, c, _size, _out, generator)
                    )


@pytest.mark.parametrize("config", CONFIGS[0])
def test_cont_0(config):
    res = config.generator.cont_0(size=config.size, out=config.out)
    if isinstance(res, np.ndarray):
        assert_allclose(res, 3.141592 * np.ones_like(res))
    else:
        assert_allclose(res, 3.141592)


@pytest.mark.parametrize("config", CONFIGS[1])
def test_cont_1(config):
    res = config.generator.cont_1(config.a, size=config.size, out=config.out)
    if isinstance(res, np.ndarray):
        assert_allclose(res, 0.5 * np.ones_like(res))
    else:
        assert_allclose(res, 0.5)


@pytest.mark.parametrize("config", CONFIGS[2])
def test_cont_2(config):
    res = config.generator.cont_2(config.a, config.b, size=config.size, out=config.out)
    if isinstance(res, np.ndarray):
        assert_allclose(res, np.ones_like(res))
    else:
        assert_allclose(res, 1.0)


@pytest.mark.parametrize("config", CONFIGS[3])
def test_cont_3(config):
    res = config.generator.cont_3(
        config.a, config.b, config.c, size=config.size, out=config.out
    )
    if isinstance(res, np.ndarray):
        assert_allclose(res, 2.5 * np.ones_like(res))
    else:
        assert_allclose(res, 2.5)


@pytest.mark.parametrize("config", CONFIGS[3])
def test_cont_3_alt_cons(config):
    res = config.generator.cont_3_alt_cons(
        1.0 + config.a, config.b, config.c, size=config.size, out=config.out
    )
    if isinstance(res, np.ndarray):
        assert_allclose(res, 3.5 * np.ones_like(res))
    else:
        assert_allclose(res, 3.5)


@pytest.mark.parametrize("config", CONFIGS[1])
def test_cont_1_float(config):
    if isinstance(config.a, np.ndarray):
        a = config.a.astype(np.float32)
    else:
        a = config.a
    out = None
    if config.out is not None:
        out = np.empty(config.out.shape, dtype=np.float32)

    res = config.generator.cont_1_float(a, size=config.size, out=out)
    if isinstance(res, np.ndarray):
        assert_allclose(res, 0.5 * np.ones_like(res))
    else:
        assert_allclose(res, 0.5)


@pytest.mark.parametrize("config", CONFIGS[1])
def test_disc_0(config):
    res = config.generator.disc_0(size=config.size)
    if isinstance(res, int):
        assert res == 3
    else:
        assert_array_equal(res, np.full_like(res, 3))


@pytest.mark.parametrize("config", CONFIGS[1])
def test_disc_d(config):
    res = config.generator.disc_d(config.a, size=config.size)
    if isinstance(res, int):
        assert res == 5
    else:
        assert_array_equal(res, np.full_like(res, 5))


@pytest.mark.parametrize("config", CONFIGS[2])
def test_disc_dd(config):
    res = config.generator.disc_dd(config.a, config.b, size=config.size)
    if isinstance(res, int):
        assert res == 2
    else:
        assert_array_equal(res, np.full_like(res, 2))


@pytest.mark.parametrize("config", CONFIGS[2])
def test_disc_di(config):
    b = 10 * config.b
    b = int(b) if isinstance(b, float) else b.astype(np.int64)
    res = config.generator.disc_di(config.a, b, size=config.size)
    if isinstance(res, int):
        assert res == 5
    else:
        assert_array_equal(res, np.full_like(res, 5))


@pytest.mark.parametrize("config", CONFIGS[1])
def test_disc_i(config):
    a = 10 * config.a
    a = int(a) if isinstance(a, float) else a.astype(np.int64)
    res = config.generator.disc_i(a, size=config.size)
    if isinstance(res, int):
        assert res == 5
    else:
        assert_array_equal(res, np.full_like(res, 5))


@pytest.mark.parametrize("config", CONFIGS[3])
def test_disc_iii(config):
    a = 10 * config.a
    a = int(a) if isinstance(a, float) else a.astype(np.int64)
    b = 10 * config.b
    b = int(b) if isinstance(b, float) else b.astype(np.int64)
    c = 10 * config.c
    c = int(c) if isinstance(c, float) else c.astype(np.int64)
    res = config.generator.disc_iii(a, b, c, size=config.size)
    if isinstance(res, int):
        assert res == 25
    else:
        assert_array_equal(res, np.full_like(res, 25))
