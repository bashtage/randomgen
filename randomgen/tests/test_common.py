import numpy as np
import pytest

from randomgen.aes import AESCounter
from randomgen.common import BitGenerator, interface
from randomgen.entropy import seed_by_array
from randomgen.lxm import LXM
from randomgen.romu import Romu
from randomgen.tests._shims import (
    byteswap_little_endian_shim,
    int_to_array_shim,
    object_to_int_shim,
    view_little_endian_shim,
)

MISSING_CFFI = False
try:
    import cffi  # noqa: F401
except ImportError:
    MISSING_CFFI = True


def test_view_little_endian():
    a = np.uint64([2**63])
    b = view_little_endian_shim(a, np.uint32)
    expected = np.array([0, 2147483648], dtype=np.uint32)
    np.testing.assert_equal(b, expected)

    c = view_little_endian_shim(b, np.uint64)
    np.testing.assert_equal(c, a)


def test_view_little_endian_err():
    a = np.double([2**51])
    with pytest.raises(ValueError):
        view_little_endian_shim(a, np.uint32)
    a = np.uint64([2**63])
    with pytest.raises(ValueError):
        view_little_endian_shim(a, np.uint16)
    a = np.array([[2**63]], np.uint64)
    with pytest.raises(ValueError):
        view_little_endian_shim(a, np.uint32)


def test_int_to_array():
    seed = 3735928495
    result = int_to_array_shim(seed, "seed", None, 64)
    expected = np.array([3735928495], dtype=np.uint64)
    np.testing.assert_equal(result, expected)

    seed = 0
    for pow in (255, 129, 93, 65, 63, 33, 1, 0):
        seed += 2**pow
    result = int_to_array_shim(seed, "seed", None, 64)
    expected = np.array(
        [9223372045444710403, 536870914, 2, 9223372036854775808], dtype=np.uint64
    )
    np.testing.assert_equal(result, expected)
    result = int_to_array_shim(seed, "seed", None, 32)
    expected = np.array(
        [3, 2147483650, 536870914, 0, 2, 0, 0, 2147483648], dtype=np.uint32
    )
    np.testing.assert_equal(result, expected)


def test_int_to_array_errors():
    with pytest.raises(ValueError):
        int_to_array_shim(1, "a", 64, 31)
    with pytest.raises(TypeError):
        int_to_array_shim("1", "a", 64, 64)
    with pytest.raises(ValueError):
        int_to_array_shim(-1, "a", 64, 64)
    with pytest.raises(ValueError):
        int_to_array_shim(2**96, "a", 64, 64)
    with pytest.raises(ValueError):
        int_to_array_shim([-1], "a", 64, 64)
    with pytest.raises(ValueError):
        int_to_array_shim([1, 2**96], "a", 64, 32)
    with pytest.raises(ValueError):
        int_to_array_shim(np.array([1], dtype=np.uint32), "a", 64, 32)


def test_seed_array():
    seed_arr = np.array([3735928495], dtype=np.uint64)
    result = seed_by_array(seed_arr, 2)
    expected = np.array([4555448849277713929, 5170625396769938207], dtype=np.uint64)
    np.testing.assert_equal(result, expected)

    seed_arr = np.array(
        [9223372045444710403, 536870914, 2, 9223372036854775808], dtype=np.uint64
    )
    result = seed_by_array(seed_arr, 4)
    expected = np.array(
        [
            7178927994527075522,
            9215441430954639631,
            7276951224970988593,
            1810038055801910983,
        ],
        dtype=np.uint64,
    )
    np.testing.assert_equal(result, expected)


def test_seed_array_errors():
    with pytest.raises(TypeError):
        seed_by_array(np.array([0.0 + 1j]), 1)
    with pytest.raises(TypeError):
        seed_by_array("1", 1)
    with pytest.raises(TypeError):
        seed_by_array(1.2, 1)
    with pytest.raises(ValueError):
        seed_by_array(-1, 1)
    with pytest.raises(ValueError):
        seed_by_array(2**65, 1)
    with pytest.raises(ValueError):
        seed_by_array([[1, 2], [3, 4]], 1)
    with pytest.raises(TypeError):
        seed_by_array(np.array([1, 2 + 1j]), 1)
    with pytest.raises(ValueError):
        seed_by_array([2**65], 1)
    with pytest.raises(ValueError):
        seed_by_array([-1], 1)
    with pytest.raises(TypeError):
        seed_by_array([1.2], 1)


def test_byteswap_little_endian():
    a = np.array([9241421688590303745], dtype=np.uint64)
    result = byteswap_little_endian_shim(a).view(np.uint8)
    expected = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    np.testing.assert_equal(result, expected)


def test_bitgenerator_error():
    with pytest.raises(NotImplementedError, match="BitGenerator is a base class"):
        BitGenerator(1)


@pytest.mark.skipif(MISSING_CFFI, reason="Requires CFFI")
def test_cffi():

    tpl = Romu().cffi
    assert isinstance(tpl, interface)
    assert hasattr(tpl, "state_address")
    assert hasattr(tpl, "state")
    assert hasattr(tpl, "next_uint64")
    assert hasattr(tpl, "next_uint32")
    assert hasattr(tpl, "next_double")
    assert hasattr(tpl, "bit_generator")


def test_object_to_int():
    res = object_to_int_shim(1, 32, "test", allowed_sizes=(32, 64))
    assert isinstance(res, int)
    res = object_to_int_shim(1, 32, "test", default_bits=32, allowed_sizes=(32, 64))
    assert isinstance(res, int)
    res = object_to_int_shim(
        np.array(1, dtype=np.uint64),
        32,
        "test",
        default_bits=32,
        allowed_sizes=(32, 64),
    )
    assert isinstance(res, int)
    res = object_to_int_shim([1], 32, "test", allowed_sizes=(32, 64))
    assert isinstance(res, int)
    res = object_to_int_shim(
        np.array(1, dtype=np.uint32), 32, "test", allowed_sizes=(32, 64)
    )
    assert isinstance(res, int)


def test_object_to_int_errors():
    with pytest.raises(TypeError):
        object_to_int_shim(np.array([1.2]), 32, "test", allowed_sizes=(32, 64))
    with pytest.raises(ValueError):
        object_to_int_shim(["a"], 32, "test", allowed_sizes=(32, 64))
    with pytest.raises(TypeError):
        object_to_int_shim([1.2], 32, "test", allowed_sizes=(32, 64))
    with pytest.raises(ValueError):
        object_to_int_shim(
            np.array([[1, 2], [3, 4]], dtype=np.uint32),
            32,
            "test",
            allowed_sizes=(32, 64),
        )
    with pytest.raises(ValueError):
        object_to_int_shim(
            [sum(2**i for i in range(63))],
            128,
            "test",
            default_bits=32,
            allowed_sizes=(32, 64),
        )


def test_uncupported_mode():
    with pytest.raises(ValueError, match="mode must be"):
        with pytest.warns(FutureWarning):
            AESCounter(mode="unsupported")


def test_check_state_array_no_array():
    bg = LXM()
    state = bg.state
    state["state"]["x"] = state["state"]["x"].tolist()
    bg.state = state
    with pytest.raises(ValueError):
        state["state"]["x"] = state["state"]["x"][:2]
        bg.state = state
