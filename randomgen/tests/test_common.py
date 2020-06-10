import numpy as np
import pytest

from randomgen.entropy import seed_by_array
from randomgen.tests._shims import (
    byteswap_little_endian_shim,
    int_to_array_shim,
    view_little_endian_shim,
)


def test_view_little_endian():
    a = np.uint64([2 ** 63])
    b = view_little_endian_shim(a, np.uint32)
    expected = np.array([0, 2147483648], dtype=np.uint32)
    np.testing.assert_equal(b, expected)

    c = view_little_endian_shim(b, np.uint64)
    np.testing.assert_equal(c, a)


def test_view_little_endian_err():
    a = np.double([2 ** 51])
    with pytest.raises(ValueError):
        view_little_endian_shim(a, np.uint32)
    a = np.uint64([2 ** 63])
    with pytest.raises(ValueError):
        view_little_endian_shim(a, np.uint16)
    a = np.array([[2 ** 63]], np.uint64)
    with pytest.raises(ValueError):
        view_little_endian_shim(a, np.uint32)


def test_int_to_array():
    seed = 3735928495
    result = int_to_array_shim(seed, "seed", None, 64)
    expected = np.array([3735928495], dtype=np.uint64)
    np.testing.assert_equal(result, expected)

    seed = 0
    for pow in (255, 129, 93, 65, 63, 33, 1, 0):
        seed += 2 ** pow
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


def test_byteswap_little_endian():
    a = np.array([9241421688590303745], dtype=np.uint64)
    result = byteswap_little_endian_shim(a).view(np.uint8)
    expected = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
    np.testing.assert_equal(result, expected)
