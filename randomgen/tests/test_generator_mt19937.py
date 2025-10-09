import hashlib
import sys

import numpy as np
from numpy.testing import assert_raises
from packaging.version import parse
import pytest

from randomgen import MT19937

NP_LT_118 = parse(np.__version__) < parse("1.18.0")

JUMP_TEST_DATA = {
    ("_jump_tester", (0,), 10): {
        "initial": {"key_md5": "64eaf265d2203179fb5ffb73380cd589", "pos": 9},
        "jumped": {"key_md5": "14e9a7d1e247f0f8565b77784c9a6b83", "pos": 601},
    },
    ("_jump_tester", (384908324,), 312): {
        "initial": {"key_md5": "e99708a47b82ff51a2c7b0625b81afb5", "pos": 311},
        "jumped": {"key_md5": "8bfd5e1ab46befd06cc54146541f1ce8", "pos": 279},
    },
    ("_jump_tester", (839438204, 980239840, 859048019, 821), 511): {
        "initial": {"key_md5": "9fcd6280df9199785e17e93162ce283c", "pos": 510},
        "jumped": {"key_md5": "f8ac8f010bd3eabc8afbc8b690220177", "pos": 478},
    },
    ("jumped", (0,), 10): {
        "initial": {"key_md5": "64eaf265d2203179fb5ffb73380cd589", "pos": 9},
        "jumped": {"key_md5": "8cb7b061136efceef5217a9ce2cc9a5a", "pos": 598},
    },
    ("jumped", (384908324,), 312): {
        "initial": {"key_md5": "e99708a47b82ff51a2c7b0625b81afb5", "pos": 311},
        "jumped": {"key_md5": "2ecdbfc47a895b253e6e19ccb2e74b90", "pos": 276},
    },
    ("jumped", (839438204, 980239840, 859048019, 821), 511): {
        "initial": {"key_md5": "9fcd6280df9199785e17e93162ce283c", "pos": 510},
        "jumped": {"key_md5": "433b85229f2ed853cde06cd872818305", "pos": 475},
    },
}


@pytest.fixture(scope="module", params=[True, False])
def endpoint(request):
    return request.param


class TestSeed:
    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, MT19937, -0.5)
        assert_raises(ValueError, MT19937, -1)

    def test_invalid_array(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, MT19937, [-0.5])
        assert_raises(ValueError, MT19937, [-1])


@pytest.mark.skipif(NP_LT_118, reason="Can only test with NumPy >= 1.18")
@pytest.mark.parametrize("config", list(JUMP_TEST_DATA.keys()))
def test_jumped(config):
    values = JUMP_TEST_DATA[config]
    typ, seed_tpl, step = config

    seed = seed_tpl[0] if len(seed_tpl) == 1 else list(seed_tpl)
    initial_state = np.random.MT19937(seed).state
    mt19937 = MT19937()
    mt19937.state = initial_state
    mt19937.random_raw(step)
    if typ == "jumped":
        jumped = mt19937.jumped()
    else:
        jumped = mt19937._jump_tester()
    key = jumped.state["state"]["key"]
    if sys.byteorder == "big":
        key = key.byteswap()
    md5 = hashlib.md5(key)
    assert md5.hexdigest() == values["jumped"]["key_md5"]
    assert jumped.state["state"]["pos"] == values["jumped"]["pos"]
