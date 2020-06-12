import numpy as np
from numpy.testing import assert_equal
import pytest

import randomgen as rg
from randomgen.tests.data.compute_hashes import BIG_ENDIAN, computed_hashes
from randomgen.tests.data.stable_hashes import known_hashes

keys = list(known_hashes.keys())
ids = ["-".join(map(str, key)) for key in known_hashes]


@pytest.mark.parametrize("key", keys, ids=ids)
@pytest.mark.parametrize(
    "hash_type", ["random_values", "initial_state_hash", "final_state_hash"]
)
def test_stability(key, hash_type):
    expected = known_hashes[key]
    computed = computed_hashes[key]
    assert expected[hash_type] == computed[hash_type]


def test_coverage():
    excluded = ["RDRAND", "UserBitGenerator"]
    required = []
    for obj_name in dir(rg):
        if obj_name.startswith("_") or obj_name in excluded:
            continue
        obj = getattr(rg, obj_name)
        if hasattr(obj, "random_raw"):
            required.append(obj_name)
    covered = set(k[0] for k in known_hashes.keys())
    assert not set(required).difference(covered)
    assert not set(covered).difference(required)


def test_aes():
    sed_seed = rg.SeedSequence(0)
    bit_gen = rg.AESCounter(sed_seed)
    bit_gen.random_raw(1)
    state = bit_gen.state
    assert_equal(
        state["s"]["counter"], np.array([4, 0, 5, 0, 6, 0, 7, 0], dtype=np.uint64)
    )
    assert_equal(state["s"]["offset"], 8)
    assert_equal(
        state["s"]["seed"],
        np.array(
            [
                15793235383387715774,
                12390638538380655177,
                14535948455753730044,
                5638452070979447378,
                9872913402542273141,
                2519700719880636476,
                6568914210857343685,
                1519603011295473815,
                10978339004909591939,
                13451446543017809343,
                1287446329370806453,
                345470396342368290,
                7757725448560514988,
                15061683594572235283,
                3306634883589962969,
                2965863995102004475,
                18090342262254749848,
                3028696162780629643,
                9873779358970902643,
                11542281185009237128,
                15127027345177397562,
                18151169116043183025,
            ],
            dtype=np.uint64,
        ),
    )
    expected_state = np.array(
        [
            12391658052641588024,
            10714498261726844411,
            16302282886881395158,
            226768610308972488,
            3645593224307478361,
            6535853598405435806,
            13777657176506182534,
            3383728099036958951,
        ],
        dtype=np.uint64,
    )
    if BIG_ENDIAN:
        expected_state = expected_state.byteswap()
    expected_state = expected_state.view(np.uint8)

    assert_equal(state["s"]["state"], expected_state)
