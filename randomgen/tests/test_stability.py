import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import randomgen as rg
from randomgen.tests.data import stability_results
from randomgen.tests.data.compute_hashes import (
    BIG_ENDIAN,
    final_configurations,
    hash_configuration,
)
from randomgen.tests.data.stable_hashes import known_hashes

all_keys = list(known_hashes.keys())
all_ids = ["-".join(map(str, key)) for key in known_hashes]

minimal_keys = [all_keys[0]]
minimal_ids = [all_ids[0]]
for key, test_id in zip(all_keys, all_ids):
    if key[0] != minimal_keys[-1][0]:
        minimal_keys.append(key)
        minimal_ids.append(test_id)

EXECUTED: dict[tuple[str, str], dict[str, str]] = {}


@pytest.fixture(params=all_keys, ids=all_ids, scope="module")
def configuration(request):
    key = request.param
    if key in EXECUTED:
        return key, EXECUTED[key]
    EXECUTED[key] = hash_configuration(final_configurations[key])
    return key, EXECUTED[key]


@pytest.fixture(params=minimal_keys, ids=minimal_ids, scope="module")
def minimal_configuration(request):
    key = request.param
    if key in EXECUTED:
        return key, EXECUTED[key]
    EXECUTED[key] = hash_configuration(final_configurations[key])
    return key, EXECUTED[key]


@pytest.fixture(params=["random_values", "initial_state_hash", "final_state_hash"])
def hash_type(request):
    return request.param


@pytest.mark.slow
def test_stability(configuration, hash_type):
    key, result = configuration
    expected = known_hashes[key]
    assert expected[hash_type] == result[hash_type]


def test_basic_stability(minimal_configuration, hash_type):
    key, result = minimal_configuration
    expected = known_hashes[key]
    assert expected[hash_type] == result[hash_type]


def test_coverage():
    excluded = ["RDRAND", "UserBitGenerator"]
    required = []
    for obj_name in dir(rg):
        if obj_name.startswith("_") or obj_name in excluded:
            continue
        obj = getattr(rg, obj_name)
        if hasattr(obj, "random_raw"):
            required.append(obj_name)
    covered = {k[0] for k in known_hashes.keys()}
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


def test_speck():
    sed_seed = rg.SeedSequence(0)
    bit_gen = rg.SPECK128(sed_seed)
    bit_gen.random_raw(1)
    state = bit_gen.state
    round_key = np.array(
        [
            15793235383387715774,
            15793235383387715774,
            18284178489097793422,
            18284178489097793422,
            15216715163726632813,
            15216715163726632813,
            4485001273524005066,
            4485001273524005066,
            5155955939136278502,
            5155955939136278502,
            6746837586821959591,
            6746837586821959591,
            17240981843582736005,
            17240981843582736005,
            15913616031120359527,
            15913616031120359527,
            10736152986357956654,
            10736152986357956654,
            17578412605774299685,
            17578412605774299685,
            11746094550028141055,
            11746094550028141055,
            12349058196842705013,
            12349058196842705013,
            12083707282361316062,
            12083707282361316062,
            5359586807691737559,
            5359586807691737559,
            9231278599223380449,
            9231278599223380449,
            17651995584888691920,
            17651995584888691920,
            12732670852050470327,
            12732670852050470327,
            9879446609994856659,
            9879446609994856659,
            3375505017917309163,
            3375505017917309163,
            2895316718560254149,
            2895316718560254149,
            15513520198647448288,
            15513520198647448288,
            17647887543272890676,
            17647887543272890676,
            3893346907942253665,
            3893346907942253665,
            5738076513710395834,
            5738076513710395834,
            18229901514159150581,
            18229901514159150581,
            2621676699598916037,
            2621676699598916037,
            18118306437787214727,
            18118306437787214727,
            17976722144562375930,
            17976722144562375930,
            12676674378702818668,
            12676674378702818668,
            2422998901875546976,
            2422998901875546976,
            16988403034310549836,
            16988403034310549836,
            17955948645788572495,
            17955948645788572495,
            3967069179853279711,
            3967069179853279711,
            4366939844701077376,
            4366939844701077376,
        ],
        dtype=np.uint64,
    )
    ctr = np.array([6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0], dtype=np.uint64)
    assert_equal(state["state"]["ctr"], ctr)
    assert_equal(state["state"]["round_key"], round_key)
    assert_equal(state["state"]["offset"], 8)
    assert_equal(state["state"]["rounds"], 34)
    buffer = np.array(
        [
            10504129596355372223,
            1318022532533367005,
            6998045227391158025,
            381710739812180401,
            1300591519051043599,
            607837839416765688,
            17151091279695289275,
            13729019831678987542,
            6486144136891887409,
            6729719913441295618,
            15183619352825529528,
            13293035453993191235,
        ],
        dtype=np.uint64,
    )
    assert_equal(state["state"]["buffer"], buffer)


def test_sfmt():
    sed_seed = rg.SeedSequence(0)
    bit_gen = rg.SFMT(sed_seed)
    bit_gen.random_raw(1)
    state = bit_gen.state
    assert_equal(state["state"]["state"], stability_results.sfmt_state)
    assert_equal(state["state"]["idx"], stability_results.sfmt_idx)
    assert_equal(state["buffer_loc"], stability_results.sfmt_buffer_loc)
    assert_equal(state["buffered_uint64"], stability_results.sfmt_buffered_uint64)


def test_dsfmt():
    sed_seed = rg.SeedSequence(0)
    bit_gen = rg.DSFMT(sed_seed)
    bit_gen.random_raw(1)
    state = bit_gen.state
    assert_equal(state["state"]["idx"], stability_results.dsfmt_idx)
    assert_equal(state["buffer_loc"], stability_results.dsfmt_buffer_loc)
    assert_allclose(
        state["buffered_uniforms"], stability_results.dsfmt_buffered_uniforms, rtol=1e-5
    )
    assert_equal(state["state"]["state"], stability_results.dsfmt_state)
