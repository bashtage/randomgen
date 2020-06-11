import pytest

from randomgen.tests.data.stable_hashes import known_hashes
from randomgen.tests.data.compute_hashes import computed_hashes

keys = list(known_hashes.keys())
ids = ["-".join(map(str,key)) for key in known_hashes]


@pytest.mark.parametrize("key", keys, ids=ids)
@pytest.mark.parametrize(
    "hash_type", ["random_values", "initial_state_hash", "final_state_hash"]
)
def test_stability(key, hash_type):
    expected = known_hashes[key]
    computed = computed_hashes[key]
    assert expected[hash_type] == computed[hash_type]
