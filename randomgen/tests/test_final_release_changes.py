import pytest

from randomgen import (
    DSFMT,
    MT19937,
    PCG32,
    PCG64,
    Philox,
    ThreeFry,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
)

bit_generators = [
    DSFMT,
    MT19937,
    PCG32,
    PCG64,
    Philox,
    ThreeFry,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
]


@pytest.fixture(scope="module", params=bit_generators)
def bit_generator(request):
    return request.param


@pytest.fixture(scope="module", params=[True, False])
def endpoint(request):
    return request.param


def test_generator_raises(bit_generator):
    bg = bit_generator()
    with pytest.raises(NotImplementedError):
        bg.generator
