import pytest

from randomgen import (
    DSFMT,
    MT19937,
    PCG32,
    PCG64,
    Generator,
    Philox,
    ThreeFry,
    Xoroshiro128,
    Xorshift1024,
    Xoshiro256,
    Xoshiro512,
)

random_gen = Generator()

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


def test_random_sample_deprecated():
    with pytest.deprecated_call():
        random_gen.random_sample()


def test_randint_deprecated():
    with pytest.deprecated_call():
        random_gen.randint(10)


def test_rand_deprecated():
    with pytest.deprecated_call():
        random_gen.rand(10)


def test_randn_deprecated():
    with pytest.deprecated_call():
        random_gen.randn(10)


def test_generator_raises(bit_generator):
    bg = bit_generator(mode="sequence")
    with pytest.raises(NotImplementedError):
        bg.generator


def test_integers_closed():
    with pytest.deprecated_call():
        random_gen.integers(0, 10, closed=True)
    with pytest.deprecated_call():
        random_gen.integers(0, 10, closed=False)


def test_integers_use_masked():
    with pytest.deprecated_call():
        random_gen.integers(0, 10, use_masked=True)


def test_integers_large_negative_value():
    with pytest.raises(ValueError):
        random_gen.integers(0, -(2 ** 65), endpoint=endpoint)
    with pytest.raises(ValueError):
        random_gen.integers(0, [-(2 ** 65)], endpoint=endpoint)
