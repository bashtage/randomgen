import pytest

from randomgen import Generator

random_gen = Generator()


def test_random_sample_deprecated():
    with pytest.deprecated_call():
        random_gen.random_sample()


def test_randint_deprecated():
    with pytest.deprecated_call():
        random_gen.randint(10)
