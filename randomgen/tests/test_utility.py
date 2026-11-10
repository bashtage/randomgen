import pickle

import numpy as np
import pytest

from randomgen.tests import _utility
from randomgen.tests._utility import unpickle_numpy_generator

NUMPY_1_ERROR = (
    "The state has size=32, but this JSF was created with size=64. "
    "Create a new JSF with the size of the state instead."
)


def _raise_value_error(message):
    raise ValueError(message)


class _FailsToUnpickle:
    """Raises ValueError with the given message when unpickled"""

    def __init__(self, message):
        self._message = message

    def __reduce__(self):
        return _raise_value_error, (self._message,)


def test_unpickle_numpy_generator_round_trip():
    gen = np.random.Generator(np.random.PCG64(1234))
    reloaded = unpickle_numpy_generator(pickle.dumps(gen))
    assert isinstance(reloaded, np.random.Generator)
    assert reloaded.bit_generator.state == gen.bit_generator.state


@pytest.mark.parametrize("np_lt_2", [True, False])
def test_unpickle_numpy_generator_unrelated_error(monkeypatch, np_lt_2):
    # Only the NumPy < 2 configuration error is expected, and any other error is raised
    monkeypatch.setattr(_utility, "NP_LT_2", np_lt_2)
    with pytest.raises(ValueError, match="unrelated"):
        unpickle_numpy_generator(pickle.dumps(_FailsToUnpickle("unrelated")))


def test_unpickle_numpy_generator_xfails_on_numpy_1(monkeypatch):
    monkeypatch.setattr(_utility, "NP_LT_2", True)
    with pytest.raises(pytest.xfail.Exception, match="default bit generator"):
        unpickle_numpy_generator(pickle.dumps(_FailsToUnpickle(NUMPY_1_ERROR)))


def test_unpickle_numpy_generator_raises_on_numpy_2(monkeypatch):
    # NumPy >= 2 restores the configuration, so the same error is a real failure
    monkeypatch.setattr(_utility, "NP_LT_2", False)
    with pytest.raises(ValueError, match="Create a new"):
        unpickle_numpy_generator(pickle.dumps(_FailsToUnpickle(NUMPY_1_ERROR)))
