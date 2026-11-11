import functools
import pickle
import sys

import numpy as np
from packaging.version import parse
import pytest

NP_LT_2 = parse(np.__version__) < parse("2.0.0")


class CustomPartial:
    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._part = functools.partial(func, *args, **kwargs)
        if sys.version_info >= (3, 13):
            self._part = staticmethod(self._part)

    @property
    def func(self):
        return self._func

    def __call__(self, *args, **kwargs):
        return self._part(*args, **kwargs)


def unpickle_numpy_generator(pickled: bytes) -> np.random.Generator:
    """
    Unpickle a NumPy Generator, xfailing a known limitation of NumPy < 2.0

    NumPy < 2.0 pickles a Generator using only the name of its bit generator.
    Unpickling builds the bit generator with its default configuration and then
    sets the state, so a bit generator with a non-default configuration (for
    example JSF with size=32 or PCG64 with variant="dxsm") cannot be restored
    and the state is rejected. NumPy >= 2.0 pickles the bit generator itself,
    which keeps the configuration. randomgen cannot change how NumPy pickles
    its Generator, so this limitation is expected on NumPy < 2.0 only.
    """
    try:
        return pickle.loads(pickled)
    except ValueError as exc:
        if NP_LT_2 and "Create a new" in str(exc):
            pytest.xfail(
                "NumPy < 2.0 unpickles a Generator with the default bit "
                "generator configuration"
            )
        raise
