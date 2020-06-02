import warnings

from randomgen.xoshiro512 import Xoshiro512


def Xoshiro512StarStar(*args, **kwargs):
    """
    This is a deprecation shim.  Use Xoshiro512
    """
    warnings.warn("Xoshiro512StarStar has been renamed Xoshiro512", DeprecationWarning)
    return Xoshiro512(*args, **kwargs)
