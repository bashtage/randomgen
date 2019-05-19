from randomgen.xoshiro256 import Xoshiro256
import warnings


def Xoshiro256StarStar(*args, **kwargs):
    """
    This is a deprecation shim.  Use Xoshiro256
    """
    warnings.warn('Xoshiro256StarStar has been renamed Xoshiro256',
                  DeprecationWarning)
    return Xoshiro256(*args, **kwargs)
