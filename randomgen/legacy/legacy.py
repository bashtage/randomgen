from randomgen.generator import RandomGenerator
from randomgen.mt19937 import MT19937
from randomgen.legacy._legacy import _LegacyGenerator
import randomgen.pickle


_LEGACY_ATTRIBUTES = tuple(a for a in dir(
    _LegacyGenerator) if not a.startswith('_'))

_LEGACY_ATTRIBUTES += ('__getstate__', '__setstate__', '__reduce__')


def with_metaclass(meta, *bases):
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.

    # From six, https://raw.githubusercontent.com/benjaminp/six
    class metaclass(type):

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)
    return type.__new__(metaclass, 'temporary_class', (), {})


class LegacyGeneratorType(type):
    def __getattribute__(self, name):
        if name in _LEGACY_ATTRIBUTES:
            return object.__getattribute__(_LegacyGenerator, name)
        return object.__getattribute__(RandomGenerator, name)


class LegacyGenerator(with_metaclass(LegacyGeneratorType, RandomGenerator)):
    """
    LegacyGenerator(brng=None)

    Container providing legacy generators.

    ``LegacyGenerator`` exposes a number of methods for generating random
    numbers for a set of distributions where the method used to produce random
    samples has changed. Three core generators have changed: normal,
    exponential and gamma. These have been replaced by faster Ziggurat-based
    methods in ``RadnomGenerator``. ``LegacyGenerator`` retains the slower
    methods to produce samples from these distributions as well as from
    distributions that depend on these such as the Chi-square, power or
    Weibull.

    **No Compatibility Guarantee**

    ``LegacyGenerator`` is evolving and so it isn't possible to provide a
    compatibility guarantee like NumPy does. In particular, better algorithms
    have already been added. This will change once ``RandomGenerator``
    stabilizes.

    Parameters
    ----------
    brng : Basic RNG, optional
        Basic RNG to use as the core generator. If none is provided, uses
        MT19937.


    Examples
    --------
    Exactly reproducing a NumPy stream requires using ``MT19937`` as
    the Basic RNG.

    >>> from randomgen import MT19937
    >>> lg = LegacyGenerator(MT19937(12345))
    >>> x = lg.standard_normal(10)
    >>> lg.shuffle(x)
    >>> x[0]
    0.09290787674371767
    >>> lg.standard_exponential()
    1.6465621229906502

    The equivalent commands from NumPy produce identical output.

    >>> from numpy.random import RandomState
    >>> rs = RandomState(12345)
    >>> x = rs.standard_normal(10)
    >>> rs.shuffle(x)
    >>> x[0]
    0.09290787674371767
    >>> rs.standard_exponential()
    1.6465621229906502
    """

    def __init__(self, brng=None):
        if brng is None:
            brng = MT19937()
        super(LegacyGenerator, self).__init__(brng)
        self.__legacy = _LegacyGenerator(brng)

    def __getattribute__(self, name):
        if name in _LEGACY_ATTRIBUTES:
            return self.__legacy.__getattribute__(name)
        return object.__getattribute__(self, name)

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        return (randomgen.pickle._experiment_ctor,
                (self.state['brng'],),
                self.state)
