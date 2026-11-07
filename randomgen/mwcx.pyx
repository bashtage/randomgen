#!python

import operator

import numpy as np

__all__ = ["MWC192X", "MWC128X", "MWC128X32", "MWC64X32"]

cdef uint64_t mwc192x_uint64(void* st) noexcept nogil:
    return mwc192x_next64(<mwc192x_state_t *>st)

cdef uint32_t mwc192x_uint32(void *st) noexcept nogil:
    return mwc192x_next32(<mwc192x_state_t *> st)

cdef double mwc192x_double(void* st) noexcept nogil:
    return uint64_to_double(mwc192x_next64(<mwc192x_state_t *>st))

cdef uint64_t mwc128x_uint64(void* st) noexcept nogil:
    return mwc128x_next64(<mwc128x_state_t *>st)

cdef uint32_t mwc128x_uint32(void *st) noexcept nogil:
    return mwc128x_next32(<mwc128x_state_t *> st)

cdef double mwc128x_double(void* st) noexcept nogil:
    return uint64_to_double(mwc128x_next64(<mwc128x_state_t *>st))

cdef uint64_t mwc128x32_uint64(void* st) noexcept nogil:
    return mwc128x32_next64(<mwc128x32_state_t *>st)

cdef uint32_t mwc128x32_uint32(void *st) noexcept nogil:
    return mwc128x32_next32(<mwc128x32_state_t *> st)

cdef double mwc128x32_double(void* st) noexcept nogil:
    return mwc128x32_next_double(<mwc128x32_state_t *>st)

cdef uint64_t mwc128x32_raw(void* st) noexcept nogil:
    return <uint64_t>mwc128x32_next32(<mwc128x32_state_t *> st)

cdef uint64_t mwc64x32_uint64(void* st) noexcept nogil:
    return mwc64x32_next64(<mwc64x32_state_t *>st)

cdef uint32_t mwc64x32_uint32(void *st) noexcept nogil:
    return mwc64x32_next32(<mwc64x32_state_t *> st)

cdef double mwc64x32_double(void* st) noexcept nogil:
    return mwc64x32_next_double(<mwc64x32_state_t *>st)

cdef uint64_t mwc64x32_raw(void* st) noexcept nogil:
    return <uint64_t>mwc64x32_next32(<mwc64x32_state_t *> st)


def _mwc_modulus(a, word_bits, lag):
    # m = a * b**lag - 1, where b = 2**word_bits
    return a * 2 ** (word_bits * lag) - 1


def _mwc_multiplier(a, word_bits, lag):
    # b**-1 mod m = a * b**(lag - 1), as a * b**lag = m + 1
    return a * 2 ** (word_bits * (lag - 1))


cdef class _MWCX(BitGenerator):
    """
    Base class for the MWCX generators

    Not part of the public API.

    The state is read as the integer s = x_0 + x_1 b + ... + x_{k-1} b^{k-1}
    + c b^k, where b = 2**_word_bits, k = _lag, x_0, ..., x_{k-1} are the
    digits (oldest first) and c is the carry. Each step multiplies s by
    _multiplier = b^-1 modulo the prime _modulus = a b^k - 1, so moving by n
    steps multiplies s by _multiplier**n mod _modulus.
    """

    def _get_words(self):
        raise NotImplementedError("Subclass must override")

    def _set_words(self, words):
        raise NotImplementedError("Subclass must override")

    def _state_int(self):
        words = self._get_words()
        return sum(
            int(words[i]) << (i * self._word_bits) for i in range(len(words))
        )

    def _set_state_int(self, s):
        mask = (1 << self._word_bits) - 1
        self._set_words(
            [(s >> (i * self._word_bits)) & mask for i in range(self._lag + 1)]
        )

    def _multiply_state(self, r):
        self._set_state_int((self._state_int() * r) % self._modulus)

    def _seed_from_seq(self):
        dtype = np.uint64 if self._word_bits == 64 else np.uint32
        state = self._get_seed_seq().generate_state(self._lag + 1, dtype)
        words = [int(v) for v in state]
        # As suggested by the reference code, the value used for the carry is
        # read as a fixed-point fraction in [0..1) and multiplied by a - 2,
        # the size of the range of valid carries
        words[self._lag] = 1 + (
            (words[self._lag] * (self._a - 2)) >> self._word_bits
        )
        self._set_words(words)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        This method is called at initialization. It can be called again to
        re-Seed the generator

        Parameters
        ----------
        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Entropy initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64), array of integers in
            [0, 2**64), a SeedSequence instance or ``None`` (the default).
            If `seed` is ``None``, then  data is read from ``/dev/urandom``
            (or the Windows analog) if available. If unavailable, a hash
            of the time and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed)

    def advance(self, delta):
        """
        advance(delta)

        Advance the underlying RNG as-if delta draws have occurred.

        Parameters
        ----------
        delta : int
            Number of draws to advance the RNG. Negative values move the RNG
            backward.

        Returns
        -------
        self : BitGenerator
            RNG advanced delta steps

        Notes
        -----
        Advancing takes time logarithmic in ``abs(delta)``. Each draw is one
        output of the generator (32 bits for ``MWC128X32`` and ``MWC64X32``).

        Advancing a RNG updates the underlying RNG state as-if a given
        number of calls to the underlying RNG have been made. In general
        there is not a one-to-one relationship between the number output
        random values from a particular distribution and the number of
        draws from the core RNG. This occurs for two reasons:

        * The random values are simulated using a rejection-based method
          and so, on average, more than one value from the underlying
          RNG is required to generate an single draw.
        * The number of bits required to generate a simulated value
          differs from the number of bits generated by the underlying
          RNG. For example, two 16-bit integer values can be simulated
          from a single draw of a 32-bit RNG.

        Advancing the RNG state resets any pre-computed random numbers.
        This is required to ensure exact reproducibility.
        """
        delta = operator.index(delta)
        r = pow(self._multiplier, delta, self._modulus)
        with self.lock:
            self._multiply_state(r)
        return self

    def jumped(self, iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned bit generator is jumped as-if
        ``iter * 2**e`` random numbers have been generated, where ``e`` is
        96 for ``MWC192X``, 64 for ``MWC128X`` and ``MWC128X32``, and 32
        for ``MWC64X32``.

        Parameters
        ----------
        iter : int
            Number of times to jump the state of the bit generator returned.
            Negative values jump backward.

        Returns
        -------
        bit_generator : BitGenerator
            New instance of generator jumped iter times

        Notes
        -----
        Jumping takes time logarithmic in ``abs(iter)``.
        """
        cdef _MWCX bit_generator

        iter = operator.index(iter)
        bit_generator = self.__class__(seed=self._copy_seed())
        bit_generator.state = self.state
        bit_generator._multiply_state(
            pow(self._jump_multiplier, iter, self._modulus)
        )
        return bit_generator

    @property
    def state(self):
        """
        Get or set the PRNG state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the PRNG
        """
        with self.lock:
            words = self._get_words()
            buffer = self._get_buffer()
        state = {"bit_generator": fully_qualified_name(self),
                 "state": dict(zip(self._names, words))}
        state.update(buffer)
        return state

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        words = [int(value["state"][name]) for name in self._names]
        for i in range(len(words)):
            if not 0 <= words[i] < 2 ** self._word_bits:
                raise ValueError(
                    f"{self._names[i]} must be in [0, 2**{self._word_bits})"
                )
        s = sum(words[i] << (i * self._word_bits) for i in range(len(words)))
        if not 0 < s < self._modulus:
            raise ValueError(
                "state is not valid: the integer formed by the digits and "
                "the carry must be positive and smaller than the modulus "
                f"{self._modulus:#x}"
            )
        with self.lock:
            self._set_words(words)
            self._set_buffer(value)

    def _get_buffer(self):
        return {}

    def _set_buffer(self, value):
        pass


cdef class MWC192X(_MWCX):
    """
    MWC192X(seed=None)

    Container for the MWC192X pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence.

    Notes
    -----
    MWC192X is a multiply-with-carry generator with 64-bit digits, lag two, and
    period approximately :math:`2^{191}`, whose output is scrambled by X. It is
    the suggested generator of the family for 64-bit outputs. Since it
    simulates a multiplicative congruential generator with prime modulus, it
    can be advanced by an arbitrary number of steps, also backward, in
    logarithmic time. See [1]_ for details.

    ``MWC192X`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``MWC192X`` state consists of two 64-bit digits and a 64-bit carry. It
    is seeded using a `SeedSequence`, which initializes the digits and the
    carry to valid values.

    **Parallel Features**

    ``MWC192X`` can be used in parallel applications by calling the method
    ``jumped`` which advances the state as-if :math:`2^{96}` random numbers
    have been generated, or ``advance``, which advances the state by an
    arbitrary number of steps. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be chained to ensure that the segments come from the
    same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import MWC192X
    >>> bit_gen = MWC192X(1234)
    >>> rg = []
    >>> for _ in range(10):
    ...    rg.append(Generator(bit_gen))
    ...    # Chain the BitGenerators
    ...    bit_gen = bit_gen.jumped()

    **Compatibility Guarantee**

    ``MWC192X`` makes a guarantee that a fixed seed will always produce the
    same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import MWC192X
    >>> rg = Generator(MWC192X(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Sebastiano Vigna, "Scrambled Multiply-with-Carry Pseudorandom
           Number Generators", arXiv:2610.27313, 2026.
           https://arxiv.org/abs/2610.27313
    """
    _a = 0xff4e24dd418f5433
    _word_bits = 64
    _lag = 2
    _names = ("x", "y", "c")
    _modulus = _mwc_modulus(0xff4e24dd418f5433, 64, 2)
    _multiplier = _mwc_multiplier(0xff4e24dd418f5433, 64, 2)
    _jump_multiplier = pow(_multiplier, 2**96, _modulus)

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &mwc192x_uint64
        self._bitgen.next_uint32 = &mwc192x_uint32
        self._bitgen.next_double = &mwc192x_double
        self._bitgen.next_raw = &mwc192x_uint64

    def _get_words(self):
        return [self.rng_state.x, self.rng_state.y, self.rng_state.c]

    def _set_words(self, words):
        self.rng_state.x = words[0]
        self.rng_state.y = words[1]
        self.rng_state.c = words[2]
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _get_buffer(self):
        return {"has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    def _set_buffer(self, value):
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]


cdef class MWC128X(_MWCX):
    """
    MWC128X(seed=None)

    Container for the MWC128X pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence.

    Notes
    -----
    MWC128X is a multiply-with-carry generator with 64-bit digits, lag one, and
    period approximately :math:`2^{127}`, whose output is scrambled by X. Since
    it simulates a multiplicative congruential generator with prime modulus, it
    can be advanced by an arbitrary number of steps, also backward, in
    logarithmic time. See [1]_ for details.

    ``MWC128X`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``MWC128X`` state consists of a 64-bit digit and a 64-bit carry. It is
    seeded using a `SeedSequence`, which initializes the digit and the carry to
    valid values.

    **Parallel Features**

    ``MWC128X`` can be used in parallel applications by calling the method
    ``jumped`` which advances the state as-if :math:`2^{64}` random numbers
    have been generated, or ``advance``, which advances the state by an
    arbitrary number of steps. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be chained to ensure that the segments come from the
    same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import MWC128X
    >>> bit_gen = MWC128X(1234)
    >>> rg = []
    >>> for _ in range(10):
    ...    rg.append(Generator(bit_gen))
    ...    # Chain the BitGenerators
    ...    bit_gen = bit_gen.jumped()

    **Compatibility Guarantee**

    ``MWC128X`` makes a guarantee that a fixed seed will always produce the
    same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import MWC128X
    >>> rg = Generator(MWC128X(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Sebastiano Vigna, "Scrambled Multiply-with-Carry Pseudorandom
           Number Generators", arXiv:2610.27313, 2026.
           https://arxiv.org/abs/2610.27313
    """
    _a = 0xff6da236274e252e
    _word_bits = 64
    _lag = 1
    _names = ("x", "c")
    _modulus = _mwc_modulus(0xff6da236274e252e, 64, 1)
    _multiplier = _mwc_multiplier(0xff6da236274e252e, 64, 1)
    _jump_multiplier = pow(_multiplier, 2**64, _modulus)

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &mwc128x_uint64
        self._bitgen.next_uint32 = &mwc128x_uint32
        self._bitgen.next_double = &mwc128x_double
        self._bitgen.next_raw = &mwc128x_uint64

    def _get_words(self):
        return [self.rng_state.x, self.rng_state.c]

    def _set_words(self, words):
        self.rng_state.x = words[0]
        self.rng_state.c = words[1]
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _get_buffer(self):
        return {"has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    def _set_buffer(self, value):
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]


cdef class MWC128X32(_MWCX):
    """
    MWC128X32(seed=None)

    Container for the MWC128X/32 pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence.

    Notes
    -----
    MWC128X/32 is a multiply-with-carry generator with 32-bit digits and
    output, lag three, and period approximately :math:`2^{127}`, whose output
    is scrambled by X. It is the suggested generator of the family for 32-bit
    outputs. Since it simulates a multiplicative congruential generator with
    prime modulus, it can be advanced by an arbitrary number of steps, also
    backward, in logarithmic time. See [1]_ for details.

    ``MWC128X32`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``MWC128X32`` state consists of three 32-bit digits and a 32-bit carry.
    It is seeded using a `SeedSequence`, which initializes the digits and the
    carry to valid values.

    **Parallel Features**

    ``MWC128X32`` can be used in parallel applications by calling the method
    ``jumped`` which advances the state as-if :math:`2^{64}` random numbers
    have been generated, or ``advance``, which advances the state by an
    arbitrary number of steps. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be chained to ensure that the segments come from the
    same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import MWC128X32
    >>> bit_gen = MWC128X32(1234)
    >>> rg = []
    >>> for _ in range(10):
    ...    rg.append(Generator(bit_gen))
    ...    # Chain the BitGenerators
    ...    bit_gen = bit_gen.jumped()

    **Compatibility Guarantee**

    ``MWC128X32`` makes a guarantee that a fixed seed will always produce the
    same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import MWC128X32
    >>> rg = Generator(MWC128X32(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Sebastiano Vigna, "Scrambled Multiply-with-Carry Pseudorandom
           Number Generators", arXiv:2610.27313, 2026.
           https://arxiv.org/abs/2610.27313
    """
    _a = 0xffcccc66
    _word_bits = 32
    _lag = 3
    _names = ("x", "y", "z", "c")
    _modulus = _mwc_modulus(0xffcccc66, 32, 3)
    _multiplier = _mwc_multiplier(0xffcccc66, 32, 3)
    _jump_multiplier = pow(_multiplier, 2**64, _modulus)

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &mwc128x32_uint64
        self._bitgen.next_uint32 = &mwc128x32_uint32
        self._bitgen.next_double = &mwc128x32_double
        self._bitgen.next_raw = &mwc128x32_raw

    def _get_words(self):
        return [self.rng_state.x, self.rng_state.y, self.rng_state.z,
                self.rng_state.c]

    def _set_words(self, words):
        self.rng_state.x = words[0]
        self.rng_state.y = words[1]
        self.rng_state.z = words[2]
        self.rng_state.c = words[3]


cdef class MWC64X32(_MWCX):
    """
    MWC64X32(seed=None)

    Container for the MWC64X/32 pseudo-random number generator.

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence.

    Notes
    -----
    MWC64X/32 is a multiply-with-carry generator with 32-bit digits and output,
    lag one, and period approximately :math:`2^{63}`, whose output is scrambled
    by X. It is meant for environments in which registers are scarce, such as
    GPUs: otherwise, use ``MWC128X32``. Since it simulates a multiplicative
    congruential generator with prime modulus, it can be advanced by an
    arbitrary number of steps, also backward, in logarithmic time. See [1]_ for
    details.

    ``MWC64X32`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``MWC64X32`` state consists of a 32-bit digit and a 32-bit carry. It is
    seeded using a `SeedSequence`, which initializes the digit and the carry to
    valid values.

    **Parallel Features**

    ``MWC64X32`` can be used in parallel applications by calling the method
    ``jumped`` which advances the state as-if :math:`2^{32}` random numbers
    have been generated, or ``advance``, which advances the state by an
    arbitrary number of steps. This allows the original sequence to be split
    so that distinct segments can be used in each worker process. All
    generators should be chained to ensure that the segments come from the
    same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import MWC64X32
    >>> bit_gen = MWC64X32(1234)
    >>> rg = []
    >>> for _ in range(10):
    ...    rg.append(Generator(bit_gen))
    ...    # Chain the BitGenerators
    ...    bit_gen = bit_gen.jumped()

    **Compatibility Guarantee**

    ``MWC64X32`` makes a guarantee that a fixed seed will always produce the
    same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import MWC64X32
    >>> rg = Generator(MWC64X32(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Sebastiano Vigna, "Scrambled Multiply-with-Carry Pseudorandom
           Number Generators", arXiv:2610.27313, 2026.
           https://arxiv.org/abs/2610.27313
    """
    _a = 0xfff145f8
    _word_bits = 32
    _lag = 1
    _names = ("x", "c")
    _modulus = _mwc_modulus(0xfff145f8, 32, 1)
    _multiplier = _mwc_multiplier(0xfff145f8, 32, 1)
    _jump_multiplier = pow(_multiplier, 2**32, _modulus)

    def __init__(self, seed=None):
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &mwc64x32_uint64
        self._bitgen.next_uint32 = &mwc64x32_uint32
        self._bitgen.next_double = &mwc64x32_double
        self._bitgen.next_raw = &mwc64x32_raw

    def _get_words(self):
        return [self.rng_state.x, self.rng_state.c]

    def _set_words(self, words):
        self.rng_state.x = words[0]
        self.rng_state.c = words[1]
