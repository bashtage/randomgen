#!python
# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
from numpy.random.bit_generator import ISeedSequence

from libc.stdint cimport intptr_t, uint8_t, uint32_t, uint64_t

from randomgen.common cimport BitGenerator

import numpy as np

cimport numpy as np

from numpy.random import SeedSequence


cdef class _TestSeninal:

    cdef bint testing

    def __init__(self, bint testing=0):
        self.testing=testing

    def get_testing(self):
        return self.testing

    def set_testing(self, bint value):
        self.testing = value

_test_sentinal = _TestSeninal()

# Module level variables to save small allocations
cdef np.ndarray WORDS_ARR = np.arange(16, dtype=np.uint8)
cdef uint8_t[::1] WORDS = WORDS_ARR


cdef void _reset_words():
    cdef int i
    for i in range(16):
        WORDS[i] = i


def _get_words():
    return np.asarray(WORDS)


cdef uint64_t squares_uint64(void* st) noexcept nogil:
    return squares_next64(<squares_state_t *>st)

cdef uint32_t squares_uint32(void *st) noexcept nogil:
    return squares_next32(<squares_state_t *> st)

cdef double squares_double(void* st) noexcept nogil:
    return squares_next_double(<squares_state_t *>st)

cdef uint64_t squares_32_uint64(void* st) noexcept nogil:
    return squares_32_next64(<squares_state_t *>st)

cdef uint32_t squares_32_uint32(void *st) noexcept nogil:
    return squares_32_next32(<squares_state_t *> st)

cdef double squares_32_double(void* st) noexcept nogil:
    return squares_32_next_double(<squares_state_t *>st)


cdef class _SequenceSampler:
    """
    Sampler that supports a limited number of transofmration
    using the state geneaated by a SeedSequence.
    """
    cdef int randoms_drawm, bit_well_loc, index
    cdef uint64_t[::1] random_well
    cdef uint64_t bit_well
    cdef object seed_seq

    def __init__(self, seed_seq: SeedSequence, int initial_draw=10):
        self.seed_seq = seed_seq
        self.randoms_drawm = initial_draw
        self.random_well = self.seed_seq.generate_state(
            self.randoms_drawm, dtype=np.uint64
        )
        self.bit_well = 0
        self.bit_well_loc = 0
        self.index = 0
        self.refill()

    cdef void refill(self):
        """Refill the bit well after it is exhausted"""
        self.bit_well = <uint64_t>self.random_well[self.index]
        self.bit_well_loc = 0
        self.index += 1
        if self.index > self.randoms_drawm:
            self.randoms_drawm *= 2
            self.random_well = self.seed_seq.generate_state(
                self.randoms_drawm, dtype=np.uint64
            )

    cdef uint64_t gen_bits(self, int nbits):
        """Generate a fixed number of bits from the bit well"""
        cdef uint64_t out

        if self.bit_well_loc + nbits > 64:
            self.refill()
        out = self.bit_well & ((1 << nbits) - 1)
        self.bit_well >>= nbits
        self.bit_well_loc += nbits
        return out

    cdef uint64_t random_inverval(self, uint64_t max_value):
        """
        Generate random intervales with max values between 1 and 16.

        No error checking is performed if max_value > 16."""
        cdef int bits
        cdef uint64_t out

        if max_value == 1:
            return 0
        elif max_value == 2:
            return self.gen_bits(1)
        elif max_value < 4:
            bits = 2
        elif max_value < 8:
            bits = 3
        else:
            bits = 4
        while True:
            out = self.gen_bits(bits)
            if out < max_value:
                return out

    cdef void shuffle(self, uint8_t[::1] array):
        """
        Shuffle an array in place. Maximum size is
        16 due to the dependence on random_interval.
        """
        cdef intptr_t i, n
        cdef uint64_t j
        n = array.shape[0]
        for i in range(n - 1):
            j = i + self.random_inverval(n - i)
            tmp = array[i]
            array[i] = array[j]
            array[j] = tmp

    cdef uint8_t next_word(self):
        """Generate 4 bits (1 word)"""
        return <uint8_t>self.gen_bits(4)


cdef uint64_t generate_key(_SequenceSampler seq_sampler):
    """
    The core key generation implementation
    """
    cdef uint64_t out, i
    cdef uint8_t[::1] tmp = WORDS
    cdef uint8_t last_value, next_value, old

    out = 0
    _reset_words()
    seq_sampler.shuffle(tmp)
    if not (tmp[0] & 0x1):
        old = tmp[0]
        tmp[0] += 1
        for i in range(1, 8):
            if tmp[i] == tmp[0]:
                tmp[i] = old
    for i in range(8):
        out |= <uint64_t> tmp[i] << (4 * i)
    last_value = tmp[7]
    # 1. Choose a random byte from 1, 3, ..., 15, call a
    # 2. Store in byte 0
    # 3. Populate an array with bytes 1...15 ex a
    # 4. Shuffle the array
    # 5. Store in Bytes 1...7 with first 7 locations form shuffled array
    # 7. Choose a random number from 1,2,...,15 ex position 7
    # 8. Store in byte 8
    # 9. Populate bytes 9...15 with random bytes
    #    using the rule that byte[i] must be different from byte[i-1]
    #    and byte[i] != 0
    for i in range(8, 16):
        next_value = 0
        while next_value == last_value or next_value == 0:
            next_value = seq_sampler.next_word()
        last_value = next_value
        out |= <uint64_t>next_value << (4 * i)

    return out


def generate_keys(seed=None, intptr_t n=1, bint unique=False):
    r"""
    generate_keys(seed=None, n=1, unique=False)

    Pre-generate keys for use with Squares

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Entropy initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64), array of integers in [0, 2**64),
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available.
    n : int, optional
        Number of keys to generate. Must be a positive integer. Default is 1.
    unique : bool, optional
        If True, return only unique keys. Default is False.

    Returns
    -------
    keys : ndarray
        Array of generated keys.

    Notes
    -----
    The keys are generated randomly using the entropy generated by a
    SeedSequence (which is either the value of ``seed`` or a SeedSequence
    created using ``seed``).

    Key are generated according to the following rules [1]_:

    - Word 0: Odd number between 1 and 15.
    - Words 1-7: Randomly selected from 1 to 15. Each word can appear at
      most once (including word 0).
    - Words 9 - 15: Randomly selected from 1 to 15. Word n is required to
      be different from word n-1.

    There is no guarantee that the keys returns are distinct unless ``unique``
    is True. Using unqiue keys can be expensive as it requires calling
    np.unique on the generated keys, and then regenerating keys additional
    keys if there are incufficient unique keys. The number of distinct keys
    is

    .. math::

       8 \cdot 14_p 7 \cdot 14^8 = 204,217,092,180,541,440

    where :math:`14_p 7` is the number of permutations of 7 elements from 14.
    The chance of observing at least one repeated key in a subsample of 25,000
    would be approximately 1 in 8,168,683,687,202.

    Examples
    --------
    >>> from numpy.random import SeedSequence
    >>> from randomgen.squares import generate_keys
    >>> ss = SeedSequence(1234)
    >>> keys = generate_keys(ss, 5)
    >>> keys
    array([ 9470186258571876535, 11789540394216366135, 11013866738698308655,
        7246136968226125645,  1784984383128236737], dtype=uint64)

    >>> for k in keys: print(hex(k))
    0x836cdc3a1af658b7
    0xa39cdc865e62a037
    0x98d91d71e39a4c2f
    0x648f732e50f1b74d
    0x18c589d6d5fa72c1

    See Also
    --------
    randomgen.squares.Squares
       The PRNG that uses the keys generated by this function.

    References
    ----------
    .. [1] Widynski, Bernard. Middle-Square Weyl Sequence RNG.
           arXiv:1704.00358. 2017. https://doi.org/10.48550/arXiv.2004.06278.
    """
    cdef intptr_t i, nunique, iter, start = 0
    cdef uint64_t[::1] out, tmp, index
    cdef np.ndarray out_arr, index_arr
    cdef _TestSeninal _internal_senintal = _test_sentinal
    cdef _SequenceSampler seq_sampler
    cdef bint incomplete = True

    if isinstance(seed, ISeedSequence):
        seed_seq = seed
    else:
        seed_seq = SeedSequence(seed)
    if n < 1:
        raise ValueError("n must be a positive integer")
    seq_sampler = _SequenceSampler(seed_seq, 10 * n)
    out_arr = np.empty(n, dtype=np.uint64)
    out = out_arr
    iter = 1
    while incomplete:
        for i in range(start, n):
            out[i] = generate_key(seq_sampler)
        if not unique:
            break
        # Ensure unique
        _, index_arr = np.unique(out_arr, return_index=True)
        index_arr.sort()
        index = index_arr = index_arr.astype(np.uint64)
        nunique = index_arr.shape[0]
        if  iter < 2 and _internal_senintal.testing:
            nunique = nunique - max(min(10, nunique // 2), 1)

        if nunique == n:
            incomplete = False
        else:
            tmp = out_arr = np.empty(n, dtype=np.uint64)
            for i in range(nunique):

                tmp[i] = out[index[i]]
            out = tmp
            start = nunique
        iter += 1

    return np.asarray(out)


cdef class Squares(BitGenerator):
    """
    Squares(seed=None, counter=None, key=None, variant=64)

    Squares counter-based PRNG

    Squares is a counter-based PRNG that uses a key to generate random
    numbers using the middle-square random number generator [1]_. The key is
    carefully generated using a SeedSequence to satisfy certain
    characteristics regarding bit density.

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Entropy initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64), array of integers in [0, 2**64),
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available.
    counter : {None, int}, optional
        The initial counter to use when constructing the PRNG.
        The defalt value is 0.
    key : {None, int}, optional
        The key to use when constructing the PRNG. If None, the key is
        generated using the seeded SeedSequence. Default is None. Setting
        this value will override the key generated by the SeedSequence.
    variant : {32, 64}, optional
        The variance of the Square to use. Default is 64.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : {None, SeedSequence}
        The SeedSequence instance used to initialize the generator if mode is
        "sequence" or is seed is a SeedSequence.

    Notes
    -----
    ``Squares`` [2]_ is a pseudo-random number generator based on the Squares
    PRNG. It comes in both 64-bit (default) and 32-bit variants. It uses a
    64-bit key and a counter. Each draw transforms the current counter using
    a number of middle-square operations and the key. The key is constructed
    using 3 rues:

    - The first word (16 bits) ia selected from the set {1, 3, 5, ...
      B, D, F}, and so is always odd.
    - The next 7 words are selected without repetition from the set
      {1, 2, 3, ..., D, E, F}, excluding the word selected in the 1st
      position.
    - The remaining 8 words are selected at random (with replacement)
      from the set {1, 2, 3, ..., D, E, F} with the run that the word
      in position i must differ from the work in position j.

    The ``SeedSequence`` used in the initialization of the bit generator is
    used as the source of randomness for

    **State and Seeding**

    The ``Squares`` state consists of a 64-bit unsigned integer key
    and a 64-bit unsigned integer counter.
    By default, the ``seed`` value is translated into the 64-bit
    unsigned integer. If ``counter`` is None, the PRNG starts with a
    counter of 0..

    **Compatibility Guarantee**

    ``Squares`` makes a guarantee that a fixed seed will always
    produce the same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import Squares
    >>> rg = Generator(Squares(1234))
    >>> rg.standard_normal()
    0.123  # random

    **Parallel Features**

    ``Squares`` can be used in parallel when combined with a ``SeedSequence``
    using ``spawn``.

    >>> from randomgen import SeedSequence
    >>> entropy = 8509285875904376097169743623867
    >>> ss = SeedSequence(entropy)
    >>> bit_gens = [Squares(child) for child in ss.spawn(1024)]

    An alternative generates a set of keys from a single seed.

    >>> from randomgen.squares import generate_keys
    >>> keys = generate_keys(1234, 1024)
    >>> bit_gens = [Squares(key=key) for key in keys]

    The final options uses the same ``seed`` value along with different
    ``counter`` values.

    >>> from randomgen import SeedSequence
    >>> bit_gens = []
    >>> for i in range(1024):
    ...     bit_gens.append(Squares(SeedSequence(entropy), counter=1_000_000_000 * i))

    See also
    --------
    randomgen.squares.generate_keys
        Key generation function for the Squares PRNG with additional
        details on key requirements.

    References
    ----------
    .. [1] "Middle-square method", Wikipedia,
           https://en.wikipedia.org/wiki/Middle-square_method
    .. [2] Widynski, Bernard. Middle-Square Weyl Sequence RNG.
           arXiv:1704.00358. 2017.
           https://doi.org/10.48550/arXiv.2004.06278
    """
    def __init__(self, seed=None, counter=None, key=None, variant=64):
        BitGenerator.__init__(self, seed)
        if counter is None:
            self.rng_state.counter = 0
        else:
            self.rng_state.counter = self._check_value(counter, "counter", False)
        if key is not None:
            self.rng_state.key =self._check_value(key, "key", True)
        if variant not in (64, 32):
            raise ValueError("variant must be either 64 or 32.")
        self.variant = variant
        self._use64 = variant == 64
        self.seed(seed, counter=counter, key=key)
        self._bitgen.state = <void *> &self.rng_state
        self._setup_bitgen()

    cdef void _setup_bitgen(self):
        if self._use64:
            self._bitgen.next_uint64 = &squares_uint64
            self._bitgen.next_uint32 = &squares_uint32
            self._bitgen.next_double = &squares_double
            self._bitgen.next_raw = &squares_uint64
        else:
            self._bitgen.next_uint64 = &squares_32_uint64
            self._bitgen.next_uint32 = &squares_32_uint32
            self._bitgen.next_double = &squares_32_double
            self._bitgen.next_raw = &squares_32_uint64

    @staticmethod
    cdef uint64_t _check_value(self, object val, object name, bint odd):
        val = int(val)
        if not 0 <= val <= np.iinfo(np.uint64).max:
            raise ValueError(f"{name} must be positive and less than 2**64.")

        if odd and not (val & 0x1):
            raise ValueError(f"{name} must be odd.")

        return <uint64_t>val

    def _seed_from_seq(self, uint64_t counter=0):
        seed_seq = self._get_seed_seq()
        self.rng_state.key = generate_key(_SequenceSampler(seed_seq))
        self.rng_state.counter = counter

    cdef void _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def seed(self, seed=None, *, counter=None, key=None):
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

        if counter is not None:
            self.rng_state.counter = self._check_value(counter, "counter", False)
        else:
            self.rng_state.counter = counter = 0
        if key is not None:
            self.rng_state.key = self._check_value(key, "key", True)
        else:
            BitGenerator._seed_with_seed_sequence(self, seed, counter=counter)
        self._reset_state_variables()

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
        return {"bit_generator": fully_qualified_name(self),
                "state": {"key": self.rng_state.key, "counter": self.rng_state.counter},
                "variant": self.variant,
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger,
                }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state = value["state"]
        self.rng_state.key = state["key"]
        self.rng_state.counter = state["counter"]
        variant = value["variant"]
        if variant not in (64, 32):
            raise ValueError("variant must be either 64 or 32.")
        self.variant = variant
        self._use64 = variant == 64
        self._setup_bitgen()

    def advance(self, uint64_t delta):
        """
        advance(delta)

        Advance the state of the PRNG

        Parameters
        ----------
        delta : int
            The number of steps to advance the PRNG. Must be an integer
            in the range [0, 2**64).

        Returns
        -------
        self : Squares
            The PRNG advanced by delta steps.
        """
        self.rng_state.counter += delta
        self._reset_state_variables()
        return self

    def jumped(self, intptr_t iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator with the state jumped

        The state of the returned big generator is jumped as-if
        2**(64 * iter) random numbers have been generated.

        Parameters
        ----------
        iter : integer, positive
            Number of times to jump the state of the bit generator returned

        Returns
        -------
        bit_generator : Squares
            New instance of generator jumped iter times

        Notes
        -----
        The step size is phi (the Golden Ratio) when divided by 2**64.
        """
        cdef Squares bit_generator
        # phi * 2**64 = 11400714819323198485
        cdef uint64_t step = 11400714819323198485

        if iter <= 0:
            raise ValueError("iter must be a positive integer")

        bit_generator = self.__class__(seed=self._copy_seed(), variant=self.variant)
        bit_generator.state = self.state
        bit_generator._reset_state_variables()
        for i in range(iter):
            bit_generator.rng_state.counter += step

        return bit_generator
