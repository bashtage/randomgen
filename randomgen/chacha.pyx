#!python
import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["ChaCha"]

cdef uint64_t chacha_uint64(void* st) noexcept nogil:
    return chacha_next64(<chacha_state_t *>st)

cdef uint32_t chacha_uint32(void *st) noexcept nogil:
    return chacha_next32(<chacha_state_t *> st)

cdef double chacha_double(void* st) noexcept nogil:
    return chacha_next_double(<chacha_state_t *>st)

cdef class ChaCha(BitGenerator):
    """
    ChaCha(seed=None, *, counter=None, key=None, rounds=20, mode="sequence")

    Container for the ChaCha family of Counter pseudo-random number generators

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**256), an array of 4 uint64 values,
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and
        process ID is used.
    counter : {None, int, array_like[uint64]}, optional
        Counter to use in the ChaCha state. Can be either
        a Python int in [0, 2**128) or a 2-element uint64 array.
        If not provided, the counter is initialized at 0.
    key : {None, int, array_like[uint64]}, optional
        Key to use in the ChaCha state. Unlike seed, which is run through
        another RNG before use, the value in key is directly set. Can be either
        a Python int in [0, 2**256) or a 4-element uint64 array.
        key and seed cannot both be used.
    rounds : int, optional
        Number of rounds to run the ChaCha mixer. Must be an even integer.
        The standard number of rounds in 20. Smaller values, usually 8 or
        more, can be used to reduce security properties of the random stream
        while improving performance.
    mode : {None, "sequence"}
        Deprecated parameter. Do not use.

        .. deprecated: 2.0.0

           Starting in version 2, only seed sequences are supported.

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
    ChaCha is a 64-bit PRNG that uses a counter-based design based on
    the ChaCha cipher [1]_. Instances using different values
    of the key produce distinct sequences. ``ChaCha`` has a period
    of :math:`2^{128}` and supports arbitrary advancing and
    jumping the sequence in increments of :math:`2^{64}`. These features allow
    multiple non-overlapping sequences to be generated.

    ``ChaCha`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``AESCounter`` a related counter-based PRNG.

    **State and Seeding**

    The ``ChaCha`` state vector consists of a 16-element array of uint32
    that capture buffered draws from the distribution, an 8-element array of
    uint32s holding the seed, and an 2-element array of uint64 that holds the
    counter ([low, high]). The elements of the seed are the value provided by
    the user (or from the entropy pool). The final value rounds contains the
    number of rounds used. Typical values are  8, 12, or 20 (for high security).

    ``ChaCha`` is seeded using either a single 256-bit unsigned integer
    or a vector of 4 64-bit unsigned integers. In either case, the seed is
    used as an input for a second random number generator,
    SplitMix64, and the output of this PRNG function is used as the initial
    state. Using a single 64-bit value for the seed can only initialize a small
    range of the possible initial state values.

    **Parallel Features**

    ``ChaCha`` can be used in parallel applications by calling the ``jump``
    method  to advances the state as-if :math:`2^{64}` random numbers have
    been generated. Alternatively, ``advance`` can be used to advance the
    counter for any positive step in [0, 2**128). When using ``jump``, all
    generators should be initialized with the same seed to ensure that the
    segments come from the same sequence.

    >>> from numpy.random import Generator
    >>> from randomgen import ChaCha
    >>> rg = [Generator(ChaCha(1234)) for _ in range(10)]
    # Advance each ChaCha instances by i jumps
    >>> for i in range(10):
    ...     rg[i].bit_generator.jump(i)

    Alternatively, ``ChaCha`` can be used in parallel applications by using
    a sequence of distinct keys where each instance uses different key.

    >>> key = 2**93 + 2**65 + 2**33 + 2**17 + 2**9
    >>> rg = [Generator(ChaCha(key=key+i)) for i in range(10)]

    **Compatibility Guarantee**

    ``ChaCha`` makes a guarantee that a fixed seed and will always produce
    the same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import ChaCha
    >>> rg = Generator(ChaCha(1234, rounds=8))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Bernstein, D. J.. ChaCha, a variant of Salsa20.
         http://cr.yp.to/papers.html#chacha. 2008.01.28.
    """
    def __init__(
            self, seed=None, *, counter=None, key=None, rounds=20, mode=_DeprecatedValue
    ):
        BitGenerator.__init__(self, seed, mode=mode)
        self.rng_state = <chacha_state_t *>PyArray_malloc_aligned(
            sizeof(chacha_state_t)
        )
        if rounds % 2 != 0 or rounds <= 0:
            raise ValueError("rounds must be even and >= 2")
        self.rng_state.rounds = rounds
        self.seed(seed, counter, key)

        self._bitgen.state = <void *>self.rng_state
        self._bitgen.next_uint64 = &chacha_uint64
        self._bitgen.next_uint32 = &chacha_uint32
        self._bitgen.next_double = &chacha_double
        self._bitgen.next_raw = &chacha_uint64

    def __dealloc__(self):
        if self.rng_state:
            PyArray_free_aligned(self.rng_state)

    def _seed_from_seq(self, counter=None):
        state = self._get_seed_seq().generate_state(4, np.uint64)
        self.seed(key=state, counter=counter)

    @property
    def use_simd(self):
        """
        Toggle use of SIMD

        Parameters
        ----------
        flag : bool
            Flag indicating whether to use SIMD

        Returns
        -------
        flag : bool
            Current flag value

        Raises
        ------
        ValueError
            If SIMD is not supported
        """
        return RANDOMGEN_USE_SIMD

    @use_simd.setter
    def use_simd(self, value):
        capable = chacha_simd_capable()
        if value and not capable:
            raise ValueError("CPU does not support SIMD implementation")
        chacha_use_simd(bool(value))

    def seed(self, seed=None, counter=None, key=None):
        """
        seed(seed=None, counter=None, key=None)

        Seed the generator

        This method is called when ``ChaCha`` is initialized. It can be
        called again to re-Seed the generator For details, see
        ``ChaCha``.

        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**256), an array of 4 uint64 values,
            a SeedSequence instance or ``None`` (the default). If `seed` is
            ``None``, then  data is read from ``/dev/urandom`` (or the Windows
            analog) if available. If unavailable, a hash of the time and
            process ID is used.
        counter : {None, int, array_like[uint64]}, optional
            Counter to use in the ChaCha state. Can be either
            a Python int in [0, 2**128) or a 2-element uint64 array.
            If not provided, the counter is initialized at 0.
        key : {None, int, array_like[uint64]}, optional
            Key to use in the ChaCha state. Unlike seed, which is run
            through another RNG before use, the value in key is directly set.
            Can be either a Python int in [0, 2**256) or a 4-element uint64
            array. key and seed cannot both be used.

        Raises
        ------
        ValueError
            If values are out of range for the PRNG.

        Notes
        -----
        The two representation of the counter and key are related through
        array[i] = (value // 2**(64*i)) % 2**64.
        """
        if seed is not None and key is not None:
            raise ValueError("seed and key cannot be simultaneously used")
        if key is None:
            BitGenerator._seed_with_seed_sequence(self, seed, counter=counter)
            return

        key = object_to_int(key, 256, "key")
        counter = object_to_int(counter, 128, "counter")
        seed = int_to_array(key, "key", 256, 32)
        _seed = seed
        if _seed.dtype != np.uint64:
            _seed = view_little_endian(_seed, np.uint64)
        _stream = _seed[2:]
        counter = 0 if counter is None else counter
        _counter = int_to_array(counter, "counter", 128, 64)

        chacha_seed(self.rng_state,
                    <uint64_t *>np.PyArray_DATA(_seed),
                    <uint64_t *>np.PyArray_DATA(_stream),
                    <uint64_t *>np.PyArray_DATA(_counter))

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
        cdef int i
        block = np.empty(16, dtype=np.uint32)
        keysetup = np.empty(8, dtype=np.uint32)
        ctr = np.empty(2, dtype=np.uint64)
        for i in range(16):
            block[i] = self.rng_state.block[i]
        for i in range(8):
            keysetup[i] = self.rng_state.keysetup[i]
        for i in range(2):
            ctr[i] = self.rng_state.ctr[i]

        return {"bit_generator": fully_qualified_name(self),
                "state": {"block": block, "keysetup": keysetup, "ctr": ctr,
                          "rounds": self.rng_state.rounds}}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))

        state = value["state"]
        block = state["block"]
        for i in range(16):
            self.rng_state.block[i] = block[i]
        keysetup = state["keysetup"]
        for i in range(8):
            self.rng_state.keysetup[i] = keysetup[i]
        ctr = state["ctr"]
        for i in range(2):
            self.rng_state.ctr[i] = ctr[i]
        self.rng_state.rounds = state["rounds"]
