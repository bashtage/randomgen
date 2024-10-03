#!python
# coding=utf-8
import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["HC128"]

cdef uint64_t hc128_uint64(void* st) noexcept nogil:
    return hc128_next64(<hc128_state_t *>st)

cdef uint32_t hc128_uint32(void *st) noexcept nogil:
    return hc128_next32(<hc128_state_t *> st)

cdef double hc128_double(void* st) noexcept nogil:
    return hc128_next_double(<hc128_state_t *> st)


cdef class HC128(BitGenerator):
    """
    HC128(seed=None, *, key=None, mode="sequence")

    Container for the HC-128 cipher-based pseudo-random number generator

    Parameters
    ----------
    seed : {None, int, array_like[uint64], SeedSequence}, optional
        Random seed initializing the pseudo-random number generator.
        Can be an integer in [0, 2**64), array of integers in [0, 2**64),
        a SeedSequence instance or ``None`` (the default). If `seed` is
        ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available. If unavailable, a hash of the time and process
        ID is used.
    key : {int, array_like[uint64]}, optional
        Key for HC128. The key is a 256-bit integer that contains both the
        key (lower 128 bits) and initial values (upper 128-bits) for the
        HC-128 cipher. key and seed cannot both be used.
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
    HC-128 was developer by Hongjun Wu and is an eSTREAM finalist ([1]_, [2]_).
    It is a cryptographic quality random number generator that produces a
    keystream is suitable for encryption. The average cycle length is expected
    to be 2**(1024*32+10-1) = 2**32777. ``HC128`` is the fastest software-only
    encryption-quality bit generator, and about 50% as fast as ``AESCounter``
    when used with AESNI.

    ``HC128`` provides a capsule containing function pointers that
    produce doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    See ``AESCounter`` and ``ChaCha`` for related PRNGs that uses a block
    cipher rather than a stream cipher.

    **State and Seeding**

    The ``HC128`` state vector consists of a 2 512-element arrays of 32-bit
    unsigned integers (p and q) and an integer in [0, 1024). In addition, the
    state contains a 16-element array that buffers values and a buffer index.

    ``HC128`` is seeded using either a single 256-bit unsigned
    integer or a vector of 64-bit unsigned integers. In either case, the seed
    is used as an input for another simple random number generator, SplitMix64,
    and the output of this PRNG function is used as the initial state.
    Alternatively, the key can be set directly using a 256-bit integer.

    **Parallel Features**

    ``HC128`` can be used in parallel applications by using distinct keys

    >>> from numpy.random import Generator
    >>> from randomgen import HC128
    >>> rg = [Generator(HC128(key=1234 + i)) for i in range(10)]

    **Compatibility Guarantee**

    ``HC128`` makes a guarantee that a fixed seed will always produce the
    same random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import HC128
    >>> rg = Generator(HC128(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Wu, Hongjun (2008). "The Stream Cipher HC-128."
        http://www.ecrypt.eu.org/stream/p3ciphers/hc/hc128_p3.pdf.
        *The eSTREAM Finalists*, LNCS 4986, pp. 39–47, Springer-Verlag.
    .. [2] Wu, Hongjun, "Stream Ciphers HC-128 and HC-256".
        https://www.ntu.edu.sg/home/wuhj/research/hc/index.html)
    """
    def __init__(self, seed=None, *, key=None, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, mode=mode)
        self.seed(seed, key)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &hc128_uint64
        self._bitgen.next_uint32 = &hc128_uint32
        self._bitgen.next_double = &hc128_double
        self._bitgen.next_raw = &hc128_uint64

    def _seed_from_seq(self):
        state = self._get_seed_seq().generate_state(4, np.uint64)
        self.seed(key=state)

    def seed(self, seed=None, key=None):
        """
        seed(seed=None, key=None)

        Seed the generator

        This method is called at initialization. It can be called again to
        re-Seed the generator

        Parameters
        ----------
        seed : {None, int, array_like[uint64], SeedSequence}, optional
            Random seed initializing the pseudo-random number generator.
            Can be an integer in [0, 2**64), array of integers in
            [0, 2**64), a SeedSequence instance or ``None`` (the default).
            If `seed` is ``None``, then  data is read from ``/dev/urandom``
            (or the Windows analog) if available. If unavailable, a hash of
            the time and process ID is used.
        key : {int, array_like[uint64]}, optional
            Key for HC128. The key is a 256-bit integer that contains both the
            key (lower 128 bits) and initial values (upper 128-bits) for the
            HC-128 cipher. key and seed cannot both be used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        if seed is not None and key is not None:
            raise ValueError("seed and key cannot be simultaneously used")
        if key is None:
            BitGenerator._seed_with_seed_sequence(self, seed)
            return

        key = object_to_int(key, 256, "key")
        state = int_to_array(key, "key", 256, 64)
        # Ensure state uint32 values are the same in LE and BE and in the same order
        state = view_little_endian(state, np.uint32)
        hc128_seed(&self.rng_state, <uint32_t *>np.PyArray_DATA(state))

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
        cdef uint32_t *p_arr
        cdef uint32_t *q_arr
        cdef uint32_t *buf_arr

        p = np.empty(512, dtype=np.uint32)
        q = np.empty(512, dtype=np.uint32)
        buffer = np.empty(16, dtype=np.uint32)
        p_arr = <uint32_t *>np.PyArray_DATA(p)
        q_arr = <uint32_t *>np.PyArray_DATA(q)
        for i in range(512):
            p_arr[i] = self.rng_state.p[i]
            q_arr[i] = self.rng_state.q[i]
        buf_arr = <uint32_t *>np.PyArray_DATA(buffer)
        for i in range(16):
            buf_arr[i] = self.rng_state.buffer[i]
        return {"bit_generator": fully_qualified_name(self),
                "state": {"p": p,
                          "q": q,
                          "hc_idx": self.rng_state.hc_idx,
                          "buffer": buffer,
                          "buffer_idx": self.rng_state.buffer_idx},
                }

    @state.setter
    def state(self, value):
        cdef int i
        cdef uint32_t *p_arr
        cdef uint32_t *q_arr
        cdef uint32_t *buf_arr

        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        state = value["state"]
        p = check_state_array(state["p"], 512, 32, "p")
        q = check_state_array(state["q"], 512, 32, "q")
        buffer = check_state_array(state["buffer"], 16, 32, "buffer")
        p_arr = <uint32_t *>np.PyArray_DATA(p)
        q_arr = <uint32_t *>np.PyArray_DATA(q)
        for i in range(512):
            self.rng_state.p[i] = p_arr[i]
            self.rng_state.q[i] = q_arr[i]
        buf_arr = <uint32_t *>np.PyArray_DATA(buffer)
        for i in range(16):
            self.rng_state.buffer[i] = buf_arr[i]
        self.rng_state.hc_idx = state["hc_idx"]
        self.rng_state.buffer_idx = state["buffer_idx"]
