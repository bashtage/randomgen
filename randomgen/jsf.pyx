#!python


import numpy as np

from randomgen._deprecated_value import _DeprecatedValue

__all__ = ["JSF"]

INT_TYPES = (int, np.integer)
JSF_DEFAULTS = {64: {"p": 7, "q": 13, "r": 37},
                32: {"p": 27, "q": 17, "r": 0}}

JSF32_ALT_PARAMETERS = ((27, 17, 0),
                        (9, 16, 0),
                        (9, 24, 0),
                        (10, 16, 0),
                        (10, 24, 0),
                        (11, 16, 0),
                        (11, 24, 0),
                        (25, 8, 0),
                        (25, 16, 0),
                        (26, 8, 0),
                        (26, 16, 0),
                        (26, 17, 0),
                        (27, 16, 0),
                        (3, 14, 24),
                        (3, 25, 15),
                        (4, 15, 24),
                        (6, 16, 28),
                        (7, 16, 27),
                        (8, 14, 3),
                        (11, 16, 23),
                        (12, 16, 22),
                        (12, 17, 23),
                        (13, 16, 22),
                        (15, 25, 3),
                        (16, 9, 3),
                        (17, 9, 3),
                        (17, 27, 7),
                        (19, 7, 3),
                        (23, 15, 11),
                        (23, 16, 11),
                        (23, 17, 11),
                        (24, 3, 16),
                        (24, 4, 16),
                        (25, 14, 3),
                        (27, 16, 6),
                        (27, 16, 7))
JSF64_ALT_PARAMETERS = ((7, 13, 37),
                        (39, 11, 0))

JSF_PARAMETERS = {32: [], 64: []}
for p, q, r in JSF32_ALT_PARAMETERS:
    JSF_PARAMETERS[32].append({"p": p, "q": q, "r": r})
for p, q, r in JSF64_ALT_PARAMETERS:
    JSF_PARAMETERS[64].append({"p": p, "q": q, "r": r})

cdef uint64_t jsf64_uint64(void* st) noexcept nogil:
    return jsf64_next64(<jsf_state_t *>st)

cdef uint32_t jsf64_uint32(void *st) noexcept nogil:
    return jsf64_next32(<jsf_state_t *> st)

cdef double jsf64_double(void* st) noexcept nogil:
    return uint64_to_double(jsf64_next64(<jsf_state_t *>st))

cdef uint64_t jsf32_uint64(void* st) noexcept nogil:
    return jsf32_next64(<jsf_state_t *>st)

cdef uint32_t jsf32_uint32(void *st) noexcept nogil:
    return jsf32_next32(<jsf_state_t *> st)

cdef double jsf32_double(void* st) noexcept nogil:
    return uint64_to_double(jsf32_next64(<jsf_state_t *>st))

cdef uint64_t jsf32_raw(void* st) noexcept nogil:
    return <uint64_t>jsf32_next32(<jsf_state_t *>st)

cdef class JSF(BitGenerator):
    """
    JSF(seed=None, *, seed_size=1, size=64, p=None, q=None, r=None, mode="sequence")

    Container for Jenkins's Fast Small (JSF) pseudo-random number generator

    Parameters
    ----------
    seed : {None, int, array_like[uint], SeedSequence}, optional
        Random seed initializing the pseudo-random number generator. Can be
        an integer in [0, 2**size), an array of integers in
        [0, 2**size), a SeedSequence or ``None`` (the default). If
        `seed` is ``None``, then  data is read from ``/dev/urandom``
        (or the Windows analog) if available. If unavailable, a hash of
        the time and process ID is used.
    seed_size : {1, 2, 3}, optional
        Number of distinct seed values used to initialize JSF. The original
        implementation uses 1 (default). Higher values increase the size of
        the seed space which is ``2**(size*seed_size)``.
    size : {32, 64}, optional
        Output size of a single iteration of JSF. 32 is better suited to 32-bit
        systems.
    p : int, optional
        One the the three parameters that defines JSF. See Notes. If not
        provided uses the default values for the selected size listed in Notes.
    q : int, optional
        One the the three parameters that defines JSF. See Notes. If not
        provided uses the default values for the selected size listed in Notes.
    r : int, optional
        One the the three parameters that defines JSF. See Notes. If not
        provided uses the default values for the selected size listed in Notes.
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
    The JSF generator uses a 4-element state of unsigned integers, either
    uint32 or uint64, a, b, c and d ([1]_). The generator also depends on
    3 parameters p, q, and r that must be between 0 and size-1.

    The algorithm is defined by:

    .. code-block:: python

      e = a - ROTL(b, p)
      a = b ^ ROTL(c, q)
      b = c + ROTL(d, r)
      c = d + e
      d = e + a

    where d is the value returned at the end of a single iteration and
    ROTL(x, y) left rotates x by y bits.

    **Default Parameters**

    The defaults are

    =========== === ========= ==========
     Parameter          32       64
    =========== === ========= ==========
        p               27        7
        q               17       13
        r               0        37
    =========== === ========= ==========

    There are many other parameterizations. See the class attribute
    ``JSF.parameters`` for a list of the values provided by Jenkins. Note
    that if ``r`` is 0, the generator uses only 2 rotations.

    **State and Seeding**

    The state consists of the 4 values a, b, c and d. The size of these values
    depends on ``size``. The seed value is an unsided integer with the same
    size as the generator (e.g., uint64 for ``size==64``).

    **Compatibility Guarantee**

    ``JSF`` makes a guarantee that a fixed seed will always produce the same
    random integer stream.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import JSF
    >>> rg = Generator(JSF(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Jenkins, Bob (2009). "A small noncryptographic PRNG".
        https://burtleburtle.net/bob/rand/smallprng.html
    """
    parameters = JSF_PARAMETERS

    def __init__(self, seed=None, *, seed_size=1, size=64, p=None, q=None,
                 r=None, mode=_DeprecatedValue):
        BitGenerator.__init__(self, seed, mode=mode)
        if size not in (32, 64) or not isinstance(size, INT_TYPES):
            raise ValueError("size must be either 32 or 64")
        if seed_size not in (1, 2, 3) or not isinstance(seed_size, INT_TYPES):
            raise ValueError("seed size must be one of 1, 2, or 3")
        for val, val_name in ((p, "p"), (q, "q"), (r, "r")):
            if (
                    val is not None and
                    not (
                            0<= val <= size-1 and
                            isinstance(val, INT_TYPES)
                    )
            ):
                raise ValueError("{0} must be an integer between 0 and"
                                 "{1}".format(val_name, size-1))
        self.size = size
        self.seed_size = seed_size
        self._bitgen.state = <void *>&self.rng_state
        self.setup_generator(p, q, r)
        self.seed(seed)

    cdef setup_generator(self, object p, object q, object r):
        if self.size == 64:
            self._bitgen.next_uint64 = &jsf64_uint64
            self._bitgen.next_uint32 = &jsf64_uint32
            self._bitgen.next_double = &jsf64_double
            self._bitgen.next_raw = &jsf64_uint64
        else:
            self._bitgen.next_uint64 = &jsf32_uint64
            self._bitgen.next_uint32 = &jsf32_uint32
            self._bitgen.next_double = &jsf32_double
            self._bitgen.next_raw = &jsf32_raw
        self.rng_state.p = p if p is not None else JSF_DEFAULTS[self.size]["p"]
        self.rng_state.q = q if q is not None else JSF_DEFAULTS[self.size]["q"]
        self.rng_state.r = r if r is not None else JSF_DEFAULTS[self.size]["r"]

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        dtype = np.uint64 if self.size == 64 else np.uint32
        state = self._get_seed_seq().generate_state(self.seed_size, dtype)
        if self.size == 64:
            jsf64_seed(&self.rng_state,
                       <uint64_t*>np.PyArray_DATA(state),
                       self.seed_size)
        else:
            jsf32_seed(&self.rng_state,
                       <uint32_t*>np.PyArray_DATA(state),
                       self.seed_size)
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        This method is called at initialization. It can be called again to
        re-Seed the generator

        Parameters
        ----------
        seed : {None, int, array_like[uint], SeedSequence}, optional
            Random seed initializing the pseudo-random number generator. Can be
            an integer in [0, 2**size), an array of integers in
            [0, 2**size), a SeedSequence or ``None`` (the default). If
            `seed` is ``None``, then  data is read from ``/dev/urandom``
            (or the Windows analog) if available. If unavailable, a hash of
            the time and process ID is used.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        BitGenerator._seed_with_seed_sequence(self, seed)

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
        if self.size == 64:
            a = self.rng_state.a.u64
            b = self.rng_state.b.u64
            c = self.rng_state.c.u64
            d = self.rng_state.d.u64
        else:
            a = self.rng_state.a.u32
            b = self.rng_state.b.u32
            c = self.rng_state.c.u32
            d = self.rng_state.d.u32
        return {"bit_generator": fully_qualified_name(self),
                "state": {"a": a, "b": b, "c": c, "d": d,
                          "p": self.rng_state.p,
                          "q": self.rng_state.q,
                          "r": self.rng_state.r},
                "size": self.size,
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger,
                "seed_size": self.seed_size}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        self.size = value["size"]
        state = value["state"]
        self.setup_generator(state["p"], state["q"], state["r"])
        if self.size == 64:
            self.rng_state.a.u64 = state["a"]
            self.rng_state.b.u64 = state["b"]
            self.rng_state.c.u64 = state["c"]
            self.rng_state.d.u64 = state["d"]
        else:
            self.rng_state.a.u32 = state["a"]
            self.rng_state.b.u32 = state["b"]
            self.rng_state.c.u32 = state["c"]
            self.rng_state.d.u32 = state["d"]

        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
        self.seed_size = value["seed_size"]
