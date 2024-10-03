#!python

import numpy as np

__all__ = ["Romu"]

cdef uint64_t romuquad_uint64(void* st) noexcept nogil:
    return romuquad_next64(<romu_state_t *>st)

cdef uint32_t romuquad_uint32(void *st) noexcept nogil:
    return romuquad_next32(<romu_state_t *> st)

cdef double romuquad_double(void* st) noexcept nogil:
    return uint64_to_double(romuquad_next64(<romu_state_t *>st))

cdef uint64_t romutrio_uint64(void* st) noexcept nogil:
    return romutrio_next64(<romu_state_t *>st)

cdef uint32_t romutrio_uint32(void *st) noexcept nogil:
    return romutrio_next32(<romu_state_t *> st)

cdef double romutrio_double(void* st) noexcept nogil:
    return uint64_to_double(romutrio_next64(<romu_state_t *>st))

cdef class Romu(BitGenerator):
    """
    Romu(seed=None, variant="quad")

    Mark A. Overton's quad and trio rotate-multiply-based generators

    Parameters
    ----------
    seed : {None, int, array_like[ints], SeedSequence}, optional
        A seed to initialize the `BitGenerator`. If None, then fresh,
        unpredictable entropy will be pulled from the OS. If an ``int`` or
        ``array_like[ints]`` is passed, then it will be passed to
        `SeedSequence` to derive the initial `BitGenerator` state. One may also
        pass in a `SeedSequence` instance.
    variant : {"quad", "trio"}, default "quad"
        The variant to use. The Quad variant is somewhat slower but has a larger
        state.

    Notes
    -----
    ``Romu`` is a rotate-multiply based generator either either a 256-bit
    state (default, "quad") or a 192-bit state ("trio") ([1]_, [2]_). ``Romu``
    has a large capacity and has a tiny chance of overlap, especially when
    using the default "quad" variant. It is extremely fast.

    ``Romu`` provides a capsule containing function pointers that produce
    doubles, and unsigned 32 and 64- bit integers. These are not
    directly consumable in Python and must be consumed by a ``Generator``
    or similar object that supports low-level access.

    **State and Seeding**

    The ``Romu`` state vector consists of 3 ("trio") or 4 unsigned 64-bit
    values. The input seed is processed by `SeedSequence` to generate all
    values, then the ``Romu`` algorithm is iterated a small number of times to
    mix.

    **Compatibility Guarantee**

    ``Romu`` makes a guarantee that a fixed seed will always produce the same
    random integer stream.

    Examples
    --------
    ``Romu`` supports parallel application using distinct seed values.

    >>> from randomgen import SeedSequence, Romu
    >>> NUM_STREAMS = 8192
    >>> seed_seq = SeedSequence(489048146361948)
    >>> bit_gens = [Romu(child) for child in seed_seq.spawn(NUM_STREAMS)]
    >>> [bg.random_raw() for bg in bit_gens[:3]]
    [9130649916277841551, 2156737186088199787, 12803009197309261862]

    References
    ----------
    .. [1] Overton, M. A. Romu: Fast Nonlinear Pseudo-Random Number
       Generators Providing High Quality.
    .. [2] Overton, M. A. (2020, April 8). Fine random number generators.
       Retrieved June 26, 2020, from https://www.romu-random.org/
    """
    _seed_seq_len = 4
    _seed_seq_dtype = np.uint64

    def __repr__(self):
        out = object.__repr__(self)
        out = out.replace("Romu",
                          f"Romu(variant={self.variant})")
        return out

    def __init__(self, seed=None, variant="quad"):
        self.variant = self._check_variant(variant)
        BitGenerator.__init__(self, seed)
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._setup_bitgen()

    cdef _setup_bitgen(self):
        if self.variant == "quad":
            self._bitgen.next_uint64 = &romuquad_uint64
            self._bitgen.next_uint32 = &romuquad_uint32
            self._bitgen.next_double = &romuquad_double
            self._bitgen.next_raw = &romuquad_uint64
        else:
            self._bitgen.next_uint64 = &romutrio_uint64
            self._bitgen.next_uint32 = &romutrio_uint32
            self._bitgen.next_double = &romutrio_double
            self._bitgen.next_raw = &romutrio_uint64

    cdef _check_variant(self, variant):
        if not isinstance(variant, str):
            raise TypeError('variant must be either "quad" or "trio"')
        elif variant.lower() not in ("quad", "trio"):
            raise ValueError('variant must be either "quad" or "trio"')
        return variant.lower()

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def _seed_from_seq(self):
        cdef int quad = 1 if self.variant == "quad" else 0

        self._setup_bitgen()

        state = self._get_seed_seq().generate_state(4, np.uint64)
        if (state == 0).all():
            # Ensure at least one non-zero, exceedingly unlikely
            state[3] |= np.uint64(0x1)
        romu_seed(&self.rng_state, state[0], state[1], state[2], state[3], quad)
        self._reset_state_variables()

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator

        This method is called at initialization. It can be called again to
        re-Seed the generator

        seed : {None, int, array_like[ints], SeedSequence}, optional
            A seed to initialize the `BitGenerator`. If None, then fresh,
            unpredictable entropy will be pulled from the OS. If an ``int`` or
            ``array_like[ints]`` is passed, then it will be passed to
            `SeedSequence` to derive the initial `BitGenerator` state. One may also
            pass in a `SeedSequence` instance.

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
        return {"bit_generator": fully_qualified_name(self),
                "state": {"w": self.rng_state.w,
                          "x": self.rng_state.x,
                          "y": self.rng_state.y,
                          "z": self.rng_state.z,
                          "variant": self.variant
                          },
                "has_uint32": self.rng_state.has_uint32,
                "uinteger": self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        self.rng_state.w = value["state"]["w"]
        self.rng_state.x = value["state"]["x"]
        self.rng_state.y = value["state"]["y"]
        self.rng_state.z = value["state"]["z"]
        self.variant = self._check_variant(value["state"]["variant"])
        self.rng_state.has_uint32 = value["has_uint32"]
        self.rng_state.uinteger = value["uinteger"]
        self._setup_bitgen()
