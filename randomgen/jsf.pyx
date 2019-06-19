try:
    from threading import Lock
except ImportError:
    from dummy_threading import Lock

from cpython.pycapsule cimport PyCapsule_New

import numpy as np
cimport numpy as np

from randomgen.common cimport *
from randomgen.distributions cimport bitgen_t
from randomgen.entropy import random_entropy, seed_by_array

__all__ = ['JSF']

np.import_array()

JSF_DEFAULTS = {64:{'p': 7,'q': 13,'r': 37},
                32:{'p':27, 'q':17, 'r': 0}}

JSF32_ALT_PARAMETERS = ((27,17,0),
                        (9,16, 0),
                        (9,24, 0),
                        (10,16, 0),
                        (10,24, 0),
                        (11,16, 0),
                        (11,24, 0),
                        (25,8, 0),
                        (25,16, 0),
                        (26,8, 0),
                        (26,16, 0),
                        (26,17, 0),
                        (27,16, 0),
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

JSF_PARAMETERS = {32:[], 64: []}
for p,q,r in JSF32_ALT_PARAMETERS:
    JSF_PARAMETERS[32].append({'p': p, 'q': q, 'r': r})
for p,q,r in JSF64_ALT_PARAMETERS:
    JSF_PARAMETERS[64].append({'p': p, 'q': q, 'r': r})

cdef extern from "src/jsf/jsf.h":

    union JSF_UINT_T:
        uint64_t u64
        uint32_t u32
    ctypedef JSF_UINT_T jsf_uint_t

    struct JSF_STATE_T:
        jsf_uint_t a
        jsf_uint_t b
        jsf_uint_t c
        jsf_uint_t d
        int p
        int q
        int r
        int has_uint32
        uint32_t uinteger

    ctypedef JSF_STATE_T jsf_state_t

    uint64_t jsf64_next64(jsf_state_t *state) nogil
    uint32_t jsf64_next32(jsf_state_t *state) nogil
    double jsf64_next_double(jsf_state_t *state) nogil
    void jsf64_seed(jsf_state_t *state, uint64_t seed)

    uint64_t jsf32_next64(jsf_state_t *state) nogil
    uint32_t jsf32_next32(jsf_state_t *state) nogil
    double jsf32_next_double(jsf_state_t *state) nogil
    void jsf32_seed(jsf_state_t *state, uint32_t seed)



cdef uint64_t jsf64_uint64(void* st) nogil:
    return jsf64_next64(<jsf_state_t *>st)

cdef uint32_t jsf64_uint32(void *st) nogil:
    return jsf64_next32(<jsf_state_t *> st)

cdef double jsf64_double(void* st) nogil:
    return uint64_to_double(jsf64_next64(<jsf_state_t *>st))

cdef uint64_t jsf32_uint64(void* st) nogil:
    return jsf32_next64(<jsf_state_t *>st)

cdef uint32_t jsf32_uint32(void *st) nogil:
    return jsf32_next32(<jsf_state_t *> st)

cdef double jsf32_double(void* st) nogil:
    return uint64_to_double(jsf32_next64(<jsf_state_t *>st))

cdef uint64_t jsf32_raw(void* st) nogil:
    return <uint64_t>jsf32_next32(<jsf_state_t *>st)

cdef class JSF:
    """
    JSF(seed=None, size=64, p=None, q=None, r=None)

    Container for Jenkins's Fast Small (JSF) pseudo-random number generator

    Parameters
    ----------
    seed : {None, int, array_like}, optional
        Random seed initializing the pseudo-random number generator. Can be
        an integer in [0, 2**size-1] or ``None`` (the default). If `seed`
        is ``None``, then  data is read from ``/dev/urandom`` (or the Windows
        analog) if available.  If unavailable, a hash of the time and process
        ID is used.
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

    Attributes
    ----------
    lock: threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.

    Notes
    -----
    The JSF generator uses a 4-element state of unsigned integers, either
    uint32 or uint64, a, b, c and d ([1]_). The generator also depends on
    3 parameters p, q, and r that must be between 0 and size-1.

    The algorithm used satisfies:

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
    The state consists of the 4 values a, b, c and d.  The size of these values
    depends on ``size``. The seed value is an unsided integer with the same
    size as the generator (e.g., uint64 for ``size==64``).

    **Compatibility Guarantee**

    ``JSF`` makes a guarantee that a fixed seed will always produce the same
    random integer stream.

    Examples
    --------
    >>> from randomgen import Generator, JSF
    >>> rg = Generator(JSF(1234))
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Jenkins, Bob (2009). "A small noncryptographic PRNG"
    """
    cdef jsf_state_t rng_state
    cdef bitgen_t _bitgen
    cdef public object capsule
    cdef object _ctypes
    cdef object _cffi
    cdef public object lock
    cdef int size
    parameters = JSF_PARAMETERS

    def __init__(self, seed=None, size=64, p=None, q=None, r=None):
        if size not in (32, 64):
            raise ValueError('size must be either 32 or 64')
        for val, val_name in ((p,'p'), (q,'q'), (r,'r')):
            if val is not None and not 0<= val <= size-1:
                raise ValueError('{0} must be between 0 and'
                                 '{1}'.format(val_name, size-1))
        self.size = size
        self._bitgen.state = <void *>&self.rng_state
        self.setup_generator(p, q, r)
        self.seed(seed)
        self.lock = Lock()

        self._ctypes = None
        self._cffi = None

        cdef const char *name = "BitGenerator"
        self.capsule = PyCapsule_New(<void *>&self._bitgen, name, NULL)

    cdef setup_generator(self, p, q, r):
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
        self.rng_state.p = p if p is not None else JSF_DEFAULTS[self.size]['p']
        self.rng_state.q = q if q is not None else JSF_DEFAULTS[self.size]['q']
        self.rng_state.r = r if r is not None else JSF_DEFAULTS[self.size]['r']

    # Pickling support:
    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state

    def __reduce__(self):
        from randomgen._pickle import __bit_generator_ctor
        return __bit_generator_ctor, (self.state['bit_generator'],), self.state

    cdef _reset_state_variables(self):
        self.rng_state.has_uint32 = 0
        self.rng_state.uinteger = 0

    def random_raw(self, size=None, output=True):
        """
        random_raw(self, size=None)

        Return randoms as generated by the underlying BitGenerator

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn.  Default is None, in which case a
            single value is returned.
        output : bool, optional
            Output values.  Used for performance testing since the generated
            values are not returned.

        Returns
        -------
        out : uint or ndarray
            Drawn samples.

        Notes
        -----
        This method directly exposes the the raw underlying pseudo-random
        number generator. All values are returned as unsigned 64-bit
        values irrespective of the number of bits produced by the PRNG.

        See the class docstring for the number of bits returned.
        """
        return random_raw(&self._bitgen, self.lock, size, output)

    def _benchmark(self, Py_ssize_t cnt, method=u'uint64'):
        return benchmark(&self._bitgen, self.lock, cnt, method)

    def seed(self, seed=None):
        """
        seed(seed=None)

        Seed the generator.

        This method is called at initialized. It can be called again to
        re-seed the generator.

        Parameters
        ----------
        seed : {int, ndarray}, optional
            Seed for PRNG. Can be a single 64 bit unsigned integer or an array
            of 64 bit unsigned integers.

        Raises
        ------
        ValueError
            If seed values are out of range for the PRNG.
        """
        ub = 2 ** self.size
        if seed is None:
            try:
                state = random_entropy(self.size // 32)
            except RuntimeError:
                state = random_entropy(8, 'fallback')
        else:
            state = seed_by_array(seed, 1)
        dtype = np.uint64 if self.size==64 else np.uint32
        state = state.view(dtype).item(0)
        if self.size == 64:
            jsf64_seed(&self.rng_state, <uint64_t>state)
        else:
            jsf32_seed(&self.rng_state, <uint32_t>state)
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
        return {'bit_generator': self.__class__.__name__,
                'state': {'a':a,'b':b,'c':c,'d':d,
                          'p':self.rng_state.p,
                          'q':self.rng_state.q,
                          'r':self.rng_state.r},
                'size': self.size,
                'has_uint32': self.rng_state.has_uint32,
                'uinteger': self.rng_state.uinteger}

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        self.size = value['size']
        state = value['state']
        self.setup_generator(state['p'],state['q'],state['r'])
        if self.size == 64:
            self.rng_state.a.u64 = state['a']
            self.rng_state.b.u64 = state['b']
            self.rng_state.c.u64 = state['c']
            self.rng_state.d.u64 = state['d']
        else:
            self.rng_state.a.u32 = state['a']
            self.rng_state.b.u32 = state['b']
            self.rng_state.c.u32 = state['c']
            self.rng_state.d.u32 = state['d']

        self.rng_state.has_uint32 = value['has_uint32']
        self.rng_state.uinteger = value['uinteger']

    @property
    def ctypes(self):
        """
        ctypes interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing ctypes wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        if self._ctypes is None:
            self._ctypes = prepare_ctypes(&self._bitgen)

        return self._ctypes

    @property
    def cffi(self):
        """
        CFFI interface

        Returns
        -------
        interface : namedtuple
            Named tuple containing CFFI wrapper

            * state_address - Memory address of the state struct
            * state - pointer to the state struct
            * next_uint64 - function pointer to produce 64 bit integers
            * next_uint32 - function pointer to produce 32 bit integers
            * next_double - function pointer to produce doubles
            * bitgen - pointer to the bit generator struct
        """
        if self._cffi is not None:
            return self._cffi
        self._cffi = prepare_cffi(&self._bitgen)
        return self._cffi

    @property
    def generator(self):
        """
        Removed, raises NotImplementedError
        """
        raise NotImplementedError('This method for accessing a Generator has'
                                  'been removed.')
