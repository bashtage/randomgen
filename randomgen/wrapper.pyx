#!python

from libc.stdint cimport uint32_t

from randomgen.common cimport BitGenerator
from randomgen.distributions cimport next_double_t, next_uint32_t, next_uint64_t

import ctypes

__all__ = ["UserBitGenerator"]

cdef object raw_32_to_64(object func):
    def f(voidp):
        return func(voidp) << 32 | func(voidp)
    return f


cdef object raw_64_to_32(object func):
    def f(voidp):
        return func(voidp) >> 32
    return f


cdef object raw_64_to_double(object func):
    def f(voidp):
        return (func(voidp) >> 11) / 9007199254740992.0
    return f


cdef object raw_32_to_double(object func):
    def f(voidp):
        a = func(voidp) >> 5
        b = func(voidp) >> 6

        return (a * 67108864.0 + b) / 9007199254740992.0
    return f


cdef class UserBitGenerator(BitGenerator):
    """
    UserBitGenerator(next_raw, bits=64, next_64=None, next_32=None, next_double=None, state=None, state_getter=None, state_setter=None)

    Construct a bit generator from  Python functions

    Parameters
    ----------
    next_raw : callable
        A callable that returns either 64 or 32 random bits. It must accept
        a single input which is a void pointer to a memory address.
    bits : {32, 64}, default 64
        The number of bits output by the next_raw callable. Must be either
        32 or 64.
    next_64 : callable, default None
        A callable with the same signature as as next_raw that always return
        64 bits. If not provided, this function is constructed using next_raw.
    next_32 : callable, default None
        A callable with the same signature as as next_raw that always return
        32 bits. If not provided, this function is constructed using next_raw.
    next_double : callable, default None
        A callable with the same signature as as next_raw that always return
        a random double in [0,1). If not provided, this function is constructed
        using next_raw.
    state : ctypes pointer, default None
        A ctypes pointer to pass into the next functions. In most cases
        this should be None, in which case the null pointer is passed.
    state_getter : callable, default None
        A callable that returns the state of the bit generator. If not
        provided, getting the ``state`` property will raise
        NotImplementedError.
    state_setter : callable, default None
        A callable that sets the state of the bit generator. Must take
        a single input. If not provided, getting the ``state`` property
        will raise NotImplementedError.

    Examples
    --------
    A generator that rotates across 4 values from random.org.

    >>> import numpy as np
    >>> rv = np.array([ 7713239619832409074, 17318243661941184039,
    ...                14412717025735663865, 521015634160378615, 0],
    ...                dtype=np.uint64)
    >>> def next_raw(voidp):
    ...     idx = int(rv[-1] % 4)
    ...     out = rv[idx]
    ...     rv[-1] += 1
    ...     return int(out)
    >>> bg = UserBitGenerator(next_raw)

    See the documentation for a more realistic example.
    """
    cdef object next_raw, next_64, next_32, next_double, funcs, state
    cdef object state_setter, state_getter, input_type
    cdef uint32_t uinteger
    cdef int has_uint
    cdef int bits
    cdef size_t next_64_ptr, next_32_ptr, next_double_ptr, next_raw_ptr
    cdef size_t state_ptr

    def __init__(self, next_raw, bits=64, next_64=None, next_32=None,
                 next_double=None, state=None, state_getter=None,
                 state_setter=None):
        BitGenerator.__init__(self, 0)
        self.funcs = {"next_raw": next_raw,
                      "next_64": next_64,
                      "next_32": next_32,
                      "next_double": next_double
                      }
        self.bits = bits
        self.next_raw = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)(next_raw)
        self.next_64 = self._setup_64()
        self.next_32 = self._setup_32()
        self.next_double = self._setup_double()
        self.state = state

        self.state_getter = state_getter
        self.state_setter = state_setter

        self.next_64_ptr = self.next_32_ptr = 0
        self.next_raw_ptr = self.next_double_ptr = 0
        self.state_ptr = 0
        self.input_type = "Python"

        self.setup_bitgen()

    def __repr__(self):
        out = object.__repr__(self)
        out = out.replace(f"{type(self).__name__}",
                          f"{type(self).__name__}({self.input_type})")
        return out

    @property
    def state(self):
        """
        Get or set the state

        Pass through function that calls ``state_getter`` or ``state_setter``.
        If these are not available, raises NotImplementedError.
        """
        if self.state_getter is None:
            raise NotImplementedError(
                "state_getter must be set during initialization to get the state."
            )
        return self.state_getter()

    @state.setter
    def state(self, value):
        if self.state_setter is None:
            raise NotImplementedError(
                "state_setter must be set during initialization to set the state."
            )
        self.state_setter(value)

    cdef setup_bitgen(self):
        self.next_64_ptr = ctypes.cast(self.next_64, ctypes.c_void_p).value
        self.next_32_ptr = ctypes.cast(self.next_32, ctypes.c_void_p).value
        self.next_raw_ptr = ctypes.cast(self.next_raw, ctypes.c_void_p).value
        self.next_double_ptr = ctypes.cast(self.next_double, ctypes.c_void_p).value
        self.state_ptr = self.state.value if self.state is not None else 0

        # Get the raw memory address
        self._bitgen.next_uint64 = <next_uint64_t> self.next_64_ptr
        self._bitgen.next_uint32 = <next_uint32_t> self.next_32_ptr
        self._bitgen.next_double = <next_double_t> self.next_double_ptr
        self._bitgen.next_raw = <next_uint64_t> self.next_raw_ptr
        self._bitgen.state = <void *> self.state_ptr

    cdef _setup_64(self):
        if self.bits == 64 and self.funcs["next_64"] is None:
            self.funcs["next_64"] = self.funcs["next_raw"]
            return self.next_raw
        if self.funcs["next_64"] is None:
            self.funcs["next_64"] = raw_32_to_64(self.funcs["next_raw"])
        c_func_type = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.c_void_p)
        return c_func_type(self.funcs["next_64"])

    cdef _setup_32(self):
        if self.bits == 32 and self.funcs["next_32"] is None:
            self.funcs["next_32"] = self.funcs["next_raw"]
            return self.next_raw
        if self.funcs["next_32"] is None:
            self.funcs["next_32"] = raw_64_to_32(self.funcs["next_raw"])
        c_func_type = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p)
        return c_func_type(self.funcs["next_32"])

    cdef _setup_double(self):
        if self.funcs["next_double"] is None:
            if self.bits == 64:
                self.funcs["next_double"] = raw_64_to_double(self.funcs["next_raw"])
            else:
                self.funcs["next_double"] = raw_32_to_double(self.funcs["next_raw"])
        c_func_type = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_void_p)
        return c_func_type(self.funcs["next_double"])

    @classmethod
    def from_cfunc(cls, next_raw, next_64, next_32, next_double, state,
                   state_getter=None, state_setter=None):
        """
        from_cfunc(next_raw, next_64, next_32, next_double, state,
                   state_getter=None, state_setter=None)

        Construct a bit generator from ctypes pointers

        Parameters
        ----------
        next_raw: numba.core.ccallback.CFunc
            A numba callback with a signature uint64(void) the return the
            next raw value from the underlying PRNG.
        next_64: numba.core.ccallback.CFunc
            A numba callback with a signature uint64(void) the return the
            next 64 bits from the underlying PRNG.
        next_32: numba.core.ccallback.CFunc
            A numba callback with a signature uint32(void) the return the
            next 32 bits from the underlying PRNG.
        next_double: numba.core.ccallback.CFunc
            A numba callback with a signature uint32(void) the return the
            next double in [0,1) from the underlying PRNG.
        state : ctypes.c_void_p
            A void pointer to the state. Passed to the next functions when
            generating random variates.
        state_getter : callable, default None
            A callable that returns the state of the bit generator. If not
            provided, getting the ``state`` property will raise
            NotImplementedError.
        state_setter : callable, default None
            A callable that sets the state of the bit generator. Must take
            a single input. If not provided, getting the ``state`` property
            will raise NotImplementedError.

        Notes
        -----
        See the documentation for an example of a generator written using
        numba.
        """
        cdef UserBitGenerator bit_gen

        try:
            from numba.core.ccallback import CFunc
        except ImportError:
            raise ImportError("numba is required for cfunc support")
        input_names = ("next_raw", "next_64", "next_32", "next_double")
        inputs = (next_raw, next_64, next_32, next_double)
        for i_n, i in zip(input_names, inputs):
            if not isinstance(i, CFunc):
                raise TypeError(f"{i_n} must be a numba CFunc {type(i)}")

        bit_gen = cls.from_ctypes(next_raw.ctypes,
                                  next_64.ctypes,
                                  next_32.ctypes,
                                  next_double.ctypes,
                                  state,
                                  state_getter=state_getter,
                                  state_setter=state_setter)
        bit_gen.input_type = "CFunc"
        return bit_gen

    @classmethod
    def from_ctypes(cls, next_raw, next_64, next_32, next_double, state,
                    state_getter=None, state_setter=None):
        """
        from_ctypes(next_raw, next_64, next_32, next_double, state,
                    state_getter=None, state_setter=None)

        Construct a bit generator from ctypes pointers

        Parameters
        ----------
        next_raw: CFunctionType
            A CFunctionType returning ctypes.c_uint64 and taking one
            ctypes.c_void_p input that returns the next raw value from the
            underlying PRNG.
        next_64: CFunctionType
            A CFunctionType returning ctypes.c_uint64 and taking one
            ctypes.c_void_p input that returns the next 64 bits from the
            underlying PRNG.
        next_32: CFunctionType
            A CFunctionType returning ctypes.c_uint64 and taking one
            ctypes.c_void_p input that returns the next 32 bits from the
            underlying PRNG.
        next_double: CFunctionType
            A CFunctionType returning ctypes.c_uint64 and taking one
            ctypes.c_void_p input that returns the next double in [0,1) value
            from the underlying PRNG.
        state : ctypes.c_void_p
            A void pointer to the state. Passed to the next functions when
            generating random variates.
        state_getter : callable, default None
            A callable that returns the state of the bit generator. If not
            provided, getting the ``state`` property will raise
            NotImplementedError.
        state_setter : callable, default None
            A callable that sets the state of the bit generator. Must take
            a single input. If not provided, getting the ``state`` property
            will raise NotImplementedError.

        Notes
        -----
        See the documentation for an example of a generator written in C
        called using this interface.
        """
        input_names = ("next_raw", "next_64", "next_32", "next_double")
        inputs = (next_raw, next_64, next_32, next_double)
        restypes = (ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint32, ctypes.c_double)
        for i_n, inp, r in zip(input_names, inputs, restypes):
            valid = hasattr(inp, "argtypes")
            valid = valid and hasattr(inp, "restype")
            if valid:
                valid = valid and inp.restype == r
            if not valid:
                raise TypeError(
                    f"{i_n} must be a ctypes function with argtypes "
                    f"{(ctypes.c_void_p,)} (or equivalent) and restype {r}."
                )

        cdef UserBitGenerator bit_gen

        def f(x):
            return 0
        assert f(1) == 0

        bit_gen = cls(f, 64)
        bit_gen.next_64 = next_64
        bit_gen.next_32 = next_32
        bit_gen.next_double = next_double
        bit_gen.next_raw = next_raw
        bit_gen.state = state

        bit_gen.state_setter = state_setter
        bit_gen.state_getter = state_getter

        bit_gen.setup_bitgen()
        bit_gen.input_type = "CTypes"
        return bit_gen
