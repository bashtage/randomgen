#!python

import numpy as np

cimport numpy as np

from threading import Lock

cimport libc.stdint
from cpython cimport PyObject
from cpython.exc cimport PyErr_Clear, PyErr_Occurred, PyErr_SetString

np.import_array()

__all__ = ["RDRAND"]

DEF BUFFER_SIZE = 256

ERROR_MSG = """\
Unable to get random values from RDRAND after {retries} retries. This can
happen if many process are accessing the hardware random number generator
simultaneously so that its capacity is being constantly exceeded. You can
increase the number of retries to slow down the generation on contested CPUs.
"""

cdef uint64_t rdrand_uint64(void* st) noexcept nogil:
    cdef PyObject *err
    cdef rdrand_state *state
    cdef int status
    cdef uint64_t val
    state = <rdrand_state*>st
    status = 1
    if state.status == 1 and state.buffer_loc < BUFFER_SIZE:
        val = state.buffer[state.buffer_loc]
        state.buffer_loc += 1
        return val
    elif state.status == 1:
        # Only refill if good status
        # This function will
        status = rdrand_fill_buffer(state)
    val = state.buffer[state.buffer_loc]
    state.buffer_loc += 1
    if status == 0:
        # Only raise on a status change
        with gil:
            err = PyErr_Occurred()
            if err == NULL:
                retries = state.retries
                msg = ERROR_MSG.format(retries=retries).encode("utf8")
                PyErr_SetString(RuntimeError, msg)
    return val


cdef uint32_t rdrand_uint32(void *st) noexcept nogil:
    # TODO: This is lazy
    return <uint32_t>rdrand_uint64(st)

cdef double rdrand_double(void* st) noexcept nogil:
    return uint64_to_double(rdrand_uint64(st))


cdef class RaisingLock:
    """
    A Lock that wraps threading.Lock can can raise errors.

    Raises the last exception set while the lock was held,
    if any. It clears the error when the lock is acquired.

    Notes
    -----
    This class has been specially designed for issues unique to RDRAND.
    """
    cdef object lock
    cdef PyObject *err

    def __init__(self):
        self.lock = Lock()
        self.err = NULL

    def acquire(self, blocking=True, timeout=-1):
        PyErr_Clear()
        return self.lock.acquire(blocking, timeout)

    def release(self):
        self.err = PyErr_Occurred()
        if self.err != NULL:
            try:
                # Python operation causes error to be raised
                print()
            except Exception as exc:
                self.release()
                raise exc
        self.lock.release()

    def __enter__(self):
        self.acquire()

    def __exit__(self, type, value, traceback):
        self.release()


cdef class RDRAND(BitGenerator):
    """
    RDRAND(seed=None, *, retries=10)

    Container for the hardware RDRAND random number generator.

    Parameters
    ----------
    seed : None
        Must be None. Raises if any other value is passed.
    retries : int
        The number of times to retry. On CPUs with many cores it is possible
        for RDRAND to fail if heavily utilized. retries sets the number of
        retries before a RuntimeError is raised. Each retry issues a pause
        instruction which waits a CPU-specific number of cycles (140 on
        Skylake [1]_). The default value of 10 is recommended by Intel ([2]_).
        You can set any value up-to the maximum integer size on your platform
        if you have issues with errors, although the practical maximum is less
        than 100. See Notes for more on the error state.

    Attributes
    ----------
    lock : threading.Lock
        Lock instance that is shared so that the same bit git generator can
        be used in multiple Generators without corrupting the state. Code that
        generates values from a bit generator should hold the bit generator's
        lock.
    seed_seq : None
        Always None since RDRAND cannot be seeded.

    Notes
    -----
    RDRAND is a hardware random number generator that is available on Intel
    processors from the Ivy Bridge line (2012) or later, and AMD processors
    starting in 2015.

    RDRAND has been audited and is reported to be a secure generator. It is
    **much slower** than software BitGenerators and so is only useful in
    applications where security is more important than performance.

    **State and Seeding**

    Results from ``RDRAND`` are **NOT** reproducible.

    ``RDRAND`` uses a hardware generated seed and so cannot be seeded. The
    state contains a single integer value ``status`` that takes the value 1
    if all calls have succeeded and 0 if any fail. A failure of a call to
    RDRAND does not propagate, and so users much check this value to determine
    if results are random.

    **Parallel Features**

    ``RDRAND`` is stateless and so multiple instances can be used in parallel.

    >>> from numpy.random import Generator
    >>> from randomgen import RDRAND
    >>> rg = [Generator(RDRAND()) for _ in range(10)]

    **Exceptions**

    Bit generators are designed to run as quickly as possible to produce
    deterministic but chaotic sequences. With the exception of RDRAND, all
    other bit generators cannot fail (short of a massive CPU issue). RDRAND
    can fail to produce a random value if many threads are all utilizing the
    same random generator, and so it is necessary to check a flag to ensure
    that the instruction has succeeded. When it does not exceed, an exception
    should be raised. However, bit generators operate *without* the Python GIL
    which means that they cannot directly raise.  Instead, if an error is
    detected when producing random values using RDRAND, the Python error flag
    is set with a RuntimError.  This error must then be checked for. In most
    applications this happens automatically since the Lock attached to this
    instance will check the error state when exiting and raise RuntimError.

    If you write custom code that uses lower-level function, e.g., the
    PyCapsule, you will either need to check the status flag in the
    state structure, or use PyErr_Occurred to see if an error occurred
    during generation.

    To see the exception you will generate, you can run this invalid code

    >>> from numpy.random import Generator
    >>> from randomgen import RDRAND
    >>> bitgen = RDRAND()
    >>> state = bitgen.state
    >>> state["retries"] = -1  # Ensure always fails
    >>> bitgen.state = state
    >>> gen = Generator(bitgen)

    The next command will always raise RuntimeError.

    >>> gen.standard_normal(size=10)

    The RDRAND-provided function ``random_raw`` also checks for success
    and will raise if not able to use RDRAND

    >>> bitgen.random_raw()

    Note that ``random_raw`` has been customized for the needs to RDRAND
    and does not rely on the Lock to raise.  Instead it checks the status
    directly and raises if the status is invalid.

    Finally, you can directly check if there have been any errors by
    inspecting the ``success`` property

    >>> bitgen = RDRAND()
    >>> assert bitgen.success  # True
    >>> bitgen.random_raw(10)
    >>> assert bitgen.success  # Still true

    You will only ever see an AssertionError if the RDRAND has failed.
    Since you will first see a RuntimeError, the second assert will not
    execute without some manual intervention.

    **No Compatibility Guarantee**

    ``RDRAND`` is hardware dependent and not reproducible, and so there is no
    stream guarantee.

    Raises
    ------
    RuntimeError
        If RDRAND is not available on the platform.

    Examples
    --------
    >>> from numpy.random import Generator
    >>> from randomgen import RDRAND
    >>> rg = Generator(RDRAND())
    >>> rg.standard_normal()
    0.123  # random

    References
    ----------
    .. [1] Software.intel.com. 2020. Intel® Intrinsics Guide. [online]
       Available at:
       <https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_pause&expand=4141>
       [Accessed 10 July 2020].
    .. [2] Intel. 2020. Intel® Digital Random Number Generator (DRNG)
       Software Implementation. (online) Available at:
       <https://software.intel.com/content/www/us/en/develop/articles/intel-digital-random-number-generator-drng-software-implementation-guide.html>
       [Accessed 10 July 2020].
    """
    cdef rdrand_state rng_state

    def __init__(self, seed=None, *, int retries=10):
        cdef int i
        if seed is not None:
            raise TypeError("seed cannot be set and so must be None")
        BitGenerator.__init__(self, seed)
        self.lock = RaisingLock()
        if not rdrand_capable():
            raise RuntimeError(  # pragma: no cover
                "The RDRAND instruction is not available"  # pragma: no cover
            )   # pragma: no cover
        self.rng_state.status = 1
        if retries < 0:
            raise ValueError("retries must be a non-negative integer.")
        self.rng_state.retries = retries
        self.rng_state.weyl_seq = 0

        self.rng_state.buffer_loc = BUFFER_SIZE
        for i in range(BUFFER_SIZE):
            self.rng_state.buffer[i] = libc.stdint.UINT64_MAX
        self.seed(seed)

        self._bitgen.state = <void *>&self.rng_state
        self._bitgen.next_uint64 = &rdrand_uint64
        self._bitgen.next_uint32 = &rdrand_uint32
        self._bitgen.next_double = &rdrand_double
        self._bitgen.next_raw = &rdrand_uint64

    def _seed_from_seq(self):
        pass

    @property
    def success(self):
        """
        Gets the flag indicating that all calls to RDRAND succeeded

        Returns
        -------
        bool
            True indicates success, false indicates failure

        Notes
        -----
        Once status is set to 0, it remains 0 unless manually reset.
        This happens to ensure that it is possible to manually verify
        the status flag.
        """
        return bool(self.rng_state.status)

    def _reset(self):
        """
        Not part of the public API

        Resets RDRAND after a failure by setting status to 1 and
        setting the buller_loc to BUFFER_SIZE so that a fresh set
        of values is pulled.
        """
        if self.rng_state.status == 0:
            # Reset and ensure a new pull
            self.rng_state.status = 1
            self.rng_state.buffer_loc = BUFFER_SIZE

    cdef _set_seed_seq(self, seed_seq):
        """
        Not part of the public API

        Allows a seed sequence to be set even though it is ignored
        """
        self._seed_seq = seed_seq

    def seed(self, seed=None):
        """
        seed(seed=None)

        Compatibility function. Not used.

        Parameters
        ----------
        seed : None
            Must be None. Raises if any other value is passed.

        Raises
        ------
        TypeError
            If seed is not None
        """
        if seed is not None:
            raise TypeError("seed cannot be set and so must be None")

    def random_raw(self, size=None, bint output=True):
        """
        random_raw(size=None, output=True)

        Return randoms as generated by the underlying BitGenerator

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn. Default is None, in which case a
            single value is returned.
        output : bool, optional
            Output values. Used for performance testing since the generated
            values are not returned.

        Returns
        -------
        out : {uint64, ndarray, None}
            Drawn samples.

        Raises
        ------
        RuntimeError
            Raised if the RDRAND instruction fails after retries.
        """
        cdef np.ndarray randoms
        cdef uint64_t *randoms_data
        cdef uint64_t value
        cdef Py_ssize_t i, n
        cdef int status

        if not output:
            n = 1 if size is None else size
            status = self.rng_state.status
            with self.lock, nogil:
                for i in range(n):
                    status &= rdrand_next64(&self.rng_state, &value)
            self.rng_state.status &= status
            if status == 0:
                raise RuntimeError(ERROR_MSG.format(retries=self.rng_state.retries))
            return

        if size is None:
            with self.lock:
                status = rdrand_next64(&self.rng_state, &value)
                if status == 0:
                    self.rng_state.status = 0
                    raise RuntimeError(ERROR_MSG.format(retries=self.rng_state.retries))
            return value

        randoms = <np.ndarray>np.empty(size, np.uint64)
        randoms_data = <uint64_t*>np.PyArray_DATA(randoms)
        n = np.PyArray_SIZE(randoms)

        status = 1
        with self.lock, nogil:
            for i in range(n):
                status &= rdrand_next64(&self.rng_state, &randoms_data[i])
        self.rng_state.status &= status
        if status == 0:
            raise RuntimeError(ERROR_MSG.format(retries=self.rng_state.retries))

        return randoms

    def jumped(self, iter=1):
        """
        jumped(iter=1)

        Returns a new bit generator

        ``RDRAND`` is stateless and so the new generator can be used in a
        parallel applications.

        Parameters
        ----------
        iter : integer, positive
            This parameter is ignored

        Returns
        -------
        bit_generator : RDRAND
            New instance of generator jumped iter times

        Notes
        -----
        Provided for API compatibility
        """
        cdef RDRAND bit_generator
        bit_generator = self.__class__()
        bit_generator._set_seed_seq(self._seed_seq)

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

        Notes
        -----
        The values returned are the buffer that is used in the filling. This
        is provided for testing and is never restored even when unpickling.
        """
        cdef uint64_t[::1] buffer
        cdef int i

        buffer = np.empty(BUFFER_SIZE, dtype=np.uint64)
        for i in range(BUFFER_SIZE):
            buffer[i] = self.rng_state.buffer[i]

        return {"bit_generator": fully_qualified_name(self),
                "status": self.rng_state.status,
                "retries": self.rng_state.retries,
                "buffer_loc": self.rng_state.buffer_loc,
                "buffer": np.asarray(buffer),
                }

    @state.setter
    def state(self, value):
        if not isinstance(value, dict):
            raise TypeError("state must be a dict")
        bitgen = value.get("bit_generator", "")
        if bitgen not in (type(self).__name__, fully_qualified_name(self)):
            raise ValueError("state must be for a {0} "
                             "PRNG".format(type(self).__name__))
        self.rng_state.retries = value["retries"]
