#!python
# cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True
import warnings

from libc.math cimport sqrt
from libc.stdint cimport int64_t, uint32_t, uint64_t

import numpy as np

cimport numpy as np

from randomgen.pcg64 import PCG64

from cpython cimport (
    PyComplex_FromDoubles,
    PyComplex_ImagAsDouble,
    PyComplex_RealAsDouble,
)
from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid
from numpy.random cimport bitgen_t
from numpy.random.c_distributions cimport (
    random_standard_normal,
    random_standard_normal_fill,
)

from randomgen cimport api
from randomgen.broadcasting cimport check_output, double_fill, float_fill
from randomgen.common cimport compute_complex
from randomgen.distributions cimport (
    random_double_fill,
    random_float,
    random_long_double_fill,
    random_long_double_size,
    random_wishart_large_df,
)

__all__ = ["ExtendedGenerator"]

np.import_array()

cdef object broadcast_shape(tuple x, tuple y, bint strict):
    cdef bint cond, bcast=True
    if x == () or y == ():
        if len(x) > len(y):
            return True, x
        return True, y
    lx = len(x)
    ly = len(y)
    if lx > ly:
        shape = list(x[:lx-ly])
        x = x[lx-ly:]
    else:
        shape = list(y[:ly-lx])
        y = y[ly-lx:]
    for xs, ys in zip(x, y):
        cond = xs == ys
        if not strict:
            cond |= min(xs, ys) == 1
        bcast &= cond
        if not bcast:
            break
        shape.append(max(xs, ys))
    return bcast, tuple(shape)


cdef _factorize(cov, meth, check_valid, tol, rank):
    if meth == "svd":
        from numpy.linalg import svd

        (u, s, vh) = svd(cov)
        if rank < cov.shape[0]:
            locs = np.argsort(s)
            s[locs[:s.shape[0]-rank]] = 0.0
        psd = np.allclose(np.dot(vh.T * s, vh), cov, rtol=tol, atol=tol)
        _factor = (u * np.sqrt(s)).T
    elif meth == "factor":
        return cov
    elif meth == "eigh":
        from numpy.linalg import eigh

        # could call linalg.svd(hermitian=True), but that calculates a
        # vh we don't need
        (s, u) = eigh(cov)
        if rank < cov.shape[0]:
            locs = np.argsort(s)
            s[locs[:s.shape[0]-rank]] = 0.0
        psd = not np.any(s < -tol)
        _factor = (u * np.sqrt(abs(s))).T
    else:
        if rank == cov.shape[0]:
            from numpy.linalg import cholesky

            _factor = cholesky(cov).T
            psd = True
        else:
            try:
                from scipy.linalg import get_lapack_funcs
            except ImportError:
                raise ImportError(
                    "SciPy is required when using Cholesky factorization with "
                    "reduced rank covariance."
                )

            func = get_lapack_funcs("pstrf")
            _factor, _, rank_c, _ = func(cov)
            _factor = np.triu(_factor)
            psd = rank_c >= rank

    if not psd and check_valid != "ignore":
        if rank < cov.shape[0]:
            msg = f"The {rank} is less than the minimum required rank."
        else:
            msg = "The covariance is not positive-semidefinite."
        if check_valid == "warn":
            warnings.warn(msg, RuntimeWarning)
        else:
            raise ValueError(msg)
    return _factor


cdef class ExtendedGenerator:
    """
    ExtendedGenerator(bit_generator=None)

    Additional random value generator using a bit generator source.

    ``ExtendedGenerator`` exposes methods for generating random numbers
    from some distributions that are not in numpy.random.Generator.

    Parameters
    ----------
    bit_generator : BitGenerator, optional
        Bit generator to use as the core generator. If none is provided, uses
        PCG64(variant="cm-dxsm").

    See Also
    --------
    numpy.random.Generator
        The primary generator of random variates.

    Examples
    --------
    >>> from randomgen import ExtendedGenerator
    >>> rg = ExtendedGenerator()
    >>> rg.complex_normal()
    -0.203 + .936j  # random

    Using a specific generator

    >>> from randomgen import MT19937
    >>> rg = ExtendedGenerator(MT19937())

    Share a bit generator with numpy

    >>> from numpy.random import Generator, PCG64
    >>> pcg = PCG64()
    >>> gen = Generator(pcg)
    >>> eg = ExtendedGenerator(pcg)
    """

    cdef public object _bit_generator
    cdef bitgen_t _bitgen
    cdef object lock, _generator

    def __init__(self, bit_generator=None):
        if bit_generator is None:
            bit_generator = PCG64(variant="dxsm")
        self._bit_generator = bit_generator

        capsule = bit_generator.capsule
        cdef const char *name = "BitGenerator"
        if not PyCapsule_IsValid(capsule, name):
            raise ValueError("Invalid bit generator. The bit generator must "
                             "be instantized.")
        self._bitgen = (<bitgen_t *> PyCapsule_GetPointer(capsule, name))[0]
        self.lock = bit_generator.lock
        from numpy.random import Generator
        self._generator = Generator(bit_generator)

    def __repr__(self):
        out = object.__repr__(self)
        return out.replace(type(self).__name__, self.__str__())

    def __str__(self):
        _str = type(self).__name__
        _str += "(" + type(self.bit_generator).__name__ + ")"
        return _str

    # Pickling support:
    def __getstate__(self):
        return self.bit_generator.state

    def __setstate__(self, state):
        self.bit_generator.state = state

    def __reduce__(self):
        from randomgen._pickle import __extended_generator_ctor
        return (__extended_generator_ctor, (self.bit_generator.state["bit_generator"],),
                self.bit_generator.state)

    @property
    def bit_generator(self):
        """
        Gets the bit generator instance used by the generator

        Returns
        -------
        bit_generator : numpy.random.BitGenerator
            The bit generator instance used by the generator
        """
        return self._bit_generator

    @property
    def state(self):
        """
        Get or set the bit generator's state

        Returns
        -------
        state : dict
            Dictionary containing the information required to describe the
            state of the bit generator

        Notes
        -----
        This is a trivial pass-through function. Generator does not
        directly contain or manipulate the bit generator's state.

        """
        return self._bit_generator.state

    @state.setter
    def state(self, value):
        self._bit_generator.state = value

    def uintegers(self, size=None, int bits=64):
        """
        uintegers(size=None, bits=64)

        Return random unsigned integers

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn. Default is None, in which case a
            single value is returned.
        bits : int {32, 64}
            Size of the unsigned integer to return, either 32 bit or 64 bit.

        Returns
        -------
        out : int or ndarray
            Drawn samples.

        Notes
        -----
        This method effectively exposes access to the raw underlying
        pseudo-random number generator since these all produce unsigned
        integers. In practice these are most useful for generating other
        random numbers.
        These should not be used to produce bounded random numbers by
        simple truncation.
        """
        cdef np.npy_intp i, n
        cdef np.ndarray array
        cdef uint32_t* data32
        cdef uint64_t* data64
        if bits == 64:
            if size is None:
                with self.lock:
                    return self._bitgen.next_uint64(self._bitgen.state)
            array = <np.ndarray>np.empty(size, np.uint64)
            n = np.PyArray_SIZE(array)
            data64 = <uint64_t *>np.PyArray_DATA(array)
            with self.lock, nogil:
                for i in range(n):
                    data64[i] = self._bitgen.next_uint64(self._bitgen.state)
        elif bits == 32:
            if size is None:
                with self.lock:
                    return self._bitgen.next_uint32(self._bitgen.state)
            array = <np.ndarray>np.empty(size, np.uint32)
            n = np.PyArray_SIZE(array)
            data32 = <uint32_t *>np.PyArray_DATA(array)
            with self.lock, nogil:
                for i in range(n):
                    data32[i] = self._bitgen.next_uint32(self._bitgen.state)
        else:
            raise ValueError("Unknown value of bits. Must be either 32 or 64.")

        return array

    def random(self, size=None, dtype=np.float64, out=None):
        """
        random(size=None, dtype='d', out=None)

        Return random floats in the half-open interval [0.0, 1.0).

        Results are from the "continuous uniform" distribution over the
        stated interval. To sample :math:`Unif[a, b), b > a` multiply
        the output of `random` by `(b-a)` and add `a`::

          (b - a) * random() + a

        Parameters
        ----------
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn. Default is None, in which case a
            single value is returned.
        dtype : {str, dtype}, optional
            Desired dtype of the result. One of 'd' ('float64' or np.float64), 'f'
            ('float32' of np.float32), or 'longdouble' (np.longdouble). All dtypes
            are determined by their name. The default value is 'd'.
        out : ndarray, optional
            Alternative output array in which to place the result. If size is not None,
            it must have the same shape as the provided size and must match the type of
            the output values.

        Returns
        -------
        out : {float, longdouble or ndarray}
            Array of random floats of shape `size` (unless ``size=None``, in which
            case a single float is returned). If dtype is np.longdouble, then the
            returned type is a scalar np.longdouble. Otherwise it is a float.

        Examples
        --------
        >>> randomgen.generator.random()
        0.47108547995356098 # random
        >>> type(randomgen.generator.random())
        <class 'float'>
        >>> randomgen.generator.random((3,))
        array([ 0.30220482,  0.86820401,  0.1654503]) # random

        Three-by-two array of random numbers from [-5, 0):

        >>> 5 * randomgen.generator.random((3, 2)) - 5
        array([[-3.99149989, -0.52338984], # random
               [-2.99091858, -0.79479508],
               [-1.23204345, -1.75224494]])
        """
        cdef np.ndarray out_array
        cdef long double *out_data
        cdef np.npy_intp out_size

        key = np.dtype(dtype).name
        if key == "float64":
            return double_fill(&random_double_fill, &self._bitgen, size, self.lock, out)
        elif key == "float32":
            return float_fill(&random_float, &self._bitgen, size, self.lock, out)
        elif key == np.dtype("longdouble").name:
            if np.longdouble(1).nbytes != random_long_double_size():
                raise RuntimeError(
                    "The platform and compiler long double size does not "
                    "match the size provided by NumPy longdouble. These "
                    "must match to generate random long doubles."
                )
            sz = 1 if size is None else size
            if out is not None:
                check_output(out, np.longdouble, size, False)
                out_array = <np.ndarray>out
            else:
                out_array = <np.ndarray>np.empty(sz, np.longdouble)
            out_data = <long double*>np.PyArray_DATA(out_array)
            out_size = out_array.size
            with self.lock, nogil:
                random_long_double_fill(&self._bitgen, out_size, out_data)
            if out is None and size is None:
                return out_array[0]
            return out_array
        else:
            raise TypeError("Unsupported dtype \"{key}\" for random".format(key=key))

    # Multivariate distributions:
    def multivariate_normal(self, mean, cov, size=None, check_valid="warn",
                            tol=1e-8, *, method="svd"):
        """
        multivariate_normal(mean, cov, size=None, check_valid='warn', tol=1e-8, *, method='svd')

        Draw random samples from a multivariate normal distribution.

        The multivariate normal, multinormal or Gaussian distribution is a
        generalization of the one-dimensional normal distribution to higher
        dimensions. Such a distribution is specified by its mean and
        covariance matrix. These parameters are analogous to the mean
        (average or "center") and variance (standard deviation, or "width,"
        squared) of the one-dimensional normal distribution.

        Parameters
        ----------
        mean : array_like
            Mean of the distribution. Must have shape (m1, m2, ..., mk, N) where
            (m1, m2, ..., mk) would broadcast with (c1, c2, ..., cj).
        cov : array_like
            Covariance matrix of the distribution. It must be symmetric and
            positive-semidefinite for proper sampling. Must have shape
            (c1, c2, ..., cj, N, N) where (c1, c2, ..., cj) would broadcast
            with (m1, m2, ..., mk).
        size : int or tuple of ints, optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
            If no shape is specified, a single (`N`-D) sample is returned.
        check_valid : {'warn', 'raise', 'ignore' }, optional
            Behavior when the covariance matrix is not positive semidefinite.
        tol : float, optional
            Tolerance when checking the singular values in covariance matrix.
            cov is cast to double before the check.
        method : {'svd', 'eigh', 'cholesky', 'factor'}, optional
            The cov input is used to compute a factor matrix A such that
            ``A @ A.T = cov``. This argument is used to select the method
            used to compute the factor matrix A. The default method 'svd' is
            the slowest, while 'cholesky' is the fastest but less robust than
            the slowest method. The method `eigh` uses eigen decomposition to
            compute A and is faster than svd but slower than cholesky. `factor`
            assumes that cov has been pre-factored so that no transformation is
            applied.

        Returns
        -------
        out : ndarray
            The drawn samples, of shape determined by broadcasting the
            leading dimensions of mean and cov with size, if not None.
            The final dimension is always N.

            In other words, each entry ``out[i,j,...,:]`` is an N-dimensional
            value drawn from the distribution.

        Notes
        -----
        The mean is a coordinate in N-dimensional space, which represents the
        location where samples are most likely to be generated. This is
        analogous to the peak of the bell curve for the one-dimensional or
        univariate normal distribution.

        Covariance indicates the level to which two variables vary together.
        From the multivariate normal distribution, we draw N-dimensional
        samples, :math:`X = [x_1, x_2, ... x_N]`. The covariance matrix
        element :math:`C_{ij}` is the covariance of :math:`x_i` and :math:`x_j`.
        The element :math:`C_{ii}` is the variance of :math:`x_i` (i.e. its
        "spread").

        Instead of specifying the full covariance matrix, popular
        approximations include:

          - Spherical covariance (`cov` is a multiple of the identity matrix)
          - Diagonal covariance (`cov` has non-negative elements, and only on
            the diagonal)

        This geometrical property can be seen in two dimensions by plotting
        generated data-points:

        >>> mean = [0, 0]
        >>> cov = [[1, 0], [0, 100]]  # diagonal covariance

        Diagonal covariance means that points are oriented along x or y-axis:

        >>> from numpy.random import ExtendedGenerator
        >>> erg = ExtendedGenerator()
        >>> import matplotlib.pyplot as plt
        >>> x, y = erg.multivariate_normal(mean, cov, 5000).T
        >>> plt.plot(x, y, 'x')
        >>> plt.axis('equal')
        >>> plt.show()

        Note that the covariance matrix must be positive semidefinite (a.k.a.
        nonnegative-definite). Otherwise, the behavior of this method is
        undefined and backwards compatibility is not guaranteed. See [1]_
        and [2]_ for more information.

        References
        ----------
        .. [1] Papoulis, A., "Probability, Random Variables, and Stochastic
               Processes," 3rd ed., New York: McGraw-Hill, 1991.
        .. [2] Duda, R. O., Hart, P. E., and Stork, D. G., "Pattern
               Classification," 2nd ed., New York: Wiley, 2001.

        Examples
        --------
        >>> from randomgen import ExtendedGenerator
        >>> erg = ExtendedGenerator()
        >>> mean = (1, 2)
        >>> cov = [[1, 0], [0, 1]]
        >>> x = erg.multivariate_normal(mean, cov, (3, 3))
        >>> x.shape
        (3, 3, 2)

        The following is probably true, given that 0.6 is roughly twice the
        standard deviation:

        >>> list((x[0,0,:] - mean) < 0.6)
        [True, True] # random

        """
        if check_valid not in ("warn", "raise", "ignore"):
            raise ValueError("check_valid must equal 'warn', 'raise', or 'ignore'")

        mean = np.array(mean)
        cov = np.array(cov, dtype=np.double)
        if mean.ndim < 1:
            raise ValueError("mean must have at least 1 dimension")
        if cov.ndim < 2:
            raise ValueError("cov must have at least 2 dimensions")
        n = mean.shape[mean.ndim - 1]
        cov_dim = cov.ndim
        if not (cov.shape[cov_dim - 1] == cov.shape[cov_dim - 2] == n):
            raise ValueError(
                f"The final two dimension of cov "
                f"({cov.shape[cov_dim - 1], cov.shape[cov_dim - 2]}) must match "
                f"the final dimension of mean ({n}). mean must be 1 dimensional"
            )

        drop_dims = (mean.ndim == 1) and (cov.ndim == 2)
        if mean.ndim == 1:
            mean = mean.reshape((1, n))
        if cov.ndim == 2:
            cov = cov.reshape((1, n, n))

        _factors = np.empty_like(cov)
        for loc in np.ndindex(*cov.shape[:len(cov.shape)-2]):
            _factors[loc] = _factorize(cov[loc], method, check_valid, tol, n)

        out_shape = np.broadcast(mean[..., 0], cov[..., 0, 0]).shape
        if size is not None:
            if isinstance(size, (int, np.integer)):
                size = (size,)
            error = len(size) < len(out_shape)
            final_size = list(size[: -len(out_shape)])
            for s, os in zip(size[-len(out_shape) :], out_shape):
                if error or not (s == 1 or os == 1 or s == os):
                    raise ValueError(
                        f"The desired out size {size} is not compatible with "
                        f"the broadcast size of mean and cov {out_shape}. "
                        f"The desired size must have the same number (or "
                        f"more) dimensions ({len(size)}) as the broadcast "
                        f"size or mean and covariance (len({out_shape}). "
                        f"The final {len(out_shape)} elements of size must be "
                        f"either 1 or the same as the corresponding element "
                        f"of the broadcast size."
                    )
                final_size.append(max(s, os))
            out_shape = tuple(final_size)

        out = self._generator.standard_normal(out_shape + (1, n,))
        prod = np.matmul(out, _factors)
        final = mean + np.squeeze(prod, axis=prod.ndim - 2)
        if drop_dims and final.shape[0] == 1:
            final = final.reshape(final.shape[1:])
        return final

    def complex_normal(self, loc=0.0, gamma=1.0, relation=0.0, size=None):
        """
        complex_normal(loc=0.0, gamma=1.0, relation=0.0, size=None)

        Draw random samples from a complex normal (Gaussian) distribution.

        Parameters
        ----------
        loc : complex or array_like of complex
            Mean of the distribution.
        gamma : float, complex or array_like of float or complex
            Variance of the distribution
        relation : float, complex or array_like of float or complex
            Relation between the two component normals
        size : int or tuple of ints, optional
            Output shape. If the given shape is, e.g., ``(m, n, k)``, then
            ``m * n * k`` samples are drawn. If size is ``None`` (default),
            a single value is returned if ``loc``, ``gamma`` and ``relation``
            are all scalars. Otherwise,
            ``np.broadcast(loc, gamma, relation).size`` samples are drawn.

        Returns
        -------
        out : complex or ndarray
            Drawn samples from the parameterized complex normal distribution.

        See Also
        --------
        numpy.random.Generator.normal : random values from a real-valued
            normal distribution

        Notes
        -----
        Complex normals are generated from a bivariate normal where the
        variance of the real component is 0.5 Re(gamma + relation), the
        variance of the imaginary component is 0.5 Re(gamma - relation), and
        the covariance between the two is 0.5 Im(relation). The implied
        covariance matrix must be positive semi-definite and so both variances
        must be zero and the covariance must be weakly smaller than the
        product of the two standard deviations. See [1]_ and [2]_ for
        additional details.

        References
        ----------
        .. [1] Wikipedia, "Complex normal distribution",
               https://en.wikipedia.org/wiki/Complex_normal_distribution
        .. [2] Leigh J. Halliwell, "Complex Random Variables" in "Casualty
               Actuarial Society E-Forum", Fall 2015.

        Examples
        --------
        Draw samples from the distribution:

        >>> s = randomgen.generator.complex_normal(size=1000)

        """
        cdef np.ndarray ogamma, orelation, oloc, randoms, v_real, v_imag, rho
        cdef double *randoms_data
        cdef double fgamma_r, fgamma_i, frelation_r, frelation_i
        cdef double fvar_r, fvar_i, floc_r, floc_i, f_real, f_imag, f_rho
        cdef np.npy_intp i, j, n, n2
        cdef np.broadcast it

        oloc = <np.ndarray>np.PyArray_FROM_OTF(
            loc, np.NPY_COMPLEX128, api.NPY_ARRAY_ALIGNED
        )
        ogamma = <np.ndarray>np.PyArray_FROM_OTF(
            gamma, np.NPY_COMPLEX128, api.NPY_ARRAY_ALIGNED
        )
        orelation = <np.ndarray>np.PyArray_FROM_OTF(
            relation, np.NPY_COMPLEX128, api.NPY_ARRAY_ALIGNED
        )

        if (np.PyArray_NDIM(ogamma) ==
                np.PyArray_NDIM(orelation) ==
                np.PyArray_NDIM(oloc) == 0):
            floc_r = PyComplex_RealAsDouble(loc)
            floc_i = PyComplex_ImagAsDouble(loc)
            fgamma_r = PyComplex_RealAsDouble(gamma)
            fgamma_i = PyComplex_ImagAsDouble(gamma)
            frelation_r = PyComplex_RealAsDouble(relation)
            frelation_i = 0.5 * PyComplex_ImagAsDouble(relation)

            fvar_r = 0.5 * (fgamma_r + frelation_r)
            fvar_i = 0.5 * (fgamma_r - frelation_r)
            if fgamma_i != 0:
                raise ValueError("Im(gamma) != 0")
            if fvar_i < 0:
                raise ValueError("Re(gamma - relation) < 0")
            if fvar_r < 0:
                raise ValueError("Re(gamma + relation) < 0")
            f_rho = 0.0
            if fvar_i > 0 and fvar_r > 0:
                f_rho = frelation_i / sqrt(fvar_i * fvar_r)
            if f_rho > 1.0 or f_rho < -1.0:
                raise ValueError("Im(relation) ** 2 > Re(gamma ** 2 - relation** 2)")

            if size is None:
                f_real = random_standard_normal(&self._bitgen)
                f_imag = random_standard_normal(&self._bitgen)

                compute_complex(&f_real, &f_imag, floc_r, floc_i, fvar_r,
                                fvar_i, f_rho)
                return PyComplex_FromDoubles(f_real, f_imag)

            randoms = <np.ndarray>np.empty(size, np.complex128)
            randoms_data = <double *>np.PyArray_DATA(randoms)
            n = np.PyArray_SIZE(randoms)

            j = 0
            with self.lock, nogil:
                for i in range(n):
                    f_real = random_standard_normal(&self._bitgen)
                    f_imag = random_standard_normal(&self._bitgen)
                    compute_complex(&f_real, &f_imag, floc_r, floc_i, fvar_r,
                                    fvar_i, f_rho)
                    randoms_data[j] = f_real
                    randoms_data[j+1] = f_imag
                    j += 2

            return randoms

        gpc = ogamma + orelation
        gmc = ogamma - orelation
        v_real = <np.ndarray>(0.5 * np.real(gpc))
        if np.any(np.less(v_real, 0)):
            raise ValueError("Re(gamma + relation) < 0")
        v_imag = <np.ndarray>(0.5 * np.real(gmc))
        if np.any(np.less(v_imag, 0)):
            raise ValueError("Re(gamma - relation) < 0")
        if np.any(np.not_equal(np.imag(ogamma), 0)):
            raise ValueError("Im(gamma) != 0")

        cov = 0.5 * np.imag(orelation)
        rho = np.zeros_like(cov)
        idx = (v_real.flat > 0) & (v_imag.flat > 0)
        rho.flat[idx] = cov.flat[idx] / np.sqrt(v_real.flat[idx] * v_imag.flat[idx])
        if np.any(cov.flat[~idx] != 0) or np.any(np.abs(rho) > 1):
            raise ValueError("Im(relation) ** 2 > Re(gamma ** 2 - relation ** 2)")

        if size is not None:
            randoms = <np.ndarray>np.empty(size, np.complex128)
        else:
            it = np.PyArray_MultiIterNew4(oloc, v_real, v_imag, rho)
            randoms = <np.ndarray>np.empty(it.shape, np.complex128)

        randoms_data = <double *>np.PyArray_DATA(randoms)
        n = np.PyArray_SIZE(randoms)

        it = np.PyArray_MultiIterNew5(randoms, oloc, v_real, v_imag, rho)
        with self.lock, nogil:
            n2 = 2 * n  # Avoid compiler noise for cast
            for i in range(n2):
                randoms_data[i] = random_standard_normal(&self._bitgen)
        with nogil:
            j = 0
            for i in range(n):
                floc_r= (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
                floc_i= (<double*>np.PyArray_MultiIter_DATA(it, 1))[1]
                fvar_r = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
                fvar_i = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
                f_rho = (<double*>np.PyArray_MultiIter_DATA(it, 4))[0]
                compute_complex(&randoms_data[j], &randoms_data[j+1], floc_r,
                                floc_i, fvar_r, fvar_i, f_rho)
                j += 2
                np.PyArray_MultiIter_NEXT(it)

        return randoms

    cdef object random_wishart_small_df(
            self, int64_t df, np.npy_intp dim, np.npy_intp num, object n
    ):
        double_fill(&random_standard_normal_fill, &self._bitgen, None, self.lock, n)
        return np.matmul(np.transpose(n, (0, 2, 1)), n)

    def standard_wishart(self, int64_t df, np.npy_intp dim, size=None, rescale=True):
        """
        standard_wishart(df, dim, size=None, rescale=True)

        Draw samples from the Standard Wishart and Pseudo-Wishart distributions

        Parameters
        ----------
        df : int
            The degree-of-freedom for the simulated Wishart variates.
        dim : int
            The dimension of the simulated Wishart variates.
        size : int or tuple of ints, optional
            Output shape, excluding trailing dims. If the given shape is, e.g.,
            ``(m, n, k)``, then ``m * n * k`` samples are drawn, each with
            shape ``(dim, dim)``. The output then has shape
            ``(m, n, k, dim, dim)``. Default is None, in which case a single
            value with shape ``(dim, dim)`` is returned.
        rescale : bool, optional
            Flag indicating whether to rescale the outputs to have expectation
            identity. The default is True. If ``rescale`` is False, then the
            expected value of the generated variates is `df` * eye(`dim`).

        Returns
        -------
        ndarray
            The generated variates from the standard wishart distribution.

        See Also
        --------
        wishart
            Generate variates with a non-identify scale. Also support array
            inputs for `df`.

        Notes
        -----
        Uses the method of Odell and Feiveson [1]_ when `df` >= `dim`.
        Otherwise variates are directly generated as the inner product
        of `df` by `dim` arrays of standard normal random variates.
        See [1]_, [2]_, and [3]_ for more information.

        References
        ----------
        .. [1] Odell, P. L. , and A. H. Feiveson (1966) A numerical procedure
           to generate a sample covariance matrix. Jour. Amer. Stat. Assoc.
           61, 199–203
        .. [2] Uhlig, H. (1994). "On Singular Wishart and Singular Multivariate
           Beta Distributions". The Annals of Statistics. 22: 395–405
        .. [3] Dıaz-Garcıa, J. A., Jáimez, R. G., & Mardia, K. V. (1997). Wishart
           and Pseudo-Wishart distributions and some applications to shape theory.
           Journal of Multivariate Analysis, 63(1), 73-87.
        """
        cdef np.npy_intp num
        cdef np.ndarray n
        cdef double *n_data
        cdef double *out_data

        if size is None or size == ():
            sz = 1
        elif np.isscalar(size):
            sz = (int(size),)
        else:
            sz = tuple(size)
        num = np.prod(sz)
        if df < dim:
            n = <np.ndarray> np.empty((num, df, dim))
            out = self.random_wishart_small_df(df, dim, num, n)
        else:
            out = <np.ndarray>np.empty((num, dim, dim))
            out_data = <double *>np.PyArray_DATA(out)
            n = <np.ndarray> np.empty((dim, dim))
            n_data = <double *> np.PyArray_DATA(n)
            with self.lock, nogil:
                random_wishart_large_df(&self._bitgen, df, dim, num, out_data, n_data)
        if size is None or size == ():
            out.shape = (dim, dim)
        else:
            out.shape = sz + (dim, dim)
        if rescale:
            out /= df
        return out

    def wishart(self, df, scale=None, size=None, *, check_valid="warn",
                tol=1e-8, rank=None, method="svd"):
        """
        wishart(df, scale, size=None, *, check_valid="warn", tol=None, rank=None, method="svd")

        Draw samples from the Wishart and pseudo-Wishart distributions.

        Parameters
        ----------
        df : {int, array_like[int]}
            Degree-of-freedom values. In array-like must broadcast with all
            but the final two dimensions of ``shape``.
        scale : array_like
            Shape matrix of the distribution. It must be symmetric and
            positive-semidefinite for sampling. Must have shape
            (c1, c2, ..., cj, N, N) where (c1, c2, ..., cj) broadcasts
            with the degree of freedom shape (d1, d2, ..., dk).
        size : int or sequence[int], optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N` by `N`, the output shape is ``(m,n,k,N,N)``.
            If no shape is specified, a single (`N` by `N`) sample is returned.
        check_valid : {'warn', 'raise', 'ignore' }, optional
            Behavior when the covariance matrix has rank less than ``rank``.
        tol : float, optional
            Tolerance when checking the rank of ``shape``. ``shape`` is cast
            to double before the check. If None, then the tolerance is
            automatically determined as a function of `N` and the limit of
            floating point precision.
        rank : int, optional
            The rank of shape when generating from the Singular Wishart
            distribution. If None, then ``rank`` is set of `N` so that the
            draws from the standard Wishart or pseudo-Wishart are generated.
        method : {'svd', 'eigh', 'cholesky', 'factor'}, optional
            The cov input is used to compute a factor matrix A such that
            ``A @ A.T = cov``. This argument is used to select the method
            used to compute the factor matrix A. The default method 'svd' is
            the slowest, while 'cholesky' is the fastest but less robust than
            the slowest method. The method `eigh` uses eigen
            decomposition to compute A and is faster than svd but slower than
            cholesky. When ``rank`` is less than `N`, then the `N` largest
            eigenvalues and their associated eigenvalues are used when method
            is `svd` or `eigh`. When method is 'cholesky, then the Cholesky
            of the upper ``rank`` by ``rank`` block is used. `factor` assumes
            that scale has been pre-factored so that no transformation is
            applied. When using `factor`, no check is performed on the rank.

        Returns
        -------
        ndarray
            The generated variates from the Wishart distribution.

        See Also
        --------
        standard_wishart
            Generate variates with an identify scale.

        Notes
        -----
        Uses the method of Odell and Feiveson [1]_ when `df` >= `dim`.
        Otherwise variates are directly generated as the inner product
        of `df` by `dim` arrays of standard normal random variates.
        See [1]_, [2]_, and [3]_ for more information.

        References
        ----------
        .. [1] Odell, P. L. , and A. H. Feiveson (1966) A numerical procedure
           to generate a sample covariance matrix. Jour. Amer. Stat. Assoc.
           61, 199–203
        .. [2] Uhlig, H. (1994). "On Singular Wishart and Singular Multivariate
           Beta Distributions". The Annals of Statistics. 22: 395–405
        .. [3] Dıaz-Garcıa, J. A., Jáimez, R. G., & Mardia, K. V. (1997). Wishart
           and Pseudo-Wishart distributions and some applications to shape theory.
           Journal of Multivariate Analysis, 63(1), 73-87.
        """
        cdef np.broadcast it
        cdef np.npy_intp block_size, dim, cnt
        cdef long shape_nd
        cdef int64_t df_i, shape_loc, last_small_df
        cdef double *lwork_data
        cdef double *large_value_data
        cdef np.ndarray df_arr, shape_arr

        # 1. Validate inputs
        df_arr = <np.ndarray>np.asarray(df, dtype=np.int64, order="C")
        if df_arr.size == 0:
            raise ValueError("At least one value is required for df")
        if not np.all(df_arr > 0):
            raise ValueError("df must contain strictly positive integer values.")
        shape_arr = <np.ndarray>np.asarray(scale, dtype=np.float64, order="C")
        shape_nd = np.PyArray_NDIM(shape_arr)
        msg = (
            "scale must be non-empty and have at least 2 dimensions. The final "
            "two dimensions must be the same so that scale's shape is (...,N,N)."
        )
        if shape_nd < 2 or shape_arr.size == 0:
            raise ValueError(msg)
        dim = np.shape(shape_arr)[shape_nd-1]
        rank_val = dim if rank is None else int(rank)
        if  np.shape(shape_arr)[shape_nd-2] != dim:
            raise ValueError(msg)
        shape_size = np.shape(shape_arr)[:shape_nd-2]
        df_size = np.shape(df_arr)
        if not df_size and not shape_size:
            df_i = df_arr.item()
            out = self.standard_wishart(df_i, dim, size, False)
            factor = _factorize(shape_arr, method, check_valid, tol, rank_val)
            np.matmul(factor, out, out=out)
            np.matmul(out, factor.T, out=out)
            return out

        if not shape_size:
            shape_dummy = np.zeros(1, dtype=np.int64)
        else:
            shape_dummy = np.arange(
                np.prod(shape_size), dtype=np.int64
            ).reshape(shape_size)
        if not df_size:
            df_dummy = np.zeros(1, dtype=np.int64)
        else:
            df_dummy = np.arange(np.prod(df_size), dtype=np.int64).reshape(df_size)
        it = np.PyArray_MultiIterNew2(df_arr, shape_dummy)
        it_shape = it.shape
        cnt = int(it.size)

        if size is None:
            out = np.empty(it_shape + (dim, dim))
            block_size = 1
        else:
            if isinstance(size, (int, np.integer)):
                size = (int(size),)
            if len(size) < len(it_shape) or size[len(size)-len(it_shape):] != it_shape:
                err_msg = (
f"""size {size} is not compatible with the broadcast shape of `df` and `scale`
{it.shape}. size must have at least as many values as the broadcasting shape
and the trailing dimensions must match exactly so that
{size[len(size) - len(it_shape):]} == {it_shape}"""
                )
                raise ValueError(err_msg)
            out = np.empty(size + (dim, dim))
            if size == it_shape:
                block_size = 1
            else:
                block_size = int(np.prod(size[:(len(size)-len(it_shape))]))
        out_shape = out.shape
        out = out.reshape((-1, np.prod(it_shape), dim, dim))
        temp_shape = np.shape(out[..., 0, :, :])
        _factors = np.empty_like(scale)
        for loc in np.ndindex(*scale.shape[:len(scale.shape) - 2]):
            _factors[loc] = _factorize(scale[loc], method, check_valid, tol, rank_val)
        _factors = _factors.reshape((-1, dim, dim))

        large_values = np.empty((block_size, dim, dim), dtype=np.float64)
        large_value_data = <double *>np.PyArray_DATA(large_values)
        large_work = np.empty((dim, dim), dtype=np.float64)
        lwork_data = <double *>np.PyArray_DATA(large_work)
        last_small_df = -1
        for i in range(cnt):
            df_i = (<int64_t *> np.PyArray_MultiIter_DATA(it, 0))[0]
            shape_loc = (<int64_t *> np.PyArray_MultiIter_DATA(it, 1))[0]
            if df_i < dim:
                if df_i != last_small_df:
                    last_small_df = df_i
                    small_work = np.empty((block_size, df_i, dim), dtype=np.float64)
                temp = self.random_wishart_small_df(df_i, dim, block_size, small_work)
            else:
                with self.lock, nogil:
                    random_wishart_large_df(&self._bitgen,
                                            df_i,
                                            <np.npy_intp>dim,
                                            <np.npy_intp>block_size,
                                            large_value_data,
                                            lwork_data)
                temp = large_values
            factor = _factors[shape_loc]
            np.matmul(factor, temp, out=temp)
            np.matmul(temp, factor.T, out=temp)
            out[..., i, :, :] = temp.reshape(temp_shape)

            np.PyArray_MultiIter_NEXT(it)
        return out.reshape(out_shape)

    def multivariate_complex_normal(self, loc, gamma=None, relation=None, size=None, *,
                                    check_valid="warn", tol=1e-8, method="svd"):
        r"""
        multivariate_complex_normal(loc, gamma=None, relation=None, size=None, *, check_valid="warn", tol=1e-8, method="svd")

        Draw random samples from a multivariate complex normal (Gaussian) distribution.

        Parameters
        ----------
        loc : array_like of complex
            Mean of the distribution.  Must have shape (m1, m2, ..., mk, N) where
            (m1, m2, ..., mk) would broadcast with (g1, g2, ..., gj) and
            (r1, r2, ..., rq).
        gamma : array_like of float or complex, optional
            Covariance of the real component of the distribution.  Must have shape
            (g1, g2, ..., gj, N, N) where (g1, g2, ..., gj) would broadcast
            with (m1, m2, ..., mk) and (r1, r2, ..., rq). If not provided,
            an identity matrix is used which produces the circularly-symmetric
            complex normal when relation is an array of 0.
        relation : array_like of float or complex, optional
            Relation between the two component normals. (r1, r2, ..., rq, N, N)
            where (r1, r2, ..., rq, N, N) would broadcast with (m1, m2, ..., mk)
            and (g1, g2, ..., gj). If not provided, set to zero which produces
            the circularly-symmetric complex normal when gamma is an identify matrix.
        size : int or tuple of ints, optional
            Given a shape of, for example, ``(m,n,k)``, ``m*n*k`` samples are
            generated, and packed in an `m`-by-`n`-by-`k` arrangement.  Because
            each sample is `N`-dimensional, the output shape is ``(m,n,k,N)``.
            If no shape is specified, a single (`N`-D) sample is returned.
        check_valid : {'warn', 'raise', 'ignore' }, optional
            Behavior when the covariance matrix implied by `gamma` and `relation`
            is not positive semidefinite.
        tol : float, optional
            Tolerance when checking the singular values in the covariance matrix
            implied by `gamma` and `relation`.
        method : {'svd', 'eigh', 'cholesky'}, optional
            The cov input is used to compute a factor matrix A such that
            ``A @ A.T = cov``. This argument is used to select the method
            used to compute the factor matrix A for the covariance implied by
            `gamma` and `relation`. The default method 'svd' is
            the slowest, while 'cholesky' is the fastest but less robust than
            the slowest method. The method `eigh` uses eigen decomposition to
            compute A and is faster than svd but slower than cholesky.

        Returns
        -------
        out : ndarray
            Drawn samples from the parameterized complex normal distributions.

        See Also
        --------
        numpy.random.Generator.normal : random values from a real-valued
            normal distribution
        randomgen.generator.ExtendedGenerator.complex_normal : random values from a
           scalar complex-valued normal distribution
        randomgen.generator.ExtendedGenerator.multivariate_normal : random values from a
           scalar complex-valued normal distribution

        Notes
        -----
        Complex normals are generated from a multivariate normal where the
        covariance matrix of the real and imaginary components is

        .. math::

           \begin{array}{c}
           X\\
           Y
           \end{array}\sim N\left(\left[\begin{array}{c}
           \mathrm{Re\left[\mu\right]}\\
           \mathrm{Im\left[\mu\right]}
           \end{array}\right],\frac{1}{2}\left[\begin{array}{cc}
           \mathrm{Re}\left[\Gamma+C\right] & \mathrm{Im}\left[C-\Gamma\right]\\
           \mathrm{Im}\left[\Gamma+C\right] & \mathrm{Re}\left[\Gamma-C\right]
           \end{array}\right]\right)

        The complex normals are then

        .. math::

           Z = X + iY

        If the implied covariance matrix is not positive semi-definite a warning
        or exception may be raised depending on the value `check_valid`.

        The implementation is based in the mathematical description in
        [1]_ and [2]_.

        References
        ----------
        .. [1] Wikipedia, "Complex normal distribution",
               https://en.wikipedia.org/wiki/Complex_normal_distribution
        .. [2] Leigh J. Halliwell, "Complex Random Variables" in "Casualty
               Actuarial Society E-Forum", Fall 2015.

        Examples
        --------
        Draw samples from the standard multivariate complex normal

        >>> from randomgen import ExtendedGenerator
        >>> eg = ExtendedGenerator()
        >>> loc = np.zeros(3)
        >>> eg.multivariate_complex_normal(loc, size=2)
        array([[ 0.42551611+0.44163456j,
                -0.18366146+0.88380663j,
                -0.3035725 -1.19754723j],
               [-0.86649667-0.88447445j,
                -0.04913229-0.04674949j,
                -0.28145563+1.04682163j]])

        Draw samples a trivariate centered circularly symmetric complex normal

        >>> rho = 0.7
        >>> gamma = rho * np.eye(3) + (1-rho) * np.diag(np.ones(3))
        >>> eg.multivariate_complex_normal(loc, gamma, size=3)
        array([[ 0.32699266-0.57787275j,  0.46716898-0.06687298j,
                -0.31483301+0.17233599j],
               [ 0.28036548-0.56994348j,  0.18011468-0.50539209j,
                 0.35185607-0.15184288j],
               [-0.1866397 +1.2701576j , -0.18419364-0.06912343j,
                -0.66462037+0.73939778j]])

        Draw samples from a bivariate distribution with
        correlation between the real and imaginary components

        >>> loc = np.array([3-7j, 2+4j])
        >>> gamma = np.array([[2, 0 + 1.0j], [-0 - 1.0j, 2]])
        >>> rel = np.array([[-1.8, 0 + 0.1j], [0 + 0.1j, -1.8]])
        >>> eg.multivariate_complex_normal(loc, gamma, size=3)
        array([[2.97279918-5.64185732j, 2.32361134+3.23587346j],
               [1.91476019-7.91802901j, 1.76788821+3.84832672j],
               [4.44740101-7.93782402j, 1.59809459+1.35360097j]])
        """
        cdef np.ndarray garr, rarr, larr
        cdef np.npy_intp *gshape
        cdef np.npy_intp *rshape
        cdef int gdim, rdim, dim, ldim

        larr = <np.ndarray>np.PyArray_FROM_OTF(loc,
                                               np.NPY_CDOUBLE,
                                               np.NPY_ARRAY_ALIGNED |
                                               np.NPY_ARRAY_C_CONTIGUOUS)
        ldim = np.PyArray_NDIM(larr)
        if ldim < 1 or larr.size == 0:
            raise ValueError("loc must be non-empty and at least 1-dimensional")
        dim = np.PyArray_DIMS(larr)[ldim - 1]

        if gamma is None:
            garr = <np.ndarray>np.eye(dim, dtype=complex)
        else:
            garr = <np.ndarray>np.PyArray_FROM_OTF(gamma,
                                                   np.NPY_CDOUBLE,
                                                   np.NPY_ARRAY_ALIGNED |
                                                   np.NPY_ARRAY_C_CONTIGUOUS)

        gdim = np.PyArray_NDIM(garr)
        gshape = np.PyArray_DIMS(garr)
        if (
                gdim < 2 or
                gshape[gdim - 2] != gshape[gdim - 1] or
                gshape[gdim - 1] != dim or
                garr.size == 0
        ):
            raise ValueError(
                "gamma must be non-empty with at least 2-dimensional and the "
                "final two dimensions must match the final dimension of loc,"
                f" {dim}."
            )
        if relation is None:
            rarr = <np.ndarray>np.zeros((dim, dim), dtype=complex)
        else:
            rarr = <np.ndarray>np.PyArray_FROM_OTF(relation,
                                                   np.NPY_CDOUBLE,
                                                   np.NPY_ARRAY_ALIGNED |
                                                   np.NPY_ARRAY_C_CONTIGUOUS)
        rdim = np.PyArray_NDIM(rarr)
        rshape = np.PyArray_DIMS(rarr)
        if (
                rdim < 2 or
                rshape[rdim - 2] != rshape[rdim - 1] or
                rshape[rdim - 1] != dim or
                rarr.size == 0
        ):
            raise ValueError(
                "relation must be non-empty with at least 2-dimensional and the "
                "final two dimensions must match the final dimension of loc,"
                f" {dim}."
            )
        can_bcast, cov_shape = broadcast_shape(np.shape(garr), np.shape(rarr), False)
        if not can_bcast:
            raise ValueError(
                f"The leading dimensions of gamma {np.shape(garr)[:gdim-2]} "
                "must broadcast with the leading dimension of relation "
                f"{np.shape(rarr)[:rdim-2]}.")
        common_shape = cov_shape[: len(cov_shape) - 2]
        l_shape = np.shape(larr)
        l_common = l_shape[: len(l_shape) - 1]
        can_bcast, bcast_shape = broadcast_shape(l_common, common_shape, False)
        if size is not None:
            if isinstance(size, (int, np.integer)):
                size = (size, )
            can_bcast, bcast_shape = broadcast_shape(tuple(size), common_shape, True)
        temp = np.empty((2 * dim, 2 * dim))
        p = np.arange(2 * dim).reshape((2, -1))
        p = p.T.ravel()
        ix = np.ix_(p, p)

        if gdim == 2:
            gidx = np.array([0])
            garr = np.reshape(garr, (1,) + np.shape(garr))
        else:
            _shape = np.shape(garr)[: gdim - 2]
            gidx = np.arange(np.prod(_shape)).reshape(_shape)
        if rdim == 2:
            ridx = np.array([0])
            rarr = np.reshape(rarr, (1,) + np.shape(rarr))
        else:
            _shape = np.shape(rarr)[: rdim - 2]
            ridx = np.arange(np.prod(_shape)).reshape(_shape)

        factors = np.empty(common_shape + (2 * dim, 2 * dim)).reshape(
            (-1, 2 * dim, 2 * dim)
        )
        fidx = 0

        for i, j, in np.broadcast(gidx, ridx):
            g = garr[i]
            r = rarr[j]
            gpr = 0.5 * (g + r)
            gmr = 0.5 * (g - r)
            temp[:dim, :dim] = gpr.real
            temp[:dim, dim:] = -gmr.imag
            temp[dim:, :dim] = gpr.imag
            temp[dim:, dim:] = gmr.real
            if not np.allclose(temp, temp.T, rtol=tol):
                raise ValueError(
                    "The covariance matrix implied by gamma and relation is "
                    "not symmetric.  Each component in gamma must be positive "
                    "semi-definite Hermetian and each component in relation "
                    "must be symmetric."
                )
            factors[fidx] = _factorize(
                temp[ix], meth=method, check_valid=check_valid, tol=tol, rank=2 * dim
            )
            fidx += 1
        factors = factors.reshape(common_shape + (2 * dim, 2 * dim))
        out = self.multivariate_normal(
            larr.view(np.float64), factors, size=size, method="factor"
        )
        return out.view(complex)
