cimport numpy as np

import numpy as np


cdef double LEGACY_POISSON_LAM_MAX = <double>np.iinfo("l").max - np.sqrt(np.iinfo("l").max)*10
cdef double POISSON_LAM_MAX = <double>np.iinfo("int64").max - np.sqrt(np.iinfo("int64").max)*10


cdef int check_array_constraint(np.ndarray val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if np.any(np.logical_and(np.logical_not(np.isnan(val)), np.signbit(val))):
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE or cons == CONS_POSITIVE_NOT_NAN:
        if cons == CONS_POSITIVE_NOT_NAN and np.any(np.isnan(val)):
            raise ValueError(name + " must not be NaN")
        elif np.any(np.less_equal(val, 0)):
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1:
        if not np.all(np.greater_equal(val, 0)) or \
                not np.all(np.less_equal(val, 1)):
            raise ValueError("{0} < 0, {0} > 1 or {0} contains NaNs".format(name))
    elif cons == CONS_BOUNDED_GT_0_1:
        if not np.all(np.greater(val, 0)) or not np.all(np.less_equal(val, 1)):
            raise ValueError("{0} <= 0, {0} > 1 or {0} contains NaNs".format(name))
    elif cons == CONS_GT_1:
        if not np.all(np.greater(val, 1)):
            raise ValueError("{0} <= 1 or {0} contains NaNs".format(name))
    elif cons == CONS_GTE_1:
        if not np.all(np.greater_equal(val, 1)):
            raise ValueError("{0} < 1 or {0} contains NaNs".format(name))
    elif cons == CONS_POISSON:
        if not np.all(np.less_equal(val, POISSON_LAM_MAX)):
            raise ValueError("{0} value too large".format(name))
        elif not np.all(np.greater_equal(val, 0.0)):
            raise ValueError("{0} < 0 or {0} contains NaNs".format(name))
    elif cons == constraint_type.LEGACY_CONS_POISSON:
        if not np.all(np.less_equal(val, LEGACY_POISSON_LAM_MAX)):
            raise ValueError("{0} value too large".format(name))
        elif not np.all(np.greater_equal(val, 0.0)):
            raise ValueError("{0} < 0 or {0} contains NaNs".format(name))

    return 0


cdef int check_constraint(double val, object name, constraint_type cons) except -1:
    if cons == CONS_NON_NEGATIVE:
        if not np.isnan(val) and np.signbit(val):
            raise ValueError(name + " < 0")
    elif cons == CONS_POSITIVE or cons == CONS_POSITIVE_NOT_NAN:
        if cons == CONS_POSITIVE_NOT_NAN and np.isnan(val):
            raise ValueError(name + " must not be NaN")
        elif val <= 0:
            raise ValueError(name + " <= 0")
    elif cons == CONS_BOUNDED_0_1:
        if not (val >= 0) or not (val <= 1):
            raise ValueError("{0} < 0, {0} > 1 or {0} is NaN".format(name))
    elif cons == CONS_BOUNDED_GT_0_1:
        if not val >0 or not val <= 1:
            raise ValueError("{0} <= 0, {0} > 1 or {0} contains NaNs".format(name))
    elif cons == CONS_GT_1:
        if not (val > 1):
            raise ValueError("{0} <= 1 or {0} is NaN".format(name))
    elif cons == CONS_GTE_1:
        if not (val >= 1):
            raise ValueError("{0} < 1 or {0} is NaN".format(name))
    elif cons == CONS_POISSON:
        if not (val >= 0):
            raise ValueError("{0} < 0 or {0} is NaN".format(name))
        elif not (val <= POISSON_LAM_MAX):
            raise ValueError(name + " value too large")
    elif cons == LEGACY_CONS_POISSON:
        if not (val >= 0):
            raise ValueError("{0} < 0 or {0} is NaN".format(name))
        elif not (val <= LEGACY_POISSON_LAM_MAX):
            raise ValueError(name + " value too large")

    return 0

cdef check_output(object out, object dtype, object size, bint require_c_array):
    """
    Check user-supplied output array properties and shape

    Parameters
    ----------
    out : {ndarray, None}
        The array to check.  If None, returns immediately.
    dtype : dtype
        The required dtype of out.
    size : {None, int, tuple[int]}
        The size passed.  If out is an ndarray, verifies that the shape of out
        matches size.
    require_c_array : bool
        Whether out must be a C-array.  If False, out can be either C- or F-
        ordered.  If True, must be C-ordered. In either case, must be
        contiguous, writable, aligned and in native byte-order.
    """
    if out is None:
        return
    cdef np.ndarray out_array = <np.ndarray> out
    if not (np.PyArray_ISCARRAY(out_array) or
            (np.PyArray_ISFARRAY(out_array) and not require_c_array)):
        req = "C-" if require_c_array else ""
        raise ValueError(
            f'Supplied output array must be {req}contiguous, writable, '
            f'aligned, and in machine byte-order.'
        )
    if out_array.dtype != dtype:
        raise TypeError('Supplied output array has the wrong type. '
                        'Expected {0}, got {1}'.format(np.dtype(dtype), out_array.dtype))
    if size is not None:
        try:
            tup_size = tuple(size)
        except TypeError:
            tup_size = tuple([size])
        if tup_size != out.shape:
            raise ValueError('size must match out.shape when used together')

cdef object double_fill(void *func, bitgen_t *state, object size, object lock, object out):
    cdef random_double_fill random_func = (<random_double_fill>func)
    cdef double out_val
    cdef double *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp n

    if size is None and out is None:
        with lock:
            random_func(state, 1, &out_val)
            return out_val

    if out is not None:
        check_output(out, np.float64, size, False)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.double)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <double *>np.PyArray_DATA(out_array)
    with lock, nogil:
        random_func(state, n, out_array_data)
    return out_array

cdef object float_fill(void *func, bitgen_t *state, object size, object lock, object out):
    cdef random_float_0 random_func = (<random_float_0>func)
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None and out is None:
        with lock:
            return random_func(state)

    if out is not None:
        check_output(out, np.float32, size, False)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float32)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <float *>np.PyArray_DATA(out_array)
    with lock, nogil:
        for i in range(n):
            out_array_data[i] = random_func(state)
    return out_array

cdef object float_fill_from_double(void *func, bitgen_t *state, object size, object lock, object out):
    cdef random_double_0 random_func = (<random_double_0>func)
    cdef float *out_array_data
    cdef np.ndarray out_array
    cdef np.npy_intp i, n

    if size is None and out is None:
        with lock:
            return <float>random_func(state)

    if out is not None:
        check_output(out, np.float32, size, False)
        out_array = <np.ndarray>out
    else:
        out_array = <np.ndarray>np.empty(size, np.float32)

    n = np.PyArray_SIZE(out_array)
    out_array_data = <float *>np.PyArray_DATA(out_array)
    with lock, nogil:
        for i in range(n):
            out_array_data[i] = <float>random_func(state)
    return out_array

cdef object cont_broadcast_1(void *func, void *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             object out):

    cdef np.ndarray randoms
    cdef double a_val
    cdef double *randoms_data
    cdef np.broadcast it
    cdef random_double_1 f = (<random_double_1>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None and out is None:
        randoms = <np.ndarray>np.empty(size, np.double)
    elif out is None:
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_DOUBLE)
    else:
        randoms = <np.ndarray>out

    randoms_data = <double *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)
    it = np.PyArray_MultiIterNew2(randoms, a_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_2(void *func, void *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             np.ndarray b_arr, object b_name, constraint_type b_constraint):
    cdef np.ndarray randoms
    cdef double a_val, b_val
    cdef double *randoms_data
    cdef np.broadcast it
    cdef random_double_2 f = (<random_double_2>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != constraint_type.CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.double)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)

    randoms_data = <double *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            randoms_data[i] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_broadcast_3(void *func, void *state, object size, object lock,
                             np.ndarray a_arr, object a_name, constraint_type a_constraint,
                             np.ndarray b_arr, object b_name, constraint_type b_constraint,
                             np.ndarray c_arr, object c_name, constraint_type c_constraint):
    cdef np.ndarray randoms
    cdef double a_val, b_val, c_val
    cdef double *randoms_data
    cdef np.broadcast it
    cdef random_double_3 f = (<random_double_3>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != constraint_type.CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if c_constraint != constraint_type.CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.double)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_DOUBLE)
        randoms = <np.ndarray>np.empty(it.shape, np.double)

    randoms_data = <double *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<double*>np.PyArray_MultiIter_DATA(it, 3))[0]
            randoms_data[i] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont(void *func, void *state, object size, object lock, int narg,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint,
                 object out):

    cdef np.ndarray a_arr, b_arr, c_arr
    cdef double _a = 0.0, _b = 0.0, _c = 0.0
    cdef bint is_scalar = True
    check_output(out, np.float64, size, narg > 0)
    if narg > 0:
        a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, api.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(a_arr) == 0
    if narg > 1:
        b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, api.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
    if narg == 3:
        c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_DOUBLE, api.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(c_arr) == 0

    if not is_scalar:
        if narg == 1:
            return cont_broadcast_1(func, state, size, lock,
                                    a_arr, a_name, a_constraint,
                                    out)
        elif narg == 2:
            return cont_broadcast_2(func, state, size, lock,
                                    a_arr, a_name, a_constraint,
                                    b_arr, b_name, b_constraint)
        else:
            return cont_broadcast_3(func, state, size, lock,
                                    a_arr, a_name, a_constraint,
                                    b_arr, b_name, b_constraint,
                                    c_arr, c_name, c_constraint)

    if narg > 0:
        _a = PyFloat_AsDouble(a)
        if a_constraint != constraint_type.CONS_NONE and is_scalar:
            check_constraint(_a, a_name, a_constraint)
    if narg > 1:
        _b = PyFloat_AsDouble(b)
        if b_constraint != constraint_type.CONS_NONE:
            check_constraint(_b, b_name, b_constraint)
    if narg == 3:
        _c = PyFloat_AsDouble(c)
        if c_constraint != constraint_type.CONS_NONE and is_scalar:
            check_constraint(_c, c_name, c_constraint)

    if size is None and out is None:
        with lock:
            if narg == 0:
                return (<random_double_0>func)(state)
            elif narg == 1:
                return (<random_double_1>func)(state, _a)
            elif narg == 2:
                return (<random_double_2>func)(state, _a, _b)
            elif narg == 3:
                return (<random_double_3>func)(state, _a, _b, _c)

    cdef np.npy_intp i, n
    cdef np.ndarray randoms
    if out is None:
        randoms = <np.ndarray>np.empty(size)
    else:
        randoms = <np.ndarray>out
    n = np.PyArray_SIZE(randoms)

    cdef double *randoms_data = <double *>np.PyArray_DATA(randoms)
    cdef random_double_0 f0
    cdef random_double_1 f1
    cdef random_double_2 f2
    cdef random_double_3 f3

    with lock, nogil:
        if narg == 0:
            f0 = (<random_double_0>func)
            for i in range(n):
                randoms_data[i] = f0(state)
        elif narg == 1:
            f1 = (<random_double_1>func)
            for i in range(n):
                randoms_data[i] = f1(state, _a)
        elif narg == 2:
            f2 = (<random_double_2>func)
            for i in range(n):
                randoms_data[i] = f2(state, _a, _b)
        elif narg == 3:
            f3 = (<random_double_3>func)
            for i in range(n):
                randoms_data[i] = f3(state, _a, _b, _c)

    if out is None:
        return randoms
    else:
        return out

cdef object discrete_broadcast_d(void *func, void *state, object size, object lock,
                                 np.ndarray a_arr, object a_name, constraint_type a_constraint):

    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_d f = (<random_uint_d>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = np.empty(size, np.int64)
    else:
        # randoms = np.empty(np.shape(a_arr), np.double)
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_INT64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew2(randoms, a_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_dd(void *func, void *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                  np.ndarray b_arr, object b_name, constraint_type b_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_dd f = (<random_uint_dd>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)
    if b_constraint != constraint_type.CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.int64)
        # randoms = np.PyArray_SimpleNew(it.nd, np.PyArray_DIMS(it), np.NPY_INT64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<double*>np.PyArray_MultiIter_DATA(it, 2))[0]
            randoms_data[i] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_di(void *func, void *state, object size, object lock,
                                  np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                  np.ndarray b_arr, object b_name, constraint_type b_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_di f = (<random_uint_di>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != constraint_type.CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew2(a_arr, b_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.int64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew3(randoms, a_arr, b_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<double*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 2))[0]
            randoms_data[i] = f(state, a_val, b_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_iii(void *func, void *state, object size, object lock,
                                   np.ndarray a_arr, object a_name, constraint_type a_constraint,
                                   np.ndarray b_arr, object b_name, constraint_type b_constraint,
                                   np.ndarray c_arr, object c_name, constraint_type c_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_iii f = (<random_uint_iii>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if b_constraint != constraint_type.CONS_NONE:
        check_array_constraint(b_arr, b_name, b_constraint)

    if c_constraint != constraint_type.CONS_NONE:
        check_array_constraint(c_arr, c_name, c_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew3(a_arr, b_arr, c_arr)
        randoms = <np.ndarray>np.empty(it.shape, np.int64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew4(randoms, a_arr, b_arr, c_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            b_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 2))[0]
            c_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 3))[0]
            randoms_data[i] = f(state, a_val, b_val, c_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object discrete_broadcast_i(void *func, void *state, object size, object lock,
                                 np.ndarray a_arr, object a_name, constraint_type a_constraint):
    cdef np.ndarray randoms
    cdef int64_t *randoms_data
    cdef np.broadcast it
    cdef random_uint_i f = (<random_uint_i>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None:
        randoms = <np.ndarray>np.empty(size, np.int64)
    else:
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr), np.PyArray_DIMS(a_arr), np.NPY_INT64)

    randoms_data = <int64_t *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)

    it = np.PyArray_MultiIterNew2(randoms, a_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<int64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

# Needs double <vec>, double-double <vec>, double-int64_t<vec>, int64_t <vec>, int64_t-int64_t-int64_t
cdef object disc(void *func, void *state, object size, object lock,
                 int narg_double, int narg_int64,
                 object a, object a_name, constraint_type a_constraint,
                 object b, object b_name, constraint_type b_constraint,
                 object c, object c_name, constraint_type c_constraint):

    cdef double _da = 0, _db = 0
    cdef int64_t _ia = 0, _ib = 0, _ic = 0
    cdef bint is_scalar = True
    if narg_double > 0:
        a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_DOUBLE, api.NPY_ARRAY_ALIGNED)
        is_scalar = is_scalar and np.PyArray_NDIM(a_arr) == 0
        if narg_double > 1:
            b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_DOUBLE, api.NPY_ARRAY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
        elif narg_int64 == 1:
            b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_INT64, api.NPY_ARRAY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
    else:
        if narg_int64 > 0:
            a_arr = <np.ndarray>np.PyArray_FROM_OTF(a, np.NPY_INT64, api.NPY_ARRAY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(a_arr) == 0
        if narg_int64 > 1:
            b_arr = <np.ndarray>np.PyArray_FROM_OTF(b, np.NPY_INT64, api.NPY_ARRAY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(b_arr) == 0
        if narg_int64 > 2:
            c_arr = <np.ndarray>np.PyArray_FROM_OTF(c, np.NPY_INT64, api.NPY_ARRAY_ALIGNED)
            is_scalar = is_scalar and np.PyArray_NDIM(c_arr) == 0

    if not is_scalar:
        if narg_int64 == 0:
            if narg_double == 1:
                return discrete_broadcast_d(func, state, size, lock,
                                            a_arr, a_name, a_constraint)
            elif narg_double == 2:
                return discrete_broadcast_dd(func, state, size, lock,
                                             a_arr, a_name, a_constraint,
                                             b_arr, b_name, b_constraint)
        elif narg_int64 == 1:
            if narg_double == 0:
                return discrete_broadcast_i(func, state, size, lock,
                                            a_arr, a_name, a_constraint)
            elif narg_double == 1:
                return discrete_broadcast_di(func, state, size, lock,
                                             a_arr, a_name, a_constraint,
                                             b_arr, b_name, b_constraint)
        elif narg_int64 == 3:
            return discrete_broadcast_iii(func, state, size, lock,
                                          a_arr, a_name, a_constraint,
                                          b_arr, b_name, b_constraint,
                                          c_arr, c_name, c_constraint)
        else:
            raise NotImplementedError("No vector path available")

    if narg_double > 0:
        _da = PyFloat_AsDouble(a)
        if a_constraint != constraint_type.CONS_NONE and is_scalar:
            check_constraint(_da, a_name, a_constraint)

        if narg_double > 1:
            _db = PyFloat_AsDouble(b)
            if b_constraint != constraint_type.CONS_NONE and is_scalar:
                check_constraint(_db, b_name, b_constraint)
        elif narg_int64 == 1:
            _ib = <int64_t>b
            if b_constraint != constraint_type.CONS_NONE and is_scalar:
                check_constraint(<double>_ib, b_name, b_constraint)
    else:
        if narg_int64 > 0:
            _ia = <int64_t>a
            if a_constraint != constraint_type.CONS_NONE and is_scalar:
                check_constraint(<double>_ia, a_name, a_constraint)
        if narg_int64 > 1:
            _ib = <int64_t>b
            if b_constraint != constraint_type.CONS_NONE and is_scalar:
                check_constraint(<double>_ib, b_name, b_constraint)
        if narg_int64 > 2:
            _ic = <int64_t>c
            if c_constraint != constraint_type.CONS_NONE and is_scalar:
                check_constraint(<double>_ic, c_name, c_constraint)

    if size is None:
        with lock:
            if narg_int64 == 0:
                if narg_double == 0:
                    return (<random_uint_0>func)(state)
                elif narg_double == 1:
                    return (<random_uint_d>func)(state, _da)
                elif narg_double == 2:
                    return (<random_uint_dd>func)(state, _da, _db)
            elif narg_int64 == 1:
                if narg_double == 0:
                    return (<random_uint_i>func)(state, _ia)
                if narg_double == 1:
                    return (<random_uint_di>func)(state, _da, _ib)
            else:
                return (<random_uint_iii>func)(state, _ia, _ib, _ic)

    cdef np.npy_intp i, n
    cdef np.ndarray randoms = <np.ndarray>np.empty(size, np.int64)
    cdef np.int64_t *randoms_data
    cdef random_uint_0 f0
    cdef random_uint_d fd
    cdef random_uint_dd fdd
    cdef random_uint_di fdi
    cdef random_uint_i fi
    cdef random_uint_iii fiii

    n = np.PyArray_SIZE(randoms)
    randoms_data = <np.int64_t *>np.PyArray_DATA(randoms)

    with lock, nogil:
        if narg_int64 == 0:
            if narg_double == 0:
                f0 = (<random_uint_0>func)
                for i in range(n):
                    randoms_data[i] = f0(state)
            elif narg_double == 1:
                fd = (<random_uint_d>func)
                for i in range(n):
                    randoms_data[i] = fd(state, _da)
            elif narg_double == 2:
                fdd = (<random_uint_dd>func)
                for i in range(n):
                    randoms_data[i] = fdd(state, _da, _db)
        elif narg_int64 == 1:
            if narg_double == 0:
                fi = (<random_uint_i>func)
                for i in range(n):
                    randoms_data[i] = fi(state, _ia)
            if narg_double == 1:
                fdi = (<random_uint_di>func)
                for i in range(n):
                    randoms_data[i] = fdi(state, _da, _ib)
        else:
            fiii = (<random_uint_iii>func)
            for i in range(n):
                randoms_data[i] = fiii(state, _ia, _ib, _ic)

    return randoms


cdef object cont_broadcast_1_f(void *func, bitgen_t *state, object size, object lock,
                               np.ndarray a_arr, object a_name, constraint_type a_constraint,
                               object out):

    cdef np.ndarray randoms
    cdef float a_val
    cdef float *randoms_data
    cdef np.broadcast it
    cdef random_float_1 f = (<random_float_1>func)
    cdef np.npy_intp i, n

    if a_constraint != constraint_type.CONS_NONE:
        check_array_constraint(a_arr, a_name, a_constraint)

    if size is not None and out is None:
        randoms = <np.ndarray>np.empty(size, np.float32)
    elif out is None:
        randoms = np.PyArray_SimpleNew(np.PyArray_NDIM(a_arr),
                                       np.PyArray_DIMS(a_arr),
                                       np.NPY_FLOAT32)
    else:
        randoms = <np.ndarray>out

    randoms_data = <float *>np.PyArray_DATA(randoms)
    n = np.PyArray_SIZE(randoms)
    it = np.PyArray_MultiIterNew2(randoms, a_arr)

    with lock, nogil:
        for i in range(n):
            a_val = (<float*>np.PyArray_MultiIter_DATA(it, 1))[0]
            randoms_data[i] = f(state, a_val)

            np.PyArray_MultiIter_NEXT(it)

    return randoms

cdef object cont_f(void *func, bitgen_t *state, object size, object lock,
                   object a, object a_name, constraint_type a_constraint,
                   object out):

    cdef np.ndarray a_arr
    cdef float _a
    cdef bint is_scalar = True
    cdef int requirements = api.NPY_ARRAY_ALIGNED | api.NPY_ARRAY_FORCECAST
    check_output(out, np.float32, size, True)
    a_arr = <np.ndarray>np.PyArray_FROMANY(a, np.NPY_FLOAT32, 0, 0, requirements)
    is_scalar = np.PyArray_NDIM(a_arr) == 0

    if not is_scalar:
        return cont_broadcast_1_f(func, state, size, lock, a_arr, a_name, a_constraint, out)

    _a = <float>PyFloat_AsDouble(a)
    if a_constraint != constraint_type.CONS_NONE:
        check_constraint(_a, a_name, a_constraint)

    if size is None and out is None:
        with lock:
            return (<random_float_1>func)(state, _a)

    cdef np.npy_intp i, n
    cdef np.ndarray randoms
    if out is None:
        randoms = <np.ndarray>np.empty(size, np.float32)
    else:
        randoms = <np.ndarray>out
    n = np.PyArray_SIZE(randoms)

    cdef float *randoms_data = <float *>np.PyArray_DATA(randoms)
    cdef random_float_1 f1 = <random_float_1>func

    with lock, nogil:
        for i in range(n):
            randoms_data[i] = f1(state, _a)

    if out is None:
        return randoms
    else:
        return out
