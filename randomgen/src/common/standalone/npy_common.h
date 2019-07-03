#ifndef _NPY_COMMON_H_
#define _NPY_COMMON_H_

#include "Python.h"

#if defined(_MSC_VER)
    #define NPY_INLINE __inline
#elif defined(__GNUC__)
    #if defined(__STRICT_ANSI__)
         #define NPY_INLINE __inline__
    #else
         #define NPY_INLINE inline
    #endif
#else
    #define NPY_INLINE
#endif

#define NPY_SIZEOF_LONG SIZEOF_LONG

typedef double npy_double;

typedef unsigned long npy_uint32;

typedef unsigned char npy_bool;
#define NPY_FALSE 0
#define NPY_TRUE 1

typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp;

#endif
