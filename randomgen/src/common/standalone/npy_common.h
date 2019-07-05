#ifndef _NPY_COMMON_H_
#define _NPY_COMMON_H_

#include "../randomgen_config.h"

#include "Python.h"

#include <stdint.h>

#define NPY_INLINE INLINE

#define NPY_SIZEOF_LONG SIZEOF_LONG

typedef double npy_double;

typedef uint32_t npy_uint32;

typedef unsigned char npy_bool;
#define NPY_FALSE 0
#define NPY_TRUE 1

typedef Py_intptr_t npy_intp;
typedef Py_uintptr_t npy_uintp;

#endif
