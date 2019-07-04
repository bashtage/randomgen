#ifndef _RANDOMDGEN__CONFIG_H_
#define _RANDOMDGEN__CONFIG_H_

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) && defined(_MSC_VER)

/* windows msvc */

#ifndef MSVCFORCEINLINE
#ifndef _DEBUG
#define MSVCFORCEINLINE __forceinline
#else
#define MSVCFORCEINLINE
#endif
#endif

#ifndef inline
#if _MSC_VER < 1600
/* msvs 2008 and earlier */
#define inline _inline MSVCFORCEINLINE
#else
/* msvs 2010 and later */
#define inline __inline MSVCFORCEINLINE
#endif
#define INLINE inline
#else
#define INLINE __inline MSVCFORCEINLINE
#endif

#if _MSC_VER < 1600

/* msvs 2008 and earlier */
#include "inttypes.h"
#include "stdbool.h"
#include "stdint.h"

#elif _MSC_VER < 1800

/* msvs 2010-2012 */
#include "inttypes.h"
#include "stdbool.h"
#include <stdint.h>

#else

/* msvs 2013 and later */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#endif

#define ALIGN_WINDOWS __declspec(align(16))
#define ALIGN_GCC_CLANG

#else

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#define INLINE inline

#define ALIGN_WINDOWS
#define ALIGN_GCC_CLANG __attribute__((aligned(16)))

#endif

#if defined(__MINGW32__)
#include <x86intrin.h>
#endif

#ifdef _WIN32
#define UNLIKELY(x) ((x))
#define LIKELY(x) ((x))
#else
#define UNLIKELY(x) (__builtin_expect((x), 0))
#define LIKELY(x) (__builtin_expect((x), 1))
#endif

#endif
