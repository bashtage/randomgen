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
#define inline __inline MSVCFORCEINLINE
#define INLINE inline
#else
#define INLINE __inline MSVCFORCEINLINE
#endif

/* msvs 2013 and later */
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#define ALIGN_WINDOWS __declspec(align(16))
#define ALIGN_WINDOWS32 __declspec(align(32))
#define ALIGN_GCC_CLANG
#define ALIGN_GCC_CLANG32

#else

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#define INLINE inline

#define ALIGN_WINDOWS
#define ALIGN_WINDOWS32
#define ALIGN_GCC_CLANG __attribute__((aligned(16)))
#define ALIGN_GCC_CLANG32 __attribute__((aligned(32)))

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
