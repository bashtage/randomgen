#ifndef _RANDOMDGEN__CONFIG_H_
#define _RANDOMDGEN__CONFIG_H_

#include <stddef.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
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
        #include "stdint.h"
        #include "inttypes.h"
        #include "stdbool.h"

    #elif _MSC_VER < 1800

        /* msvs 2010-2012 */
        #include <stdint.h>
        #include "inttypes.h"
        #include "stdbool.h"

    #else

        /* msvs 2013 and later */
        #include <stdint.h>
        #include <inttypes.h>
        #include <stdbool.h>

    #endif

    #define M128I_CAST

    #define ALIGN_WINDOWS __declspec(align(16))
    #define ALIGN_GCC_CLANG

    #if _MSC_VER >= 1900 && _M_AMD64
        #include <intrin.h>
        #pragma intrinsic(_umul128)
    #endif

#else

    #include <stdint.h>
    #include <inttypes.h>
    #include <stdbool.h>

    #define INLINE inline

    #define M128I_CAST (__m128i)

    #define ALIGN_WINDOWS
    #define ALIGN_GCC_CLANG __attribute__((aligned(16)))

#endif

#if defined(__MINGW32__)
    #include <x86intrin.h>
#endif

#undef HAVE_IMMINTRIN
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#if defined(_MSC_VER) && defined(_WIN32)
#if _MSC_VER >= 1900
#include <immintrin.h>
#define HAVE_IMMINTRIN 1
#endif
#else
#include <immintrin.h>
#define HAVE_IMMINTRIN 1
#endif
#endif

#ifdef _WIN32
#define UNLIKELY(x) ((x))
#else
#define UNLIKELY(x) (__builtin_expect((x), 0))
#endif

#endif
