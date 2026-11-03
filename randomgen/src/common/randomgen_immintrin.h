#ifndef _RANDOMGEN__IMMINTRIN_H_
#define _RANDOMGEN__IMMINTRIN_H_

#if (defined(_M_IX86_FP) && _M_IX86_FP >= 2) || defined(_M_X64)
#if !defined(__SSE2__)
#define __SSE2__ 1
#endif
#if defined(_MSC_VER) && defined(_WIN32) && _MSC_VER >= 1900
#if !defined(__SSSE3__)
#define __SSSE3__ 1
#endif
#if !defined(__AES__)
#define __AES__ 1
#endif
#if !defined(__RDRND__)
#define __RDRND__ 1
#endif
#endif
#endif

/*
 * AVX2 is only enabled for functions marked RANDOMGEN_AVX2_TARGET so that
 * the rest of a module runs on CPUs without AVX2. These functions must only
 * be called after avx2_capable() returns 1.
 */
#if defined(__x86_64__) || defined(_M_X64)
#define RANDOMGEN_HAVE_AVX2 1
#if defined(__GNUC__) || defined(__clang__)
#define RANDOMGEN_AVX2_TARGET __attribute__((target("avx2")))
#else
#define RANDOMGEN_AVX2_TARGET
#endif
#endif

/*
 * The same applies to SSSE3, which ChaCha and SPECK-128 use after checking
 * the CPU. Enabling it for a whole module lets the compiler use it anywhere,
 * so the module would not load on an x86-64 CPU without SSSE3.
 */
#if defined(__x86_64__) || defined(_M_X64)
#define RANDOMGEN_HAVE_SSSE3 1
#if defined(__GNUC__) || defined(__clang__)
#define RANDOMGEN_SSSE3_TARGET __attribute__((target("ssse3")))
#else
#define RANDOMGEN_SSSE3_TARGET
#endif
#endif

#if defined(__SSE2__) && __SSE2__
#include <emmintrin.h>
#endif

#if (defined(__SSSE3__) && __SSSE3__) || (defined(__AES__) && __AES__) ||      \
    (defined(__RDRND__) && __RDRND__) || defined(RANDOMGEN_HAVE_AVX2)
#include <immintrin.h>
#endif

#endif /*  _RANDOMGEN__IMMINTRIN_H_ */
