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

#if defined(__SSE2__) && __SSE2__
#include <emmintrin.h>
#endif

#if (defined(__SSSE3__) && __SSSE3__) || (defined(__AES__) && __AES__) ||      \
    (defined(__RDRND__) && __RDRND__)
#include <immintrin.h>
#endif

#endif /*  _RANDOMGEN__IMMINTRIN_H_ */
