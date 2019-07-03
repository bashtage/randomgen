#ifndef _RANDOMGEN__IMMINTRIN_H_
#define _RANDOMGEN__IMMINTRIN_H_

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

#endif /*  _RANDOMGEN__IMMINTRIN_H_ */