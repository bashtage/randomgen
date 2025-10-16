#ifndef _RANDOMGEN_CPU_FEATURES_H_
#define _RANDOMGEN_CPU_FEATURES_H_

#include "randomgen_config.h"

#if defined(__APPLE__) && defined(__x86_64__)
#include <cpuid.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #if defined(_MSC_VER)
    /* MSVC on Windows uses <intrin.h> for __cpuidex and _xgetbv */
    #include <intrin.h>
  #elif (defined(__GNUC__) || defined(__clang__))
    /* GCC/Clang requires <immintrin.h> for _xgetbv. */
    #include <immintrin.h>
  #endif
#endif


#define RANDOMGEN_EAX 0
#define RANDOMGEN_EBX 1
#define RANDOMGEN_ECX 2
#define RANDOMGEN_EDX 3

#undef HAVE_CPUID
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#if defined(_MSC_VER) && defined(_WIN32)
#if _MSC_VER >= 1500
#define HAVE_CPUID 1
#endif
#else
#define HAVE_CPUID 1
#include <cpuid.h>
#endif
#endif

void feature_flags(int flags[32], int major);
int avx2_capable(void);
#endif /* _RANDOMGEN_CPU_FEATURES_H */