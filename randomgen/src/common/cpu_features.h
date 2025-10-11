#ifndef _RANDOMGEN_CPU_FEATURES_H_
#define _RANDOMGEN_CPU_FEATURES_H_

#include "randomgen_config.h"

#if defined(_MSC_VER)
    // MSVC on Windows uses <intrin.h> for __cpuidex and _xgetbv
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    // GCC/Clang requires <immintrin.h> for _xgetbv.
    // We define a wrapper for __cpuidex using inline assembly for maximum portability,
    // as the intrinsic name and signature can vary on these compilers.
    #include <immintrin.h>

    /**
     * @brief Cross-platform wrapper for the CPUID instruction.
     */
    /*
    Disable as widely available
    static void __rg_cpuidex(int cpuInfo[4], int leaf, int subleaf) {
        __asm__ __volatile__(
            "cpuid"
            : "=a" (cpuInfo[0]), "=b" (cpuInfo[1]), "=c" (cpuInfo[2]), "=d" (cpuInfo[3])
            : "a" (leaf), "c" (subleaf)
        );
    }
    */
    // _xgetbv is available via _xgetbv(0) when <immintrin.h> is included.
#else
    #error "Unsupported compiler or platform. Cannot access CPUID or XGETBV intrinsics."
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