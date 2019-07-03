#ifndef _RANDOMGEN_CPU_FEATURES_H_
#define _RANDOMGEN_CPU_FEATURES_H_

#include "randomgen_config.h"

#undef HAVE_CPUID
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) ||             \
    defined(_M_IX86)
#if defined(_MSC_VER) && defined(_WIN32)
#if _MSC_VER >= 1900
#define HAVE_CPUID 1
#endif
#else
#define HAVE_CPUID 1
#include <cpuid.h>
#endif
#endif

void feature_flags(int flags[]);

#endif /* _RANDOMGEN_CPU_FEATURES_H */