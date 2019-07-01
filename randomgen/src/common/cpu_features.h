#if defined(_WIN32) && defined(_MSC_VER) && _MSC_VER == 1500
#include "../common/inttypes.h"
#else
#include <inttypes.h>
#endif

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
