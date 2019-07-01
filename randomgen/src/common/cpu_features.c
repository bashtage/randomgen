#include "cpu_features.h"

void feature_flags(int flags[])
{
    int i;
#if defined(HAVE_CPUID)
#if defined(__clang__) || defined(__GNUC__)
    uint32_t num_ids = 0, eax = 0, ebx = 0, ecx = 0, edx = 0;
    num_ids = __get_cpuid_max(0, &ebx);
    ebx = 0;
    if (num_ids >= 1)
    {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    }
#elif defined(_MSC_VER) && defined(_WIN32)
    int cpu_info[4];
    int num_ids, ecx = 0;
    __cpuid(&cpu_info, 0);
    num_ids = (int)cpu_info[0];
    if (num_ids >= 1)
    {
        __cpuidex(cpu_info, 1, 0);
        ecx = cpu_info[2];
    }
#endif
#else
    uint32_t ecx;
    ecx = 0;
#endif
    for (i = 0; i < 32; i++)
    {
        flags[i] = (ecx >> i) & 0x1;
    }
}
