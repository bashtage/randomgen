#include "cpu_features.h"

void feature_flags(int flags[32], int major)
{
    int i;
#if defined(HAVE_CPUID) && HAVE_CPUID
#if defined(__clang__) || defined(__GNUC__)
    uint32_t num_ids = 0, reg  = 0, eax = 0, ebx = 0, ecx = 0, edx = 0;
    num_ids = __get_cpuid_max(0, &ebx);
    ebx = 0;
    if (num_ids >= 1)
    {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    }
#elif defined(_MSC_VER) && defined(_WIN32)
    int cpu_info[4] = {0};
    int num_ids, reg = 0, eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid(cpu_info, 0);
    num_ids = (int)cpu_info[0];
    if (num_ids >= 1)
    {
        __cpuidex(cpu_info, 1, 0);
        eax = cpu_info[0];
        ebx = cpu_info[1];
        ecx = cpu_info[2];
        edx = cpu_info[3];
    }
#endif
#else
    uint32_t reg, eax, ebx, ecx, edx;
    reg = 0; eax = 0; ebx = 0; ecx = 0; edx = 0;
#endif
    switch(major){
        case 0:
        reg = eax;
        break;

        case 1:
        reg = ebx;
        break;

        case 2:
        reg = ecx;
        break;

        case 3:
        reg = edx;
        break;
    }
    for (i = 0; i < 32; i++)
    {
        flags[i] = (reg >> i) & 0x1;
    }
}
