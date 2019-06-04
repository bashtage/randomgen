#include "rdrand.h"
#include <stdio.h>

#if (defined(__clang__) || defined(__GNUC__)) && defined(HAS_IMMINTRIN)
#include <cpuid.h>
#endif

extern INLINE uint64_t rdrand_next64(rdrand_state* state);
extern INLINE uint32_t rdrand_next32(rdrand_state* state);

int rdrand_capable(void)
{
    int status = 0;
#if defined(HAS_IMMINTRIN)
#if defined(__clang__) || defined(__GNUC__)
    uint32_t eax, ebx, ecx, edx, num_ids;
    __cpuid(0, num_ids, ebx, ecx, edx);
    ecx = 0;
    if (num_ids >= 1)
    {
        __cpuid(1, eax, ebx, ecx, edx);
    }
    status = (ecx >> 30) & 0x1;
#elif defined(_MSC_VER) && defined(_WIN32)
    int cpu_info[4];
    int num_ids, ecx = 0;
    __cpuid(&cpu_info, 0);
    num_ids = cpu_info[0];
    if (num_ids >= 1)
    {
        __cpuidex(cpu_info, 1, 0);
        ecx = cpu_info[2];
    }
    status = (ecx >> 30) & 0x1;
#endif
#endif
    return status;
}

#if 0
int main(int a, char** b)
{
	rdrand_state state;
    int status = rdrand_capable();
    if (status == 1){
        printf("Has RDRAND\n");
        printf("%" PRIu64 "\n", rdrand_next64(&state));
    }
    else{
        printf("Missing RDRAND\n");
    }
}
#endif
