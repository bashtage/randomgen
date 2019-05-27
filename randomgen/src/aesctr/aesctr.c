#include "aesctr.h"

#define AES_FEATURE_FLAG 25

#if (defined(__clang__) || defined(__GNUC__)) && defined(HAVE_IMMINTRIN)
#include <cpuid.h>
#endif

int aes_capable(void)
{
    int status = 0;
#if defined(HAVE_IMMINTRIN)
#if defined(__clang__) || defined(__GNUC__)
    uint32_t eax, ebx, ecx, edx, num_ids;
    __cpuid(0, num_ids, ebx, ecx, edx);
    ecx = 0;
    if (num_ids >= 1)
    {
        __cpuid(1, eax, ebx, ecx, edx);
    }
    status = (ecx >> AES_FEATURE_FLAG) & 0x1;
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
    for (int i = 0; i < 32; i++)
        status = (ecx >> AES_FEATURE_FLAG) & 0x1;
#endif
#endif
    return status;
}

void aesctr_seed(aesctr_state_t *aesctr, uint64_t *seed)
{
#if defined(HAVE_SSE2)
    aesctr_seed_r(aesctr, seed);
#endif
}

void aesctr_get_seed_counter(aesctr_state_t *aesctr, uint64_t *seed,
                             uint64_t *counter)
{
#if defined(HAVE_SSE2)
    memcpy(seed, &aesctr->seed, (AESCTR_ROUNDS + 1) * sizeof(__m128i));
    memcpy(counter, &aesctr->ctr, AESCTR_UNROLL * sizeof(__m128i));
#endif
}

void aesctr_set_counter(aesctr_state_t *aesctr, uint64_t *counter)
{
#if defined(HAVE_SSE2)
    memcpy(&aesctr->ctr, counter, AESCTR_UNROLL * sizeof(__m128i));
#endif
}

void aesctr_set_seed_counter(aesctr_state_t *aesctr, uint64_t *seed,
                             uint64_t *counter)
{
#if defined(HAVE_SSE2)
    memcpy(&aesctr->seed, seed, (AESCTR_ROUNDS + 1) * sizeof(__m128i));
    aesctr_set_counter(aesctr, counter);
#endif
}

void aesctr_advance(aesctr_state_t *aesctr, uint64_t *step)
{
#if defined(HAVE_SSE2)
    uint64_t low;
    uint64_t temp[2];
    int i;

    for (i = 0; i < AESCTR_UNROLL; i++)
    {
        memcpy(&temp, &aesctr->ctr[i], sizeof(__m128i));
        low = temp[0];
        temp[0] += step[0];
        temp[1] += step[1];
        if (temp[0] < low)
        {
            temp[1] + 1;
        }
        memcpy(&aesctr->ctr[i], &temp, sizeof(__m128i));
    }
    aesctr->offset = 16 * AESCTR_UNROLL;
#endif
}

extern INLINE uint64_t aes_next64(aesctr_state_t *aesctr);

extern INLINE uint32_t aes_next32(aesctr_state_t *aesctr);

extern INLINE double aes_next_double(aesctr_state_t *aesctr);
