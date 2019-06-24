#include "aesctr.h"
#include <stdio.h>
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

void aesctr_seed(aesctr_state_t *state, uint64_t *seed)
{
#if defined(HAVE_SSE2)
    aesctr_seed_r(state, seed);
#endif
}

void aesctr_get_seed_counter(aesctr_state_t *state, uint64_t *seed,
                             uint64_t *counter)
{
#if defined(HAVE_SSE2)
    memcpy(seed, &state->seed, (AESCTR_ROUNDS + 1) * sizeof(__m128i));
    memcpy(counter, &state->ctr, AESCTR_UNROLL * sizeof(__m128i));
#endif
}

void aesctr_set_counter(aesctr_state_t *state, uint64_t *counter)
{
#if defined(HAVE_SSE2)
    memcpy(&state->ctr, counter, AESCTR_UNROLL * sizeof(__m128i));
#endif
}

void aesctr_set_seed_counter(aesctr_state_t *state, uint64_t *seed,
                             uint64_t *counter)
{
#if defined(HAVE_SSE2)
    memcpy(&state->seed, seed, (AESCTR_ROUNDS + 1) * sizeof(__m128i));
    aesctr_set_counter(state, counter);
#endif
}

void aesctr_advance(aesctr_state_t *state, uint64_t *step) {
#if defined(HAVE_SSE2)
    uint64_t low;
    uint64_t temp[2];
    uint64_t adj_step[2];
    size_t new_offset;
    int i;
    if (state->offset == 64) {
     /* Force update and reset the offset to simplify */
     aesctr_r(state);
     state->offset = 0;
    }
    /* Handle odd with buffer update */
    state->offset = state->offset + 8 * (step[0] % 2);
    adj_step[0] = (step[0] / 2) + ((step[1] % 2) << 63);
    adj_step[1] = (step[1] / 2) + ((step[2] % 2) << 63);
    /* Early return if no counter change */
    if ((adj_step[0] == 0) && (adj_step[1] == 0)) { return; }
    /* Update the counters to new **next** values */
    for (i = 0; i < AESCTR_UNROLL; i++) {
        memcpy(&temp, &state->ctr[i], sizeof(__m128i));
        low = temp[0];
        temp[0] += adj_step[0];
        temp[1] += adj_step[1];
        if (temp[0] < low) { temp[1]++; };
        memcpy(&state->ctr[i].m128, &temp, sizeof(__m128i));
    }
    /* Subtract 4 to get previous counter, and regenerate */
    for (i = 0; i < AESCTR_UNROLL; i++) {
        memcpy(&temp, &state->ctr[i], sizeof(__m128i));
        low = temp[0];
        temp[0] -=4;
        if (temp[0] > low) { temp[1]--; } /* Borrow 1 */
        memcpy(&state->ctr[i].m128, &temp, sizeof(__m128i));
    }
    /* Force update */
    new_offset = state->offset;
    state->offset = 64;
    aesctr_r(state);
    /* Reset the offset */
    state->offset = new_offset;
#endif
}

extern INLINE uint64_t aes_next64(aesctr_state_t *state);

extern INLINE uint32_t aes_next32(aesctr_state_t *state);

extern INLINE double aes_next_double(aesctr_state_t *state);
