#include "aesctr.h"
#include "../common/cpu_features.h"

#define AES_FEATURE_FLAG 25

int RANDOMGEN_USE_AESNI;

int aes_capable(void)
{
#if defined(__AES__) && __AES__
    int flags[32];
    feature_flags(flags, RANDOMGEN_ECX);
    RANDOMGEN_USE_AESNI = flags[AES_FEATURE_FLAG];
    return RANDOMGEN_USE_AESNI;
#else
    RANDOMGEN_USE_AESNI = 0;
    return 0;
#endif
}

#define AES_ROUND(rcon, index)                                                 \
    do                                                                         \
    {                                                                          \
        __m128i k2 = _mm_aeskeygenassist_si128(k, rcon);                       \
        k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                            \
        k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                            \
        k = _mm_xor_si128(k, _mm_slli_si128(k, 4));                            \
        k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3, 3, 3, 3)));  \
        state->seed[index].m128 = k;                                           \
    } while (0)

void aesctr_seed_r(aesctr_state_t *state, uint64_t *seed)
{
    int i;
    /* Call to ensure  RANDOMGEN_USE_AESNI is assigned*/
    aes_capable();
    /*static const uint8_t rcon[] = {
        0x8d, 0x01, 0x02, 0x04,
        0x08, 0x10, 0x20, 0x40,
        0x80, 0x1b, 0x36
    };*/
    if (RANDOMGEN_USE_AESNI)
    {
#if defined(__AES__) && __AES__
        __m128i k = _mm_set_epi64x(seed[1], seed[0]);
        state->seed[0].m128 = k;
        // D. Lemire manually unrolled following loop since
        // _mm_aeskeygenassist_si128 requires immediates
        /*for(int i = 1; i <= AESCTR_ROUNDS; ++i)
        {
            __m128i k2 = _mm_aeskeygenassist_si128(k, rcon[i]);
            k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
            k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
            k = _mm_xor_si128(k, _mm_slli_si128(k, 4));
            k = _mm_xor_si128(k, _mm_shuffle_epi32(k2, _MM_SHUFFLE(3,3,3,3)));
            state->seed[i] = k;
        }*/
        AES_ROUND(0x01, 1);
        AES_ROUND(0x02, 2);
        AES_ROUND(0x04, 3);
        AES_ROUND(0x08, 4);
        AES_ROUND(0x10, 5);
        AES_ROUND(0x20, 6);
        AES_ROUND(0x40, 7);
        AES_ROUND(0x80, 8);
        AES_ROUND(0x1b, 9);
        AES_ROUND(0x36, 10);
        for (i = 0; i < AESCTR_UNROLL; ++i)
        {
            state->ctr[i].m128 = _mm_set_epi64x(0, i);
        }
#endif
    }
    else
    {
        for (i = 0; i < AESCTR_UNROLL; ++i)
        {
        /* TODO: Counter setting need to be adjusted for BE:
            always store as LE, but do math in math in native
            **might be right, but needs check**
        */

            state->ctr[i].u64[0] = 0;
            state->ctr[i].u64[1] = 0;
            /* Always set first byte to deal with endianness */
            state->ctr[i].u8[0] = i;
        }
#if defined(RANDOMGEN_LITTLE_ENDIAN) && !(RANDOMGEN_LITTLE_ENDIAN)
        /* Need to apply a byte swap on seed here so that the bytes are the same as LE
           128-bit seed
           We don't do in place to be careful
        */
        uint64_t bwap_seed[2] = {bswap_64(seed[0]), bswap_64(seed[1])};
        seed = (uint64_t *)&bwap_seed;
#endif
        tinyaes_expand_key((uint8_t *)&state->seed, (uint8_t *)seed);
    }
    state->offset = 16 * AESCTR_UNROLL;
}

#undef AES_ROUND

void aesctr_use_aesni(int val) { RANDOMGEN_USE_AESNI = val; }

void aesctr_seed(aesctr_state_t *state, uint64_t *seed)
{
    aesctr_seed_r(state, seed);
}

void aesctr_get_seed_counter(aesctr_state_t *state, uint64_t *seed,
                             uint64_t *counter)
{
    memcpy(seed, &state->seed, (AESCTR_ROUNDS + 1) * sizeof(aes128_t));
    memcpy(counter, &state->ctr, AESCTR_UNROLL * sizeof(aes128_t));
#if defined(RANDOMGEN_LITTLE_ENDIAN) && !(RANDOMGEN_LITTLE_ENDIAN)
        /* Need to apply a byte swap on seed here so that the bytes are the same as LE
           128-bit seed
           We don't do in place to be careful
        */
        for (int i=0;i<(2*(AESCTR_ROUNDS + 1));i++){
            seed[i] = bswap_64(seed[i]);
        }
#endif
}

void aesctr_set_counter(aesctr_state_t *state, uint64_t *counter)
{
    memcpy(&state->ctr, counter, AESCTR_UNROLL * sizeof(aes128_t));
}

void aesctr_set_seed_counter(aesctr_state_t *state, uint64_t *seed,
                             uint64_t *counter)
{
#if defined(RANDOMGEN_LITTLE_ENDIAN) && !(RANDOMGEN_LITTLE_ENDIAN)
        /* Need to apply a byte swap on seed here so that the bytes are the same as LE
           128-bit seed
           We don't do in place to be careful
        */
        uint64_t bwap_seed[2*(AESCTR_ROUNDS + 1)];
        for (int i=0;i<(2*(AESCTR_ROUNDS + 1));i++){
            bwap_seed[i] = bswap_64(seed[i]);
        }
        seed = (uint64_t *)&bwap_seed;
#endif

    memcpy(&state->seed, seed, (AESCTR_ROUNDS + 1) * sizeof(aes128_t));
    aesctr_set_counter(state, counter);
}

void aesctr_advance(aesctr_state_t *state, uint64_t *step)
{
    uint64_t low;
    uint64_t temp[2];
    uint64_t adj_step[2];
    size_t new_offset;
    int i;
    if (state->offset == 64)
    {
        /* Force update and reset the offset to simplify */
        aesctr_r(state);
        state->offset = 0;
    }
    /* Handle odd with buffer update */
    state->offset = state->offset + 8 * (step[0] % 2);
    adj_step[0] = (step[0] / 2) + ((step[1] % 2) << 63);
    adj_step[1] = (step[1] / 2) + ((step[2] % 2) << 63);
    /* Early return if no counter change */
    if ((adj_step[0] == 0) && (adj_step[1] == 0))
    {
        return;
    }
    /* Update the counters to new **next** values */
    /* TODO: These memcpy are not needed since can use .u64 directly */
    for (i = 0; i < AESCTR_UNROLL; i++)
    {
        memcpy(&temp, &state->ctr[i], sizeof(aes128_t));
        low = temp[0];
        temp[0] += adj_step[0];
        temp[1] += adj_step[1];
        if (temp[0] < low)
        {
            temp[1]++;
        };
        memcpy(&state->ctr[i].u64[0], &temp, sizeof(aes128_t));
    }
    /* Subtract 4 to get previous counter, and regenerate */
    for (i = 0; i < AESCTR_UNROLL; i++)
    {
        memcpy(&temp, &state->ctr[i], sizeof(aes128_t));
        low = temp[0];
        temp[0] -= 4;
        if (temp[0] > low)
        {
            temp[1]--;
        } /* Borrow 1 */
        memcpy(&state->ctr[i].u64[0], &temp, sizeof(aes128_t));
    }
    /* Force update */
    new_offset = state->offset;
    state->offset = 64;
    aesctr_r(state);
    /* Reset the offset */
    state->offset = new_offset;
}

extern INLINE uint64_t aes_next64(aesctr_state_t *state);

extern INLINE uint32_t aes_next32(aesctr_state_t *state);

extern INLINE double aes_next_double(aesctr_state_t *state);
