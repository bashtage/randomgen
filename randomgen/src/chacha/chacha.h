/*
    Copyright (c) 2015 Orson Peters <orsonpeters@gmail.com>

    This software is provided 'as-is', without any express or implied warranty. In no event will the
    authors be held liable for any damages arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose, including commercial
    applications, and to alter it and redistribute it freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not claim that you wrote the
       original software. If you use this software in a product, an acknowledgment in the product
       documentation would be appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be misrepresented as
       being the original software.

    3. This notice may not be removed or altered from any source distribution.
*/

#ifndef _RANDOMDGEN__CHACHA_H_
#define _RANDOMDGEN__CHACHA_H_


#include "../common/randomgen_config.h"
#include "../common/randomgen_immintrin.h"

#if defined(_WIN32) && defined(_MSC_VER)
#define M128I_CAST
#else
#define M128I_CAST (__m128i)
#endif

extern int RANDOMGEN_USE_SIMD;


typedef double * aligned_double_ptr ;

ALIGN_WINDOWS struct CHACHA_STATE_T {
    ALIGN_WINDOWS uint32_t block[16] ALIGN_GCC_CLANG;
    uint32_t keysetup[8];
    uint64_t ctr[2];
    int rounds;
} ALIGN_GCC_CLANG;

typedef struct CHACHA_STATE_T chacha_state_t;


#if defined(__SSE2__) && __SSE2__
// Get an efficient _mm_roti_epi32 based on enabled features.
#if !defined(__XOP__)
    #if defined(__SSSE3__) && __SSSE3__
        #undef _mm_roti_epi32 /* Silence warnings on some compiler */
        #define _mm_roti_epi32(r, c) (                              \
            ((c) == 8) ?                                            \
                _mm_shuffle_epi8((r), _mm_set_epi8(14, 13, 12, 15,  \
                                                   10,  9,  8, 11,  \
                                                    6,  5,  4,  7,  \
                                                    2,  1,  0,  3)) \
            : ((c) == 16) ?                                         \
                _mm_shuffle_epi8((r), _mm_set_epi8(13, 12, 15, 14,  \
                                                    9,  8, 11, 10,  \
                                                    5,  4,  7,  6,  \
                                                    1,  0,  3,  2)) \
            : ((c) == 24) ?                                         \
                _mm_shuffle_epi8((r), _mm_set_epi8(12, 15, 14, 13,  \
                                                    8, 11, 10,  9,  \
                                                    4,  7,  6,  5,  \
                                                    0,  3,  2,  1)) \
            :                                                       \
                _mm_xor_si128(_mm_slli_epi32((r), (c)),             \
                              _mm_srli_epi32((r), 32-(c)))          \
        )
    #else
        #undef _mm_roti_epi32 /* Silence warnings on some compiler */
        #define _mm_roti_epi32(r, c) _mm_xor_si128(_mm_slli_epi32((r), (c)), \
                                                   _mm_srli_epi32((r), 32-(c)))
    #endif
#else
    #include <xopintrin.h>
#endif

static INLINE void chacha_core_ssse3(chacha_state_t *state) {
    // ROTVn rotates the elements in the given vector n places to the left.
    int i;

    #define CHACHA_ROTV1(x) _mm_shuffle_epi32(M128I_CAST x, 0x39)
    #define CHACHA_ROTV2(x) _mm_shuffle_epi32(M128I_CAST x, 0x4e)
    #define CHACHA_ROTV3(x) _mm_shuffle_epi32(M128I_CAST x, 0x93)

    __m128i a = _mm_load_si128((__m128i*) (&state->block[0]));
    __m128i b = _mm_load_si128((__m128i*) (&state->block[4]));
    __m128i c = _mm_load_si128((__m128i*) (&state->block[8]));
    __m128i d = _mm_load_si128((__m128i*) (&state->block[12]));

    for (i = 0; i < state->rounds; i += 2) {
        a = _mm_add_epi32(a, b);
        d = _mm_xor_si128(d, a);
        d = _mm_roti_epi32(d, 16);
        c = _mm_add_epi32(c, d);
        b = _mm_xor_si128(b, c);
        b = _mm_roti_epi32(b, 12);
        a = _mm_add_epi32(a, b);
        d = _mm_xor_si128(d, a);
        d = _mm_roti_epi32(d, 8);
        c = _mm_add_epi32(c, d);
        b = _mm_xor_si128(b, c);
        b = _mm_roti_epi32(b, 7);

        b = CHACHA_ROTV1(b);
        c = CHACHA_ROTV2(c);
        d = CHACHA_ROTV3(d);

        a = _mm_add_epi32(a, b);
        d = _mm_xor_si128(d, a);
        d = _mm_roti_epi32(d, 16);
        c = _mm_add_epi32(c, d);
        b = _mm_xor_si128(b, c);
        b = _mm_roti_epi32(b, 12);
        a = _mm_add_epi32(a, b);
        d = _mm_xor_si128(d, a);
        d = _mm_roti_epi32(d, 8);
        c = _mm_add_epi32(c, d);
        b = _mm_xor_si128(b, c);
        b = _mm_roti_epi32(b, 7);

        b = CHACHA_ROTV3(b);
        c = CHACHA_ROTV2(c);
        d = CHACHA_ROTV1(d);
    }

    _mm_store_si128((__m128i*) (&state->block[0]), a);
    _mm_store_si128((__m128i*) (&state->block[4]), b);
    _mm_store_si128((__m128i*) (&state->block[8]), c);
    _mm_store_si128((__m128i*) (&state->block[12]), d);

    #undef CHACHA_ROTV3
    #undef CHACHA_ROTV2
    #undef CHACHA_ROTV1
}
#endif

static INLINE void chacha_core(chacha_state_t *state) {
    int i;
    #define CHACHA_ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

    #define CHACHA_QUARTERROUND(x, a, b, c, d) \
        x[a] = x[a] + x[b]; x[d] ^= x[a]; x[d] = CHACHA_ROTL32(x[d], 16); \
        x[c] = x[c] + x[d]; x[b] ^= x[c]; x[b] = CHACHA_ROTL32(x[b], 12); \
        x[a] = x[a] + x[b]; x[d] ^= x[a]; x[d] = CHACHA_ROTL32(x[d],  8); \
        x[c] = x[c] + x[d]; x[b] ^= x[c]; x[b] = CHACHA_ROTL32(x[b],  7)

    for (i = 0; i < state->rounds ; i += 2) {
        CHACHA_QUARTERROUND(state->block, 0, 4,  8, 12);
        CHACHA_QUARTERROUND(state->block, 1, 5,  9, 13);
        CHACHA_QUARTERROUND(state->block, 2, 6, 10, 14);
        CHACHA_QUARTERROUND(state->block, 3, 7, 11, 15);
        CHACHA_QUARTERROUND(state->block, 0, 5, 10, 15);
        CHACHA_QUARTERROUND(state->block, 1, 6, 11, 12);
        CHACHA_QUARTERROUND(state->block, 2, 7,  8, 13);
        CHACHA_QUARTERROUND(state->block, 3, 4,  9, 14);
    }

    #undef CHACHA_QUARTERROUND
    #undef CHACHA_ROTL32
}

static INLINE void generate_block(chacha_state_t *state) {
    int i;
    uint32_t constants[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};

    uint32_t input[16];
    for (i = 0; i < 4; ++i) input[i] = constants[i];
    for (i = 0; i < 8; ++i) input[4 + i] = state->keysetup[i];
    // Using a 128-bit counter.
    input[12] = (state->ctr[0] / 16) & 0xffffffffu;
    // Carry from the top part of ctr
    input[13] = (((uint32_t)(state->ctr[1]) % 16) << 28) | ((state->ctr[0] / 16) >> 32);
    input[14] = (state->ctr[1] / 16) & 0xffffffffu;
    input[15] = (state->ctr[1] / 16) >> 32;

    for (i = 0; i < 16; ++i) state->block[i] = input[i];
#if defined(__SSE2__) && __SSE2__
    if LIKELY(RANDOMGEN_USE_SIMD > 0) {
        chacha_core_ssse3(state);
    } else {
#endif
        chacha_core(state);
#if defined(__SSE2__) && __SSE2__
    }
#endif
    for (i = 0; i < 16; ++i) state->block[i] += input[i];
}

static INLINE uint32_t chacha_next32(chacha_state_t *state){
    int idx = state->ctr[0] % 16;
    if UNLIKELY(idx == 0) generate_block(state);
    ++state->ctr[0];
    if (state->ctr[0] == 0) ++state->ctr[1];

    return state->block[idx];
}

static INLINE uint64_t chacha_next64(chacha_state_t *state){
    uint64_t out =  chacha_next32(state) | ((uint64_t)chacha_next32(state) << 32);
    return out;
}

static INLINE double chacha_next_double(chacha_state_t *state){
    return (chacha_next64(state) >> 11) * (1.0/9007199254740992.0);
}

void chacha_use_simd(int flag);
int chacha_simd_capable(void);
void chacha_seed(chacha_state_t *state, uint64_t *seedval, uint64_t *stream, uint64_t *ctr);
void chacha_advance(chacha_state_t *state, uint64_t *delta);

#endif /* _RANDOMDGEN__CHACHA_H_ */
