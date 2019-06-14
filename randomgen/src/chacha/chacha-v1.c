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

#include <stdint.h>
#include <limits.h>

#define _ROUNDS 20

struct _STATE_T {
    //alignas(16) uint32_t block[16];
    uint32_t block[16];
    uint32_t keysetup[8];
    uint64_t ctr[2];
};

typedef struct _STATE_T _state_t;

void seed(_state_t *state, uint64_t *seedval, uint64_t *stream) {
    state->ctr[0] = state->ctr[1] = 0;
    // Using a 128-bit seed.
    state->keysetup[0] = seedval[0] & 0xffffffffu;
    state->keysetup[1] = seedval[0] >> 32;
    state->keysetup[2] = seedval[1] & 0xffffffffu;
    state->keysetup[3] = seedval[1] >> 32;
    state->keysetup[4] = stream & 0xffffffffu;
    state->keysetup[5] = stream >> 32;
    state->keysetup[6] = state->keysetup[7] = 0xdeadbeef;      // Could use 128-bit stream.
}


#ifdef __SSE2__
#include "emmintrin.h"

// Get an efficient _mm_roti_epi32 based on enabled features.
#if !defined(__XOP__)
    #if defined(__SSSE3__)
        #include <tmmintrin.h>
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
        #define _mm_roti_epi32(r, c) _mm_xor_si128(_mm_slli_epi32((r), (c)), \
                                                   _mm_srli_epi32((r), 32-(c)))
    #endif
#else
    #include <xopintrin.h>
#endif


inline void chacha_core(_state_t *state) {
    // ROTVn rotates the elements in the given vector n places to the left.
    #define CHACHA_ROTV1(x) _mm_shuffle_epi32((__m128i) x, 0x39)
    #define CHACHA_ROTV2(x) _mm_shuffle_epi32((__m128i) x, 0x4e)
    #define CHACHA_ROTV3(x) _mm_shuffle_epi32((__m128i) x, 0x93)

    __m128i a = _mm_load_si128((__m128i*) (state->block));
    __m128i b = _mm_load_si128((__m128i*) (state->block + 4));
    __m128i c = _mm_load_si128((__m128i*) (state->block + 8));
    __m128i d = _mm_load_si128((__m128i*) (state->block + 12));

    for (int i = 0; i < _ROUNDS; i += 2) {
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

    _mm_store_si128((__m128i*) (state->block), a);
    _mm_store_si128((__m128i*) (state->block + 4), b);
    _mm_store_si128((__m128i*) (state->block + 8), c);
    _mm_store_si128((__m128i*) (state->block + 12), d);

    #undef CHACHA_ROTV3
    #undef CHACHA_ROTV2
    #undef CHACHA_ROTV1
}
#else
inline void chacha_core(_state_t *state) {
    #define CHACHA_ROTL32(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

    #define CHACHA_QUARTERROUND(x, a, b, c, d) \
        x[a] = x[a] + x[b]; x[d] ^= x[a]; x[d] = CHACHA_ROTL32(x[d], 16); \
        x[c] = x[c] + x[d]; x[b] ^= x[c]; x[b] = CHACHA_ROTL32(x[b], 12); \
        x[a] = x[a] + x[b]; x[d] ^= x[a]; x[d] = CHACHA_ROTL32(x[d],  8); \
        x[c] = x[c] + x[d]; x[b] ^= x[c]; x[b] = CHACHA_ROTL32(x[b],  7)

    for (int i = 0; i < _ROUNDS; i += 2) {
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
#endif

inline void generate_block(_state_t *state) {
    uint32_t constants[4] = {0x61707865, 0x3320646e, 0x79622d32, 0x6b206574};

    uint32_t input[16];
    for (int i = 0; i < 4; ++i) input[i] = constants[i];
    for (int i = 0; i < 8; ++i) input[4 + i] = state->keysetup[i];
    input[12] = (state->ctr / 16) & 0xffffffffu;
    input[13] = (state->ctr / 16) >> 32;
    input[14] = input[15] = 0xdeadbeef; // Could use 128-bit counter.

    for (int i = 0; i < 16; ++i) state->block[i] = input[i];
    chacha_core(state);
    for (int i = 0; i < 16; ++i) state->block[i] += input[i];
}

inline uint32_t next(_state_t *state){
    int idx = state->ctr % 16;
    if (idx == 0) generate_block(state);
    ++state->ctr;

    return state->block[idx];
}

inline void advance(_state_t *state, uint64_t n) {
    int idx = state->ctr % 16;
    state->ctr += n;
    if (idx + n >= 16 && state->ctr % 16 != 0) generate_block(state);
}

