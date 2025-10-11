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

#ifndef _RANDOMDGEN__BLABLA_H_
#define _RANDOMDGEN__BLABLA_H_

#include "../common/randomgen_config.h"
#include "../common/randomgen_immintrin.h"

#if defined(_WIN32) && defined(_MSC_VER)
#define M128I_CAST
#else
#define M128I_CAST (__m128i)
#endif

ALIGN_WINDOWS32 struct BLABLA_STATE_T {
    ALIGN_WINDOWS32 uint64_t block[16] ALIGN_GCC_CLANG32;
    uint64_t keysetup[4];
    uint64_t block_idx[2];
    uint64_t ctr[2];
    int rounds;
    int has_uint32;
    uint32_t next_uint32;
} ALIGN_GCC_CLANG32;

typedef struct BLABLA_STATE_T blabla_state_t;

#define R 20 /* Use 20 for now */

extern int RANDOMGEN_USE_AVX2;
extern void blabla_use_avx2(int flag);
extern int blabla_avx2_capable(void);
extern void blabla_seed(blabla_state_t* state, uint64_t seedval[2], uint64_t stream[2], uint64_t ctr[2]);
extern void blabla_advance(blabla_state_t* state, uint64_t delta[2]);
static inline void blabla_core(blabla_state_t* state);
static inline void blabla_core_avx2(blabla_state_t* state);


static inline void blabla_core_avx2(blabla_state_t* state) {
	#define _mm256_roti_epi64(r, c) _mm256_xor_si256(_mm256_srli_epi64((r), (c)), _mm256_slli_epi64((r), 64-(c)))

	// ROTVn rotates the elements in the given vector n places to the left.
	#define CHACHA_ROTV1(x) _mm256_permute4x64_epi64(x, 0x39)
	#define CHACHA_ROTV2(x) _mm256_permute4x64_epi64(x, 0x4e)
	#define CHACHA_ROTV3(x) _mm256_permute4x64_epi64(x, 0x93)
    uint64_t *block = &state->block[0];
	__m256i a = _mm256_load_si256((__m256i*) (block));
	__m256i b = _mm256_load_si256((__m256i*) (block + 4));
	__m256i c = _mm256_load_si256((__m256i*) (block + 8));
	__m256i d = _mm256_load_si256((__m256i*) (block + 12));

	for (uint32_t i = 0; i < state->rounds; ++i) {
		a = _mm256_add_epi64(a, b);
		d = _mm256_xor_si256(d, a);
		d = _mm256_roti_epi64(d, 32);
		c = _mm256_add_epi64(c, d);
		b = _mm256_xor_si256(b, c);
		b = _mm256_roti_epi64(b, 24);
		a = _mm256_add_epi64(a, b);
		d = _mm256_xor_si256(d, a);
		d = _mm256_roti_epi64(d, 16);
		c = _mm256_add_epi64(c, d);
		b = _mm256_xor_si256(b, c);
		b = _mm256_roti_epi64(b, 63);

		b = CHACHA_ROTV1(b);
		c = CHACHA_ROTV2(c);
		d = CHACHA_ROTV3(d);

		a = _mm256_add_epi64(a, b);
		d = _mm256_xor_si256(d, a);
		d = _mm256_roti_epi64(d, 32);
		c = _mm256_add_epi64(c, d);
		b = _mm256_xor_si256(b, c);
		b = _mm256_roti_epi64(b, 24);
		a = _mm256_add_epi64(a, b);
		d = _mm256_xor_si256(d, a);
		d = _mm256_roti_epi64(d, 16);
		c = _mm256_add_epi64(c, d);
		b = _mm256_xor_si256(b, c);
		b = _mm256_roti_epi64(b, 63);

		b = CHACHA_ROTV3(b);
		c = CHACHA_ROTV2(c);
		d = CHACHA_ROTV1(d);
	}
	_mm256_store_si256((__m256i*) (block), a);
	_mm256_store_si256((__m256i*) (block + 4), b);
	_mm256_store_si256((__m256i*) (block + 8), c);
	_mm256_store_si256((__m256i*) (block + 12), d);

	#undef CHACHA_ROTV3
	#undef CHACHA_ROTV2
	#undef CHACHA_ROTV1
	#undef _mm256_roti_epi64
}



inline void blabla_core(blabla_state_t* state) {
	#define rotate_right(x, n) ((x >> n) | (x << (64 - n)))

#define mix_func(a, b, c, d)                             \
    state->block[a] += state->block[b];                  \
    state->block[d] ^= state->block[a];                  \
    state->block[d] = rotate_right(state->block[d], 32); \
    state->block[c] += state->block[d];                  \
    state->block[b] ^= state->block[c];                  \
    state->block[b] = rotate_right(state->block[b], 24); \
    state->block[a] += state->block[b];                  \
    state->block[d] ^= state->block[a];                  \
    state->block[d] = rotate_right(state->block[d], 16); \
    state->block[c] += state->block[d];                  \
    state->block[b] ^= state->block[c];                  \
    state->block[b] = rotate_right(state->block[b], 63);

    for (uint32_t i = 0; i < R; ++i) {
        mix_func(0, 4, 8, 12);
        mix_func(1, 5, 9, 13);
        mix_func(2, 6, 10, 14);
        mix_func(3, 7, 11, 15);
        mix_func(0, 5, 10, 15);
        mix_func(1, 6, 11, 12);
        mix_func(2, 7, 8, 13);
        mix_func(3, 4, 9, 14);
    }
#undef mix_func
#undef rotate_right
}

static inline void generate_block(blabla_state_t* state)
{
    uint64_t constants[4] = { 0x6170786593810fab, 0x3320646ec7398aee, 0x79622d3217318274, 0x6b206574babadada };

    uint64_t input[16];
    for (uint32_t i = 0; i < 4; ++i)
        input[i] = constants[i];
    for (uint32_t i = 0; i < 4; ++i)
        input[4 + i] = state->keysetup[i];
    input[8] = 0x2ae36e593e46ad5f;
    input[9] = 0xb68f143029225fc9;
    input[10] = 0x8da1e08468303aa6;
    input[11] = 0xa48a209acd50a4a7;
    input[12] = 0x7fdc12f23f90778c;
    input[13] = state->block_idx[0] + 1;
    input[14] = state->block_idx[1];
    input[15] = 0; // Could be used for 192-bit counter.

    for (uint32_t i = 0; i < 16; ++i)
        state->block[i] = input[i];
#if defined(__AVX2__) && __AVX2__
    if LIKELY(RANDOMGEN_USE_AVX2 > 0) {
         blabla_core_avx2(state);
    } else {
#endif
        blabla_core(state);
#if defined(__AVX2__) && __AVX2__
    }
#endif




    for (uint32_t i = 0; i < 16; ++i)
        state->block[i] += input[i];
}

static INLINE uint64_t blabla_next64(blabla_state_t* state)
{
    uint64_t next_block_idx[2];
    next_block_idx[0] = ((state->ctr[1] & 0xFULL) << 60) | (state->ctr[0] / 16ULL);
    next_block_idx[1] = state->ctr[1] / 16uLL;
    uint64_t idx_in_block = state->ctr[0] % 16;
    if ((next_block_idx[0] != state->block_idx[0]) || (next_block_idx[1] != state->block_idx[1])) {
        state->block_idx[0] = next_block_idx[0];
        state->block_idx[1] = next_block_idx[1];
        generate_block(state);
    }
    state->ctr[0]++;
    if (state->ctr[0] == 0) {
        state->ctr[1]++;
    };

    return state->block[idx_in_block];
}

static INLINE uint32_t blabla_next32(blabla_state_t* state)
{
    if (state->has_uint32 == 1) {
        state->has_uint32 = 0;
        return state->next_uint32;
    }
    uint64_t val = blabla_next64(state);
    state->next_uint32 = (uint32_t)(val >> 32);
    return (uint32_t)(val & 0xFFFFFFFF);
}

static INLINE double blabla_next_double(blabla_state_t* state)
{
    return (blabla_next64(state) >> 11) * (1.0 / 9007199254740992.0);
}

#endif /* _RANDOMDGEN__BLABLA_H_ */
