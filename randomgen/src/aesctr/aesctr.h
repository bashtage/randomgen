/***
 * Daniel Baker has adapted this code to C++:
 * https://github.com/dnbaker/hll/blob/master/aesctr.h
 * He reports that UNROLL_COUNT of 8 (rather than 4) is measurably
 * but not enormously faster.
 ***/

#ifndef AESCTR_H
#define AESCTR_H
// contributed by Samuel Neves

#include "../common/randomgen_config.h"
#include "../common/randomgen_immintrin.h"
#include "../common/randomgen_endian.h"

#if defined(RANDOMGEN_FORCE_SOFTAES) && RANDOMGEN_FORCE_SOFTAES
#undef __AES__
#endif

#include "softaes.h"

#define AESCTR_UNROLL 4
#define AESCTR_ROUNDS 10

extern int RANDOMGEN_USE_AESNI;

union AES128_T {
#if defined(__AES__) && __AES__
  __m128i m128;
#endif
  uint64_t u64[2];
  uint32_t u32[4];
  uint8_t u8[16];
};

typedef union AES128_T aes128_t;

struct AESCTR_STATE_T {
  ALIGN_WINDOWS aes128_t ctr[AESCTR_UNROLL] ALIGN_GCC_CLANG;
  ALIGN_WINDOWS aes128_t seed[AESCTR_ROUNDS + 1] ALIGN_GCC_CLANG;
  ALIGN_WINDOWS uint8_t state[16 * AESCTR_UNROLL] ALIGN_GCC_CLANG;
  size_t offset;
  int has_uint32;
  uint32_t uinteger;
};

typedef struct AESCTR_STATE_T aesctr_state_t;

static INLINE uint64_t aesctr_r(aesctr_state_t *state) {
  uint64_t output;
  if (UNLIKELY(state->offset >= 16 * AESCTR_UNROLL)) {
    if (RANDOMGEN_USE_AESNI) {
#if defined(__AES__) && __AES__
      __m128i work[AESCTR_UNROLL];
      for (int i = 0; i < AESCTR_UNROLL; ++i) {
        work[i] = _mm_xor_si128(state->ctr[i].m128, state->seed[0].m128);
      }
      for (int r = 1; r <= AESCTR_ROUNDS - 1; ++r) {
        const __m128i subkey = state->seed[r].m128;
        for (int i = 0; i < AESCTR_UNROLL; ++i) {
          work[i] = _mm_aesenc_si128(work[i], subkey);
        }
      }
      for (int i = 0; i < AESCTR_UNROLL; ++i) {
        state->ctr[i].m128 =
            _mm_add_epi64(state->ctr[i].m128, _mm_set_epi64x(0, AESCTR_UNROLL));
        if (UNLIKELY(state->ctr[i].u64[0] < AESCTR_UNROLL)) {
          /* rolled, add carry */
          state->ctr[i].m128 =
              _mm_add_epi64(state->ctr[i].m128, _mm_set_epi64x(1, 0));
        }
        _mm_storeu_si128(
            (__m128i *)&state->state[16 * i],
            _mm_aesenclast_si128(work[i], state->seed[AESCTR_ROUNDS].m128));
      }
      state->offset = 0;
#endif
    } else {
      int i;
      memcpy(&state->state, &state->ctr, sizeof(state->state));

#if defined(RANDOMGEN_LITTLE_ENDIAN) && !(RANDOMGEN_LITTLE_ENDIAN)
      uint64_t *block = (uint64_t *)&state->state[0];
      for (i=0; i<(2 * AESCTR_UNROLL);i++){
        block[i] = bswap_64(block[i]);
      }
#endif
      for (i = 0; i < 4; i++) {
        /* On BE, the encrypted data has LE order*/
        tiny_encrypt((state_t *)&state->state[16 * i], (uint8_t *)&state->seed);
      }

      for (i = 0; i < 4; i++) {
        state->ctr[i].u64[0] += AESCTR_UNROLL;
        /* Rolled if less than AESCTR_UNROLL */
        state->ctr[i].u64[1] += (state->ctr[i].u64[0] < AESCTR_UNROLL);
      }
      state->offset = 0;
    }
  }
  output = 0;
  memcpy(&output, &state->state[state->offset], sizeof(output));
  state->offset += sizeof(output);
#if defined(RANDOMGEN_LITTLE_ENDIAN) && !(RANDOMGEN_LITTLE_ENDIAN)
  /* On BE, the encrypted data has LE order*/
  output = bswap_64(output);
#endif

  return output;
}

static INLINE uint64_t aes_next64(aesctr_state_t *state) {
  return aesctr_r(state);
}

static INLINE uint32_t aes_next32(aesctr_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = aesctr_r(state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE double aes_next_double(aesctr_state_t *state) {
  return (aesctr_r(state) >> 11) * (1. / (UINT64_C(1) << 53));
}

extern void aesctr_use_aesni(int val);
extern int aes_capable(void);
extern void aesctr_seed(aesctr_state_t *state, uint64_t *seed);
extern void aesctr_set_counter(aesctr_state_t *state, uint64_t *counter);
extern void aesctr_set_seed_counter(aesctr_state_t *state, uint64_t *seed,
                                    uint64_t *counter);
extern void aesctr_get_seed_counter(aesctr_state_t *state, uint64_t *seed,
                                    uint64_t *counter);
extern void aesctr_advance(aesctr_state_t *state, uint64_t *step);
//#endif // __AES__
#endif
