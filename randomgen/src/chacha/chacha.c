#include "chacha.h"
#include "../common/cpu_features.h"

int RANDOMGEN_USE_SIMD;

extern INLINE uint32_t chacha_next32(chacha_state_t *state);
extern INLINE uint64_t chacha_next64(chacha_state_t *state);
extern INLINE double chacha_next_double(chacha_state_t *state);

#if defined(__SSE2__) && __SSE2__
#if defined(__SSSE3__) && __SSSE3__
#define CHACHA_FEATURE_REG RANDOMGEN_ECX
#define CHACHA_FEATURE_FLAG 9
#else
#define CHACHA_FEATURE_REG RANDOMGEN_EDX
#define CHACHA_FEATURE_FLAG 26
#endif
#else
#define CHACHA_FEATURE_FLAG 0
#endif

int chacha_simd_capable(void) {
#if defined(__SSE2__) && __SSE2__
  int flags[32];
  feature_flags(flags, CHACHA_FEATURE_REG);
  RANDOMGEN_USE_SIMD = flags[CHACHA_FEATURE_FLAG];
  return RANDOMGEN_USE_SIMD;
#else
  RANDOMGEN_USE_SIMD = 0;
  return 0;
#endif
}

void chacha_use_simd(int flag) { RANDOMGEN_USE_SIMD = flag; }

void chacha_seed(chacha_state_t *state, uint64_t *seedval, uint64_t *stream,
                 uint64_t *ctr) {
  chacha_simd_capable();
  // Using a 128-bit seed.
  state->keysetup[0] = seedval[0] & 0xffffffffu;
  state->keysetup[1] = seedval[0] >> 32;
  state->keysetup[2] = seedval[1] & 0xffffffffu;
  state->keysetup[3] = seedval[1] >> 32;
  // Using a 128-bit stream.
  state->keysetup[4] = stream[0] & 0xffffffffu;
  state->keysetup[5] = stream[0] >> 32;
  state->keysetup[6] = stream[1] & 0xffffffffu;
  state->keysetup[7] = stream[1] >> 32;

  /* Ensure str[0] is at a node where a block would be generated */
  state->ctr[0] = ((ctr[0] >> 4) << 4);
  state->ctr[1] = ctr[1];
  generate_block(state);
  /* Store correct value of counter */
  state->ctr[0] = ctr[0];
}

void chacha_advance(chacha_state_t *state, uint64_t *delta) {
  int carry, idx = state->ctr[0] % 16;
  uint64_t orig;
  orig = state->ctr[0];
  state->ctr[0] += delta[0];
  carry = state->ctr[0] < orig;
  state->ctr[1] += (delta[1] + carry);
  if ((idx + delta[0] >= 16 || delta[1]) && ((state->ctr[0] % 16) != 0)) {
    generate_block(state);
  }
}
