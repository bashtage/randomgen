#include "philox4x32.h"

extern INLINE uint64_t philox4x32_next64(philox4x32_state *state);

extern INLINE uint32_t philox4x32_next32(philox4x32_state *state);

extern INLINE double philox4x32_next_double(philox4x32_state *state);

extern void philox4x32_advance(uint32_t *step, philox4x32_state *state) {
  int i, carry = 0;
  uint32_t v_orig;
  for (i = 0; i < 4; i++) {
    if (carry == 1) {
      state->ctr->v[i]++;
      carry = state->ctr->v[i] == 0 ? 1 : 0;
    }
    v_orig = state->ctr->v[i];
    state->ctr->v[i] += step[i];
    if (state->ctr->v[i] < v_orig && carry == 0) {
      carry = 1;
    }
  }
}
