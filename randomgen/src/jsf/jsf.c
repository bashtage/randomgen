#include "jsf.h"
#include <inttypes.h>
#include <stdio.h>
#include <time.h>

extern INLINE uint64_t jsf64_next64(jsf_state_t *state);
extern INLINE uint32_t jsf64_next32(jsf_state_t *state);
extern INLINE double jsf64_next_double(jsf_state_t *state);
extern INLINE uint64_t jsf32_next64(jsf_state_t *state);
extern INLINE uint32_t jsf32_next32(jsf_state_t *state);
extern INLINE double jsf32_next_double(jsf_state_t *state);

void jsf64_seed(jsf_state_t *state, uint64_t seed) {
  int i;
  state->a.u64 = 0xf1ea5eed;
  state->b.u64 = seed;
  state->c.u64 = seed;
  state->d.u64 = seed;
  for (i = 0; i < 20; ++i) {
    next64(state);
  }
}

void jsf32_seed(jsf_state_t *state, uint32_t seed) {
  int i;
  state->a.u32 = 0xf1ea5eed;
  state->b.u32 = seed;
  state->c.u32 = seed;
  state->d.u32 = seed;
  for (i = 0; i < 20; ++i) {
    next32(state);
  }
}
