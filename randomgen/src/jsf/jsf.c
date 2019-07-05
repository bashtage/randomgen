#include "jsf.h"

extern INLINE uint64_t jsf64_next64(jsf_state_t *state);
extern INLINE uint32_t jsf64_next32(jsf_state_t *state);
extern INLINE double jsf64_next_double(jsf_state_t *state);
extern INLINE uint64_t jsf32_next64(jsf_state_t *state);
extern INLINE uint32_t jsf32_next32(jsf_state_t *state);
extern INLINE double jsf32_next_double(jsf_state_t *state);

void jsf64_seed(jsf_state_t *state, uint64_t *seed, int size) {
  int i;
  state->a.u64 = 0xf1ea5eed;
  state->b.u64 = seed[0];
  state->c.u64 = seed[0];
  state->d.u64 = seed[0];
  if (size > 1) state->c.u64 = seed[1];
  if (size > 2) state->d.u64 = seed[2];
  for (i = 0; i < 20; ++i) {
    next64(state);
  }
}

void jsf32_seed(jsf_state_t *state, uint32_t* seed, int size) {
  int i;
  state->a.u32 = 0xf1ea5eed;
  state->b.u32 = seed[0];
  state->c.u32 = seed[0];
  state->d.u32 = seed[0];
  if (size > 1) state->c.u32 = seed[1];
  if (size > 2) state->d.u32 = seed[2];
  for (i = 0; i < 20; ++i) {
    next32(state);
  }
}
