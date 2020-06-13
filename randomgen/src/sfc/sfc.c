#include "sfc.h"

extern INLINE uint64_t sfc_next64(sfc_state_t *state);
extern INLINE uint32_t sfc_next32(sfc_state_t *state);

void sfc_seed(sfc_state_t *state, uint64_t *seed, uint64_t w, uint64_t k){
    state->a = seed[0];
    state->b = seed[1];
    state->c = seed[2];
    state->w = w;
    state->k = k;
    for (int i=0; i<12; i++) {
      next64(state);
    }
}
