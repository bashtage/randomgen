#include "romu.h"

extern INLINE uint64_t romuquad_next64(romu_state_t *state);
extern INLINE uint64_t romutrio_next64(romu_state_t *state);
extern INLINE uint32_t romuquad_next32(romu_state_t *state);
extern INLINE uint32_t romutrio_next32(romu_state_t *state);

void romu_seed(romu_state_t *state, uint64_t w, uint64_t x, uint64_t y, uint64_t z, int quad) {
    state->w = w;
    state->x = x;
    state->y = y;
    state->z = z;
    /* Recommended in the paper */
    for (int i = 0; i < 10; i++) {
        if (quad != 0) {
            romuquad_next64(state);
        } else {
            romutrio_next64(state);
        }
    }
}
