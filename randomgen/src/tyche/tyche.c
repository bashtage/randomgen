#include "tyche.h"

void tyche_seed(tyche_state_t *state, uint64_t seed, uint32_t idx,
                int original) {
  state->a = (uint32_t)(seed >> 32);
  state->b = (uint32_t)(seed & 0xFFFFFFFFULL);
  state->c = 2654435769;
  state->d = 1367130551 ^ idx;

  for (int i = 0; i < 20; i++) {
    if (original == 1) {
      mix(state);
    } else if (original == 0) {
      mix_openrand(state);
    }
  }
}
