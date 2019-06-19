#include "philox.h"
#include <stdio.h>

#define _philoxNxW_advance_tpl(N, W)                                     \
void philox##N##x##W##_advance(philox_all_t *state, uint##W##_t *step) { \
  int i;                                                                 \
  uint##W##_t last, carry;                                               \
  carry = 0;                                                             \
  for (i=0; i<N; i++){                                                   \
      last = state->state##N##x##W.ctr.v[i];                             \
      state->state##N##x##W.ctr.v[i] += step[i] + carry;                 \
      carry = (last < state->state##N##x##W.ctr.v[i]);                   \
  }                                                                      \
}

_philoxNxW_advance_tpl(2, 32)                                                
_philoxNxW_advance_tpl(4, 32)                                                
_philoxNxW_advance_tpl(2, 64)                                                
_philoxNxW_advance_tpl(4, 64)                                                

#define _philoxNxW_next_extern_tpl(N, W)                                            \
extern INLINE uint64_t philox##N##x##W##_next64(philox_all_t *state);    \
extern INLINE uint32_t philox##N##x##W##_next32(philox_all_t *state);    \
extern INLINE double philox##N##x##W##_next_double(philox_all_t *state);

_philoxNxW_next_extern_tpl(2, 32)
_philoxNxW_next_extern_tpl(4, 32)
_philoxNxW_next_extern_tpl(2, 64)
_philoxNxW_next_extern_tpl(4, 64)
