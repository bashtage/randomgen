#include "threefry.h"
#include <stdio.h>

#define _threefryNxW_advance_tpl(N, W)                                     \
void threefry##N##x##W##_advance(threefry_all_t *state, uint##W##_t *step) { \
  int i;                                                                 \
  uint##W##_t last, carry;                                               \
  carry = 0;                                                             \
  for (i=0; i<N; i++){                                                   \
      last = state->state##N##x##W.ctr.v[i];                             \
      state->state##N##x##W.ctr.v[i] += step[i] + carry;                 \
      carry = (last < state->state##N##x##W.ctr.v[i]);                   \
  }                                                                      \
}

_threefryNxW_advance_tpl(2, 32)                                                
_threefryNxW_advance_tpl(4, 32)                                                
_threefryNxW_advance_tpl(2, 64)                                                
_threefryNxW_advance_tpl(4, 64)                                                

#define _threefryNxW_next_extern_tpl(N, W)                                            \
extern INLINE uint64_t threefry##N##x##W##_next64(threefry_all_t *state);    \
extern INLINE uint32_t threefry##N##x##W##_next32(threefry_all_t *state);    \
extern INLINE double threefry##N##x##W##_next_double(threefry_all_t *state);

_threefryNxW_next_extern_tpl(2, 32)
_threefryNxW_next_extern_tpl(4, 32)
_threefryNxW_next_extern_tpl(2, 64)
_threefryNxW_next_extern_tpl(4, 64)
