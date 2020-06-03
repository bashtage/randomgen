#include "lxm.h"

extern inline void lcg(lxm_state_t *state);

extern inline uint64_t lxm_next64(lxm_state_t *state);

void lcg_jump(lxm_state_t *state) {
    uint64_t acc_mult = 1u;
    uint64_t acc_plus = 0u;
    uint64_t cur_plus = state->b;
    uint64_t cur_mult = LCG_MULT;
    /* 2^128 has bit 1 in location 128, and 0 else where */
    for (int i=0; i < 129; i++){
        if (i == 128) {
            acc_mult *= cur_mult;
            acc_plus = acc_plus * cur_mult + cur_plus;
        }
        cur_plus = (cur_mult + 1) * cur_plus;
        cur_mult *= cur_mult;
    }
    state->lcg_state = acc_mult * state->lcg_state + acc_plus;
}

void xorshift_jump(lxm_state_t *state)
{
  static const uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa,
                                  0x39abdc4529b1661c};

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  uint64_t s2 = 0;
  uint64_t s3 = 0;
  for (int i = 0; i < (int)(sizeof(JUMP) / sizeof(*JUMP)); i++)
    for (int b = 0; b < 64; b++)
    {
      if (JUMP[i] & UINT64_C(1) << b)
      {
        s0 ^= state->x[0];
        s1 ^= state->x[1];
        s2 ^= state->x[2];
        s3 ^= state->x[3];
      }
      xorshift(state);
    }

  state->x[0] = s0;
  state->x[1] = s1;
  state->x[2] = s2;
  state->x[3] = s3;
}

void lxm_jump(lxm_state_t *state)
{

  /*
   * lcg jump is a no-op since we are using a multiplier
   * the full cycle
   *
   * lcg_jump(state);
   */
  xorshift_jump(state);
}
