#ifndef lcg128mix_H_INCLUDED
#define lcg128mix_H_INCLUDED 1

#include "pcg64-common.h"

typedef struct lcg128mix_RANDOM_T {
  pcg128_t state;
  pcg128_t inc;
  pcg128_t multiplier;
  uint64_t dxsm_multiplier;
  int post;
  int output_idx;
  pcg_output_func_t output_func;
} lcg128mix_random_t;

static INLINE void lcg128mix_step(lcg128mix_random_t *rng) {
#if defined(_WIN32) && _MSC_VER >= 1900 && _M_AMD64 &&                         \
    defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  uint64_t h1, multiplier_low = PCG_LOW(rng->multiplier);
  pcg128_t product;

  /* Manually INLINE the multiplication and addition using intrinsics */
  h1 = rng->state.high * multiplier_low +
       rng->state.low * PCG_HIGH(rng->multiplier);
  product.low = _umul128(rng->state.low, multiplier_low, &(product.high));
  product.high += h1;
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
#else
  rng->state = pcg128_add(pcg128_mult(rng->state, rng->multiplier), rng->inc);
#endif
}

static INLINE void lcg128mix_initialize(lcg128mix_random_t *rng,
                                           pcg128_t initstate,
                                           pcg128_t initseq) {
  rng->state = PCG_128BIT_CONSTANT(0ULL, 0ULL);
#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  rng->inc.high = initseq.high << 1u;
  rng->inc.high |= initseq.low >> 63u;
  rng->inc.low = (initseq.low << 1u) | 1u;
#else
  rng->inc = (initseq << 1U) | 1U;
#endif
  lcg128mix_step(rng);
  rng->state = pcg128_add(rng->state, initstate);
  lcg128mix_step(rng);
}

static INLINE uint64_t lcg128mix_next(lcg128mix_random_t *rng) {
  pcg128_t out;
  if (rng->post != 1) {
    out = rng->state;
    lcg128mix_step(rng);
  } else {
    lcg128mix_step(rng);
    out = rng->state;
  }
  switch (rng->output_idx) {
  case 0:
    return pcg_output_xsl_rr(PCG_HIGH(out), PCG_LOW(out));
  case 1:
    return pcg_output_dxsm(PCG_HIGH(out), PCG_LOW(out), rng->dxsm_multiplier);
  case 2:
    return pcg_output_murmur3(PCG_HIGH(out), PCG_LOW(out));
  case 3:
    return pcg_output_upper(PCG_HIGH(out), PCG_LOW(out));
  case 4:
    return pcg_output_lower(PCG_HIGH(out), PCG_LOW(out));
  case -1:
    return rng->output_func(PCG_HIGH(out), PCG_LOW(out));
  }
  return (uint64_t)(-1);
}

typedef struct lcg128mix_STATE_T {
  lcg128mix_random_t *pcg_state;
  int use_dxsm;
  int has_uint32;
  uint32_t uinteger;
} lcg128mix_state_t;

static INLINE uint64_t lcg128mix_next64(lcg128mix_state_t *state) {
  return lcg128mix_next(state->pcg_state);
}

static INLINE uint32_t lcg128mix_next32(lcg128mix_state_t *state) {
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  uint64_t value = lcg128mix_next(state->pcg_state);
  state->has_uint32 = 1;
  state->uinteger = value >> 32;
  return (uint32_t)value;
}

void lcg128mix_set_state(lcg128mix_random_t *rng, uint64_t state[],
                            uint64_t inc[], uint64_t multiplier[]);
void lcg128mix_get_state(lcg128mix_random_t *rng, uint64_t state[],
                            uint64_t inc[], uint64_t multiplier[]);
void lcg128mix_seed(lcg128mix_random_t *rng, uint64_t state[],
                       uint64_t inc[], uint64_t multiplier[]);
void lcg128mix_advance(lcg128mix_state_t *rng, uint64_t step[]);

#endif /* lcg128mix_H_INCLUDED */