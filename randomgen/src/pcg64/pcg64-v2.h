#ifndef PCG64_H_INCLUDED
#define PCG64_H_INCLUDED 1

#include "pcg64-common.h"

#define PCG_DEFAULT_MULTIPLIER_HIGH 2549297995355413924ULL
#define PCG_DEFAULT_MULTIPLIER_LOW 4865540595714422341ULL
#define PCG_DEFAULT_MULTIPLIER_128                                             \
  PCG_128BIT_CONSTANT(PCG_DEFAULT_MULTIPLIER_HIGH, PCG_DEFAULT_MULTIPLIER_LOW)
#define PCG_DEFAULT_INCREMENT_128                                              \
  PCG_128BIT_CONSTANT(6364136223846793005ULL, 1442695040888963407ULL)
#define PCG_CHEAP_MULTIPLIER_HIGH 0ULL
#define PCG_CHEAP_MULTIPLIER_LOW 0xda942042e4dd58b5ULL
#define PCG_CHEAP_MULTIPLIER_128                                               \
  PCG_128BIT_CONSTANT(0ULL, 0xda942042e4dd58b5ULL)

typedef struct PCG_RANDOM_T {
  pcg128_t state;
  pcg128_t inc;
} pcg64_random_t;

static INLINE void pcg64_step_r(pcg64_random_t *rng) {
#if defined _WIN32 && _MSC_VER >= 1900 && _M_AMD64 &&                          \
    defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  uint64_t h1;
  pcg128_t product;

  /* Manually INLINE the multiplication and addition using intrinsics */
  h1 = rng->state.high * PCG_DEFAULT_MULTIPLIER_LOW +
       rng->state.low * PCG_DEFAULT_MULTIPLIER_HIGH;
  product.low =
      _umul128(rng->state.low, PCG_DEFAULT_MULTIPLIER_LOW, &(product.high));
  product.high += h1;
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
#else
  rng->state =
      pcg128_add(pcg128_mult(rng->state, PCG_DEFAULT_MULTIPLIER_128), rng->inc);
#endif
}

static INLINE void pcg64_cm_step_r(pcg64_random_t *rng) {
  // TODO: Fix pcg128_mult to not use the high (Maybe)
#if defined _WIN32 && _MSC_VER >= 1900 && _M_AMD64 &&                          \
    defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  uint64_t h1;
  pcg128_t product;

  /* Manually INLINE the multiplication and addition using intrinsics */
  h1 = rng->state.high * PCG_CHEAP_MULTIPLIER_LOW + rng->state.low * 0U;
  product.low =
      _umul128(rng->state.low, PCG_CHEAP_MULTIPLIER_LOW, &(product.high));
  product.high += h1;
  _addcarry_u64(_addcarry_u64(0, product.low, rng->inc.low, &(rng->state.low)),
                product.high, rng->inc.high, &(rng->state.high));
#else
  rng->state =
      pcg128_add(pcg128_mult(rng->state, PCG_CHEAP_MULTIPLIER_128), rng->inc);
#endif
}

static INLINE void pcg64_initialize(pcg64_random_t *rng, pcg128_t initstate,
                                    pcg128_t initseq, int cheap_multiplier) {
  rng->state = PCG_128BIT_CONSTANT(0ULL, 0ULL);
#if defined(PCG_EMULATED_128BIT_MATH) && PCG_EMULATED_128BIT_MATH
  rng->inc.high = initseq.high << 1u;
  rng->inc.high |= initseq.low >> 63u;
  rng->inc.low = (initseq.low << 1u) | 1u;
#else
  rng->inc = (initseq << 1U) | 1U;
#endif
  if (cheap_multiplier == 0) {
    pcg64_step_r(rng);
  } else {
    pcg64_cm_step_r(rng);
  }
  rng->state = pcg128_add(rng->state, initstate);
  if (cheap_multiplier == 0) {
    pcg64_step_r(rng);
  } else {
    pcg64_cm_step_r(rng);
  }
}

static INLINE uint64_t pcg64_random_r(pcg64_random_t *rng, int use_dxsm) {
  pcg64_step_r(rng);
  if (use_dxsm != 0) {
    return pcg_output_dxsm(PCG_HIGH(rng->state), PCG_LOW(rng->state),
                           PCG_CHEAP_MULTIPLIER_LOW);
  } else {
    return pcg_output_xsl_rr(PCG_HIGH(rng->state), PCG_LOW(rng->state));
  }
}

static INLINE uint64_t pcg64_dxsm_random_r(pcg64_random_t *rng) {
  uint64_t out = pcg_output_dxsm(PCG_HIGH(rng->state), PCG_LOW(rng->state),
                                 PCG_CHEAP_MULTIPLIER_LOW);
  pcg64_cm_step_r(rng);
  return out;
}

typedef struct PCG64_STATE_T {
  pcg64_random_t *pcg_state;
  int use_dxsm;
  int has_uint32;
  uint32_t uinteger;
} pcg64_state_t;

static INLINE uint64_t pcg64_next64(pcg64_state_t *state) {
  return pcg64_random_r(state->pcg_state, state->use_dxsm);
}

static INLINE uint64_t pcg64_cm_dxsm_next64(pcg64_state_t *state) {
  return pcg64_dxsm_random_r(state->pcg_state);
}

static INLINE uint32_t pcg64_next32(pcg64_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = pcg64_random_r(state->pcg_state, state->use_dxsm);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE uint32_t pcg64_cm_dxsm_next32(pcg64_state_t *state) {
  uint64_t next;
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  next = pcg64_dxsm_random_r(state->pcg_state);
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

void pcg64_set_seed(pcg64_state_t *state, uint64_t *seed, uint64_t *inc,
                    int cheap_multiplier);
void pcg64_advance(pcg64_state_t *state, uint64_t *step, int cheap_multiplier);
void pcg64_set_state(pcg64_state_t *state, uint64_t *state_arr, int use_dxsm,
                     int has_uint32, uint32_t uinteger);
void pcg64_get_state(pcg64_state_t *state, uint64_t *state_arr, int *use_dxsm,
                     int *has_uint32, uint32_t *uinteger);

#endif