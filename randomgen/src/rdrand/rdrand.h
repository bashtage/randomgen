#ifndef _RANDOMDGEN__RDRAND_H_
#define _RANDOMDGEN__RDRAND_H_

#include "../common/randomgen_config.h"
#include "../common/randomgen_immintrin.h"

#define BUFFER_SIZE 256

/* Golden ratio phi - 1 when divided by 2 ** 64 */
#define PHI_M_1 11400714819323198485ULL

typedef struct s_rdrand_state {
  uint64_t buffer[BUFFER_SIZE];
  int buffer_loc;
  int status;
  int retries;
  uint64_t weyl_seq;
} rdrand_state;

/*
next checks buffer, if available, fills
if not available, calls fill_buffer
fill_buffer will spin lock and pause
check status flag from fill buffer
if
*/

int rdrand_capable(void);

static INLINE int rdrand_fill_buffer(rdrand_state *state) {
  /*
   * You **must** check the returned value:
   *   1 if success
   *   0 if failure
   */
  int status = 0, retries_cnt;
  uint64_t val;
  /* Assume success */
  for (int i = 0; i < BUFFER_SIZE; i++) {
    status = 0;
    retries_cnt = 0;
    while ((status == 0) && (retries_cnt++ <= state->retries)) {
#if defined(__RDRND__) && __RDRND__
#if defined(__x86_64__) || defined(_M_X64)
      status = _rdrand64_step((long long unsigned int *)&val);
#else
      uint32_t low, high;
      status = _rdrand32_step(&low);
      status &= _rdrand32_step(&high);
      val = ((uint64_t)high) << 32 | low;
#endif
#else
      /* Never called on platforms without RDRAND */
      return 0;
#endif
      if (status != 0) {
        state->buffer[i] = val;
      } else {
#if defined(__RDRND__) && __RDRND__
        _mm_pause();
#endif
      }
    }
    state->status &= status;
    if (status == 0) {
      /* This is done to ensure that rejection samplers always exit.
       * This scenario could happen if RDRAND was being used in a
       * NumPy generator with any rejection-based sampler. If RDRAND
       * fails then this will use a naive PRNG which will provide enough
       * randomness for the rejection sampler to complete. More importantly,
       * status and state->status are both 0 which will be returned when
       * the final set of draws completes even if RDRAND starts operating
       * again. This allows the code to raise RuntimeError in Python.
       */
      if (i > 0) {
        state->buffer[i] = state->buffer[i - 1] + PHI_M_1;
      } else {
        state->buffer[i] = PHI_M_1;
      }
    }
  }
  /* Reset only on success */
  state->buffer_loc = 0;
  return state->status;
}

static INLINE int rdrand_next64(rdrand_state *state, uint64_t *val) {
  /*
   * You **must** check the returned status
   *   1 if success
   *   0 if failure
   */
  int status = 0, retries_cnt = 0;
  while ((status == 0) && (retries_cnt++ <= state->retries)) {
#if defined(__RDRND__) && __RDRND__
#if defined(__x86_64__) || defined(_M_X64)
    status = _rdrand64_step((long long unsigned int *)val);
#else
    uint32_t low, high;
    status = _rdrand32_step(&low);
    status &= _rdrand32_step(&high);
    val[0] = ((uint64_t)high) << 32 | low;
#endif
#else
    /* Never called on platforms without RDRAND */
    state->status = 0;
    return 0;
#endif
    if (status == 0) {
#if defined(__RDRND__) && __RDRND__
      _mm_pause();
#endif
    }
  }
  state->status &= status;
  return state->status;
}

#endif /* _RANDOMDGEN__RDRAND_H_ */