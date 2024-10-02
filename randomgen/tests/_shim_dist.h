#include <inttypes.h>
#include "numpy/random/distributions.h"

extern double double0_func(bitgen_t *state);
extern double double1_func(bitgen_t *state, double a);
extern double double2_func(bitgen_t *state, double a, double b);
extern double double3_func(bitgen_t *state, double a, double b, double c);

extern float float_0(bitgen_t *state);
extern float float_1(bitgen_t *state, float a);

extern int64_t int_0(void *state);
extern int64_t int_d(void *state, double a);
extern int64_t int_dd(void *state, double a, double b);
extern int64_t int_di(void *state, double a, uint64_t b);
extern int64_t int_i(void *state, int64_t a);
extern int64_t int_iii(void *state, int64_t a, int64_t b, int64_t c);

/*
extern uint32_t uint_0_32(bitgen_t *state);
extern uint32_t uint_1_i_32(bitgen_t *state, uint32_t a);

extern int32_t int_2_i_32(bitgen_t *state, int32_t a, int32_t b);
extern int64_t int_2_i(bitgen_t *state, int64_t a, int64_t b);
*/