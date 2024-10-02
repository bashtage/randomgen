#include "_shim_dist.h"

double double0_func(bitgen_t *state) { return 3.141592; }

double double1_func(bitgen_t *state, double a) { return a; }

double double2_func(bitgen_t *state, double a, double b) { return a + b; }

double double3_func(bitgen_t *state, double a, double b, double c) {
  return a + b + c;
}

float float_0(bitgen_t *state) { return 3.141592; }

float float_1(bitgen_t *state, float a) { return a; }

int64_t int_0(void *state) { return 3; }

int64_t int_d(void *state, double a) { return (int64_t)(10 * a); };

int64_t int_dd(void *state, double a, double b) {
  return (int64_t)(10 * a * b);
};

int64_t int_di(void *state, double a, uint64_t b) {
  return (int64_t)2 * a * b;
};

int64_t int_i(void *state, int64_t a) { return a; };

int64_t int_iii(void *state, int64_t a, int64_t b, int64_t c) {
  return a + b + c;
};
