#include "_shim_dist.h"

double double0_func(bitgen_t *state) { return 3.141592; }

double double1_func(bitgen_t *state, double a) { return a; }

double double2_func(bitgen_t *state, double a, double b) { return a + b; }

double double3_func(bitgen_t *state, double a, double b, double c) {
  return a + b + c;
}