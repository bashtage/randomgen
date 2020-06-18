/*
 * Generate testing csv files
 *
 *  cl lxm-test-data-gen.c lxm.c /Ox
 * lxm-test-data-gen.exe *
 *
 *  gcc lxm-test-data-gen.c lxm.c -o lxm-test-data-gen
 *  ./lxm-test-data-gen
 *
 */

#include "lxm.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  int i;
  uint64_t seed = 0xDEADBEAF;
  lxm_state_t state;
  state.b = LCG_ADD;
  /* Generated from SeedSequece(0xDEADBEAF) */
  state.x[0] = 5778446405158232650ULL;
  state.x[1] = 4639759349701729399ULL;
  state.x[2] = 13222832537653397986ULL;
  state.x[3] = 2330059127936092250ULL;
  state.lcg_state = 6380887635277085283ULL;

  uint64_t store[N];
  for (i = 0; i < N; i++) {
    store[i] = lxm_next64(&state);
  }

  FILE *fp;
  fp = fopen("lxm-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);

  /* Generated from SeedSequece(0x0) */
  state.x[0] = 15793235383387715774ULL;
  state.x[1] = 12390638538380655177ULL;
  state.x[2] = 2361836109651742017ULL;
  state.x[3] = 3188717715514472916ULL;
  state.lcg_state = 648184599915300350ULL;

  for (i = 0; i < N; i++) {
    store[i] = lxm_next64(&state);
  }
  fp = fopen("lxm-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx64 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
