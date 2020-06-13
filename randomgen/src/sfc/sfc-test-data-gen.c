/*
 * Generate testing csv files
 *
 *  cl sfc-test-data-gen.c sfc.c /Ox
 * sfc-test-data-gen.exe *
 *
 *  gcc sfc-test-data-gen.c sfc.c -o sfc-test-data-gen
 *  ./sfc-test-data-gen
 *
 */

#include "sfc.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  int i;
  uint64_t seed = 0xDEADBEAF;
  sfc_state_t state;
  /* Generated from SeedSequece(0xDEADBEAF) */
  state.a = 5778446405158232650ULL;
  state.b = 4639759349701729399ULL;
  state.c = 13222832537653397986ULL;
  state.w = 1ULL;
  state.k = 1ULL;
  for (i=0; i<12; i++){
    next64(&state);
  }
  printf("%"PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", state.a, state.b, state.c, state.w, state.k);

  uint64_t store[N];
  for (i = 0; i < N; i++) {
    store[i] = sfc_next64(&state);
  }

  FILE *fp;
  fp = fopen("sfc-testset-1.csv", "w");
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

  seed = 0x0;
  /* Generated from SeedSequece(0x0) */
  state.a = 15793235383387715774ULL;
  state.b = 12390638538380655177ULL;
  state.c = 2361836109651742017ULL;
  state.w = 1ULL;
  state.k = 1ULL;
  for (i=0; i<12; i++){
    next64(&state);
  }
  printf("%"PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", state.a, state.b, state.c, state.w, state.k);

  for (i = 0; i < N; i++) {
    store[i] = sfc_next64(&state);
  }
  fp = fopen("sfc-testset-2.csv", "w");
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
