/*
 * Generate testing csv files
 *
 *  cl xoshiro256-test-data-gen.c xoshiro256.orig.c /Ox
 * xoshiro256-test-data-gen.exe *
 *
 *  gcc xoshiro256-test-data-gen.c xoshiro256.orig.c /
 *  -o xoshiro256-test-data-gen
 *  ./xoshiro256-test-data-gen
 *
 */

#include "xoshiro256.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  uint64_t sum = 0;
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  int i;
  /* SeedSequence(0xDEADBEAF).generate_state(4, dtype=np.uint64) */
  s[0] = 5778446405158232650;
  s[1] = 4639759349701729399;
  s[2] = 13222832537653397986;
  s[3] = 2330059127936092250;
  uint64_t store[N];
  for (i = 0; i < N; i++) {
    store[i] = next();
  }

  FILE *fp;
  fp = fopen("xoshiro256-testset-1.csv", "w");
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

  seed = state = 0;
  /* SeedSequence(0).generate_state(4, dtype=np.uint64) */
  s[0] = 15793235383387715774;
  s[1] = 12390638538380655177;
  s[2] = 2361836109651742017;
  s[3] = 3188717715514472916;
  for (i = 0; i < N; i++) {
    store[i] = next();
  }
  fp = fopen("xoshiro256-testset-2.csv", "w");
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
