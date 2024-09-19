/*
 * Generate testing csv files
 *
 * cl  pcg32-test-data-gen.c /O2
 * gcc  -o2 pcg32-test-data-gen.c pcg32.orig.c -o pgc64-test-data-gen
 */

#include "pcg_variants.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  pcg32_random_t rng;
  uint64_t seed = 0xDEADBEAF;
  /* SeedSequence(0xDEADBEAF).generate_state(2, dtype=np.uint64) */
  uint64_t seed_seq_deadbeaf[2] = {5778446405158232650ULL, 4639759349701729399ULL};
  int i;
  uint64_t store[N];

  pcg32_srandom_r(&rng, seed_seq_deadbeaf[0], seed_seq_deadbeaf[1]);
  printf("State: %" PRIu64  "\n", rng.state);
  printf("Inc: %" PRIu64 "\n", rng.inc);
  for (i = 0; i < N; i++) {
    store[i] = pcg32_random_r(&rng);
  }

  FILE *fp;
  fp = fopen("pcg32-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, %" PRIu64 "\n", i, store[i]);
    }
  }
  fclose(fp);

  seed = 0;
  /* SeedSequence(0).generate_state(2, dtype=np.uint64) */
  uint64_t seed_seq_0[2] = {15793235383387715774ULL, 12390638538380655177ULL};
  pcg32_srandom_r(&rng, seed_seq_0[0], seed_seq_0[1]);
  printf("State: %" PRIu64  "\n", rng.state);
  printf("Inc: %" PRIu64 "\n", rng.inc);
  for (i = 0; i < N; i++) {
    store[i] = pcg32_random_r(&rng);
  }
  fp = fopen("pcg32-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx64 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, %" PRIu64 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
