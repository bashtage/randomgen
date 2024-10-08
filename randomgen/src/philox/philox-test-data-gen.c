/*
 * Generate testing csv files
 *
 *  cl philox-test-data-gen.c /Ox
 *  philox-test-data-gen.exe
 *
 *  gcc philox-test-data-gen.c -o philox-test-data-gen
 *  ./philox-test-data-gen
 *
 * Requres the Random123 directory containing header files to be located in the
 * same directory (not included).
 *
 */

#include "Random123/philox.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  philox4x64_ctr_t ctr = {{0, 0, 0, 0}};
  philox4x64_key_t key = {{0, 0}};
  uint64_t state, seed = 0xDEADBEAF;
  /* SeedSequence(0xDEADBEAF).generate_state(2, dtype=np.uint64) */
  uint64_t seed_seq[2] = {5778446405158232650, 4639759349701729399};
  philox4x64_ctr_t out;
  uint64_t store[N];
  state = seed;
  int i, j;
  for (i = 0; i < 2; i++) {
    key.v[i] = seed_seq[i];
  }
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = philox4x64_R(philox4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  FILE *fp;
  fp = fopen("philox-testset-1.csv", "w");
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

  ctr.v[0] = 0;
  state = seed = 0;
  /* SeedSequence(0).generate_state(2, dtype=np.uint64) */
  key.v[0] = 15793235383387715774;
  key.v[1] = 12390638538380655177;

  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = philox4x64_R(philox4x64_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  fp = fopen("philox-testset-2.csv", "w");
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
