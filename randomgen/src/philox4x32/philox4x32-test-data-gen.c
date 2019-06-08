/*
 * Generate testing csv files
 *
 *  cl philox-test-data-gen.c /Ox
 *  philox4x32-test-data-gen.exe
 *
 *  gcc philox4x32-test-data-gen.c -o philox4x32-test-data-gen
 *  ./philox4x32-test-data-gen
 *
 * Requres the Random123 directory containing header files to be located in the
 * same directory (not included).
 *
 */

#include "../splitmix64/splitmix64.h"
#include "./Random123/philox.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  philox4x32_ctr_t ctr = {{0, 0, 0, 0}};
  philox4x32_key_t key = {{0, 0}};
  uint64_t state, seed = 0xDEADBEAF;
  philox4x32_ctr_t out;
  uint32_t store[N];
  state = seed;
  int i, j;
  state = splitmix64_next(&state);
  key.v[0] = (uint32_t)state;
  key.v[1] = (uint32_t)(state >> 32);
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = philox4x32_R(philox4x32_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  FILE *fp;
  fp = fopen("philox4x32-testset-1.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx32 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx32 "\n", i, store[i]);
    }
  }
  fclose(fp);

  ctr.v[0] = 0;
  state = seed = 0;
  state = splitmix64_next(&state);
  key.v[0] = (uint32_t)state;
  key.v[1] = (uint32_t)(state >> 32);
  for (i = 0; i < N / 4UL; i++) {
    ctr.v[0]++;
    out = philox4x32_R(philox4x32_rounds, ctr, key);
    for (j = 0; j < 4; j++) {
      store[i * 4 + j] = out.v[j];
    }
  }

  fp = fopen("philox4x32-testset-2.csv", "w");
  if (fp == NULL) {
    printf("Couldn't open file\n");
    return -1;
  }
  fprintf(fp, "seed, 0x%" PRIx64 "\n", seed);
  for (i = 0; i < N; i++) {
    fprintf(fp, "%d, 0x%" PRIx32 "\n", i, store[i]);
    if (i == 999) {
      printf("%d, 0x%" PRIx32 "\n", i, store[i]);
    }
  }
  fclose(fp);
}
