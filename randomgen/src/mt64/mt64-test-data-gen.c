/*
 * Generate testing csv files
 *
 *  cl mt64-test-data-gen.c mt64.orig.c /Ox
 *  mt19937-64-test-data-gen.exe
 *
 *  gcc mt64-test-data-gen.c mt64-64.orig.c -o mt64-test-data-gen
 *  ./mt19937-64-test-data-gen
 *
 */

#include "mt64-test-data-seed.h"
#include "mt64.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  uint64_t sum = 0;
  uint64_t state, seed = 0xDEADBEAF;
  state = seed;
  int i;
  uint64_t store[N];
  init_by_array64(&seed_seq_deadbeaf, 312);

  for (i = 0; i < N; i++) {
    store[i] = genrand64_int64();
  }

  FILE *fp;
  fp = fopen("mt64-testset-1.csv", "w");
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
  init_by_array64(&seed_seq_0, 312);
  for (i = 0; i < N; i++) {
    store[i] = genrand64_int64();
  }
  fp = fopen("mt64-testset-2.csv", "w");
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
