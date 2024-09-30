/*
 * Generate testing csv files
 *
 *  cl squares-test-data-gen.c /O2
 * squares-test-data-gen.exe *
 *
 *  gcc -o2 squares-test-data-gen.c -o squares-test-data-gen
 *  ./squares-test-data-gen
 *
 */

#include "squares.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

int main() {
  int i;
  uint64_t seed = 0xDEADBEAF;
  /* key =  randomgen.squares.generate_keys(0xDEADBEAF, 1)[0] */
  uint64_t key = 0x3bc8bca4d301945b;
  uint64_t ctr = 0x0;
  uint64_t a, b;
  uint64_t store[N];
  for (i = 0; i < N; i++) {
    store[i] = squares64(key, ctr);
    ctr++;
  }

  FILE *fp;
  fp = fopen("squares-testset-1.csv", "w");
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

  ctr = 0;
  for (i = 0; i < N; i++) {
    a = (uint64_t)squares32(key, ctr);
    ctr++;
    b = (uint64_t)squares32(key, ctr);
    ctr++;
    store[i] = a | (b << 32);
}

  fp = fopen("squares-32-testset-1.csv", "w");
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
  /* key =  squares.generate_keys(0, 1)[0] */
  key = 0x32d1638d0d582acf;
  ctr = 0x0;

  for (i = 0; i < N; i++) {
    store[i] = squares64(key, ctr);
    ctr++;
  }
  fp = fopen("squares-testset-2.csv", "w");
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

  ctr = 0;
  for (i = 0; i < N; i++) {
    a = (uint64_t)squares32(key, ctr);
    ctr++;
    b = (uint64_t)squares32(key, ctr);
    ctr++;
    store[i] = a | (b << 32);
  }

  fp = fopen("squares-32-testset-2.csv", "w");
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
