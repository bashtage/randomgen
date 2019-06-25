/*
 * Generate testing csv files
 *
 *  cl hc-128-test-data-gen.c hc-128.orig.c  util.orig.c /
 * ../splitmix64/splitmix64.c /Ox -DLITTLE_ENDIAN
 * hc-128-test-data-gen.exe *
 *
 *  gcc hc-128-test-data-gen.c hc-128.orig.c  util.orig.c /
 * ../splitmix64/splitmix64.c /Ox -DLITTLE_ENDIAN -o hc-128-test-data-gen
 *  ./hc-128-test-data-gen
 *
 */

#include "../splitmix64/splitmix64.h"
#include "hc-128.orig.h"
#include <inttypes.h>
#include <stdio.h>

#define N 1000

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

int main() {
  uint64_t sum = 0;
  uint64_t state, seed = 0xDEADBEAF;
  uint64_t key_iv[4] = { 0,0,0,0 };
  int i, j;
  hc128_state hstate;
  state = seed;
  for (i = 0; i < 4; i++) {
    key_iv[i] = splitmix64_next(&state);
    printf("%"PRIu64"\n", key_iv[i]);
  }
  hc128_init(&hstate, (uint8_t *)&key_iv[0], (uint8_t *)&key_iv[2]);
  uint64_t store[N];
  uint32_t temp[2];
  for (i = 0; i < N; i++) {
    for (j=0; j<2; j++) {
      hc128_extract(&hstate, (uint8_t *)&temp[j]);
    }
    store[i] = temp[0] | ((uint64_t)temp[1])<<32;
  }

  FILE *fp;
  int err = fopen_s(&fp, "hc-128-testset-1.csv", "w");
  if (err != 0) {
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
  for (i = 0; i < 4; i++) {
    key_iv[i] = splitmix64_next(&state);
    printf("%"PRIu64"\n", key_iv[i]);
  }
  hc128_init(&hstate, (uint8_t *)&key_iv[0], (uint8_t *)&key_iv[2]);
  for (i = 0; i < N; i++) {
    for (j=0; j<2; j++) {
      hc128_extract(&hstate, (uint8_t *)&temp[j]);
    }
    store[i] = temp[0] | ((uint64_t)temp[1])<<32;
  }
  err = fopen_s(&fp, "hc-128-testset-2.csv", "w");
  if (err != 0) {
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
