/*
 * Generate testing csv files
 *
 * cl hc-128-test-data-gen.c hc-128.orig.c  util.orig.c /Ox -DLITTLE_ENDIAN
 * hc-128-test-data-gen.exe *
 *
 *  gcc hc-128-test-data-gen.c hc-128.orig.c  util.orig.c /O2 -DLITTLE_ENDIAN -o hc-128-test-data-gen
 *  ./hc-128-test-data-gen
 *
 */

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
  /* SeedSequence(0xDEADBEAF).generate_state(4, dtype=np.uint64) */
  uint64_t seed_seq_deafbeaf[4] = {5778446405158232650, 4639759349701729399, 13222832537653397986, 2330059127936092250};
  for (i = 0; i < 4; i++) {
    key_iv[i] = seed_seq_deafbeaf[i];
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
  /* SeedSequence(0).generate_state(4, dtype=np.uint64) */
  uint64_t seed_seq_0[4] = {15793235383387715774, 12390638538380655177, 2361836109651742017, 3188717715514472916};
  for (i = 0; i < 4; i++) {
    key_iv[i] = seed_seq_0[i];
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
