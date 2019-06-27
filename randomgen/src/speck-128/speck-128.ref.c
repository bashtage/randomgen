/*
 * Based on:
 * SIMON and SPECK Implementation Guide
 * Ray Beaulieu, Douglas Shors, Jason Smith, Stefan Treatman-Clark, Bryan Weeks, Louis Wingers
 * National Security Agency
 * 9800 Savage Road, Fort Meade, MD, 20755, USA
 * January 15, 2019
 */

#include <inttypes.h>
#include <stdio.h>
#include "../splitmix64/splitmix64.h"

#define u8 uint8_t
#define u32 uint32_t
#define u64 uint64_t

#define ROTL32(x, r) (((x) << (r)) | (x >> (32 - (r))))
#define ROTR32(x, r) (((x) >> (r)) | ((x) << (32 - (r))))
#define ROTL64(x, r) (((x) << (r)) | (x >> (64 - (r))))
#define ROTR64(x, r) (((x) >> (r)) | ((x) << (64 - (r))))

void Words32ToBytes(u32 words[], u8 bytes[], int numwords) {
  int i, j = 0;
  for (i = 0; i < numwords; i++) {
    bytes[j] = (u8)words[i];
    bytes[j + 1] = (u8)(words[i] >> 8);
    bytes[j + 2] = (u8)(words[i] >> 16);
    bytes[j + 3] = (u8)(words[i] >> 24);
    j += 4;
  }
}

void Words64ToBytes(u64 words[], u8 bytes[], int numwords) {
  int i, j = 0;
  for (i = 0; i < numwords; i++) {
    bytes[j] = (u8)words[i];
    bytes[j + 1] = (u8)(words[i] >> 8);
    bytes[j + 2] = (u8)(words[i] >> 16);
    bytes[j + 3] = (u8)(words[i] >> 24);
    bytes[j + 4] = (u8)(words[i] >> 32);
    bytes[j + 5] = (u8)(words[i] >> 40);
    bytes[j + 6] = (u8)(words[i] >> 48);
    bytes[j + 7] = (u8)(words[i] >> 56);
    j += 8;
  }
}

void BytesToWords32(u8 bytes[], u32 words[], int numbytes) {
  int i, j = 0;
  for (i = 0; i < numbytes / 4; i++) {
    words[i] = (u32)bytes[j] | ((u32)bytes[j + 1] << 8) |
               ((u32)bytes[j + 2] << 16) | ((u32)bytes[j + 3] << 24);
    j += 4;
  }
}
void BytesToWords64(u8 bytes[], u64 words[], int numbytes) {
  int i, j = 0;
  for (i = 0; i < numbytes / 8; i++) {
    words[i] = (u64)bytes[j] | ((u64)bytes[j + 1] << 8) |
               ((u64)bytes[j + 2] << 16) | ((u64)bytes[j + 3] << 24) |
               ((u64)bytes[j + 4] << 32) | ((u64)bytes[j + 5] << 40) |
               ((u64)bytes[j + 6] << 48) | ((u64)bytes[j + 7] << 56);
    j += 8;
  }
}

static inline void TF83(uint64_t *x, uint64_t *y, const uint64_t k) {
  x[0] = ((x[0] >> 8) | (x[0] << (64 - 8)));
  x[0] += y[0];
  x[0] ^= k;
  y[0] = ((y[0] << 3) | (y[0] >> (64 - 3)));
  y[0] ^= x[0];
}

#define ER64(x, y, s) (x = (ROTR64(x, 8) + y) ^ (s), y = ROTL64(y, 3) ^ x)
void Speck128256Encrypt(u64 Pt[], u64 Ct[], u64 rk[]) {
  u64 i;
  Ct[0] = Pt[0];
  Ct[1] = Pt[1];
  for (i = 0; i < 34; i++)
    ER64(Ct[1], Ct[0], rk[i]);
    // TF83(Ct + 1, Ct + 0, rk[i]);
}

void Speck128256KeySchedule(u64 K[], u64 rk[]) {
  u64 i, D = K[3], C = K[2], B = K[1], A = K[0];
  for (i = 0; i < 33; i += 3) {
    rk[i] = A;
    ER64(B, A, i);
    rk[i + 1] = A;
    ER64(C, A, i + 1);
    rk[i + 2] = A;
    ER64(D, A, i + 2);
  }
  rk[33] = A;
}

#ifdef __unix
#define fopen_s(pFile,filename,mode) ((*(pFile))=fopen((filename),(mode)))==NULL
#endif

#define N 24
int main() {

  //u64 testKey[4] = {1,2,3,4};
  u64 testKey[] = {16294208416658607535ULL, 7960286522194355700ULL,   487617019471545679ULL, 17909611376780542444ULL};
  u64 rk[34] = {0};
  Speck128256KeySchedule(testKey, rk);
  u64 Pt[] = {0, 0};
  u64 Ct[] = {0, 0};
  for (int i = 0; i < N/2; i++) {
    Pt[0] = i;
    Speck128256Encrypt(Pt, Ct, rk);
    printf("%" PRIu64 "\n", Ct[0]);
    printf("%" PRIu64 "\n", Ct[1]);
  }

#if 0
  printf("------------------------\n");
  for (int i=0;i<34;i++){
    printf("%"PRIu64"\n",rk[i]);
  }
#endif
}