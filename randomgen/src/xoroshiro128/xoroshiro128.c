/*  Written in 2016-2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <https://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is xoroshiro128+ 1.0, our best and fastest small-state generator
   for floating-point numbers. We suggest to use its upper bits for
   floating-point generation, as it is slightly faster than
   xoroshiro128**. It passes all tests we are aware of except for the four
   lower bits, which might fail linearity tests (and just those), so if
   low linear complexity is not considered an issue (as it is usually the
   case) it can be used to generate 64-bit outputs, too; moreover, this
   generator has a very mild Hamming-weight dependency making our test
   (https://prng.di.unimi.it/hwd.php) fail after 5 TB of output; we believe
   this slight bias cannot affect any application. If you are concerned,
   use xoroshiro128** or xoshiro256+.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s.

   NOTE: the parameters (a=24, b=16, b=37) of this version give slightly
   better results in our test than the 2016 version (a=55, b=14, c=36).
*/

#include "xoroshiro128.h"

extern INLINE uint64_t xoroshiro128_next64(xoroshiro128_state_t *state);

extern INLINE uint64_t xoroshiro128plusplus_next64(xoroshiro128_state_t *state);

extern INLINE uint32_t xoroshiro128_next32(xoroshiro128_state_t *state);

extern INLINE uint32_t xoroshiro128plusplus_next32(xoroshiro128_state_t *state);

void xoroshiro128_jump(xoroshiro128_state_t *state)
{
  int i, b;
  uint64_t s0;
  uint64_t s1;
  static const uint64_t JUMP[] = {0xdf900294d8f554a5, 0x170865df4b3201fc};

  s0 = 0;
  s1 = 0;
  for (i = 0; i < (int)(sizeof(JUMP) / sizeof(*JUMP)); i++)
    for (b = 0; b < 64; b++)
    {
      if (JUMP[i] & UINT64_C(1) << b)
      {
        s0 ^= state->s[0];
        s1 ^= state->s[1];
      }
      xoroshiro128_next(&state->s[0]);
    }

  state->s[0] = s0;
  state->s[1] = s1;
}

void xoroshiro128plusplus_jump(xoroshiro128_state_t *state)
{
  static const uint64_t JUMP[] = { 0x2bd7a6a6e99c2ddc, 0x0992ccaf6a6fca05 };

  uint64_t s0 = 0;
  uint64_t s1 = 0;
  for(int i = 0; i < (int)(sizeof(JUMP) / sizeof(*JUMP)); i++)
    for(int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= state->s[0];
        s1 ^= state->s[1];
      }
      xoroshiro128plusplus_next(&state->s[0]);
    }

  state->s[0] = s0;
  state->s[1] = s1;
}
