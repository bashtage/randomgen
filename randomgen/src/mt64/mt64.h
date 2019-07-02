/*
   A C-program for MT19937-64 (2014/2/23 version).
   Coded by Takuji Nishimura and Makoto Matsumoto.

   This is a 64-bit version of Mersenne Twister pseudorandom number
   generator.

   Before using, initialize the state by using init_genrand64(seed)
   or init_by_array64(init_key, key_length).

   Copyright (C) 2004, 2014, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

   References:
   T. Nishimura, ``Tables of 64-bit Mersenne Twisters''
     ACM Transactions on Modeling and
     Computer Simulation 10. (2000) 348--357.
   M. Matsumoto and T. Nishimura,
     ``Mersenne Twister: a 623-dimensionally equidistributed
       uniform pseudorandom number generator''
     ACM Transactions on Modeling and
     Computer Simulation 8. (Jan. 1998) 3--30.

   Any feedback is very welcome.
   http://www.math.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove spaces)
*/

#ifndef _RANDOMDGEN__MT64_H_
#define _RANDOMDGEN__MT64_H_

#include "../common/randomgen_config.h"

#define NN 312
#define MM 156
#define MATRIX_A UINT64_C(0xB5026F5AA96619E9)
#define UM UINT64_C(0xFFFFFFFF80000000) /* Most significant 33 bits */
#define LM UINT64_C(0x7FFFFFFF)         /* Least significant 31 bits */

struct MT64_STATE_T {
  uint64_t mt[NN];
  int mti;
  int has_uint32;
  uint32_t uinteger;
};
typedef struct MT64_STATE_T mt64_state_t;

/* initializes mt[NN] with a seed */
extern void mt64_seed(mt64_state_t *mt64, uint64_t seed);

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
extern void mt64_init_by_array(mt64_state_t *mt64, uint64_t init_key[],
                               uint64_t key_length);

/* generates a random number on [0, 2^64-1]-interval */
uint64_t genrand64_int64(mt64_state_t *mt64);

extern void mt64_gen(mt64_state_t *mt64);

static INLINE uint64_t mt64_next(mt64_state_t *mt64) {
  uint64_t x;
  if (mt64->mti >= NN) { /* generate NN words at one time */
    /* Move to function to help inlining */
    mt64_gen(mt64);
  }
  x = mt64->mt[mt64->mti++];

  x ^= (x >> 29) & UINT64_C(0x5555555555555555);
  x ^= (x << 17) & UINT64_C(0x71D67FFFEDA60000);
  x ^= (x << 37) & UINT64_C(0xFFF7EEE000000000);
  x ^= (x >> 43);

  return x;
}

static INLINE uint64_t mt64_next64(mt64_state_t *mt64) { return mt64_next(mt64); }

static INLINE uint32_t mt64_next32(mt64_state_t *mt64) {
  uint64_t next;
  if (mt64->has_uint32) {
    mt64->has_uint32 = 0;
    return mt64->uinteger;
  }
  next = mt64_next(mt64);
  mt64->has_uint32 = 1;
  mt64->uinteger = (uint32_t)(next >> 32);
  return (uint32_t)(next & 0xffffffff);
}

static INLINE double mt64_next_double(mt64_state_t *mt64) {
  return (mt64_next(mt64) >> 11) * (1. / (UINT64_C(1) << 53));
}

#endif
