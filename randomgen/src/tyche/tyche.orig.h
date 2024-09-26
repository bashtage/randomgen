// @HEADER
// *******************************************************************************
//                                OpenRAND                                       *
//   A Performance Portable, Reproducible Random Number Generation Library       *
//                                                                               *
// Copyright (c) 2023, Michigan State University                                 *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy  *
// of this software and associated documentation files (the "Software"), to deal *
// in the Software without restriction, including without limitation the rights  *
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
// copies of the Software, and to permit persons to whom the Software is         *
// furnished to do so, subject to the following conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in    *
// all copies or substantial portions of the Software.                           *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
// SOFTWARE.                                                                     *
//********************************************************************************
// @HEADER

// Modifications by Kevin Sheppard c2024. BSD 3-Clause License

#ifndef OPENRAND_TYCHE_H_
#define OPENRAND_TYCHE_H_

#include "base_state.h"

#include <cstdint>
#include <iostream>
#include <limits>


#if !defined(__USE_OPENRAND_TYCHE__) && !defined(__USE_ORIGINAL_TYCHE__)
error("No implementation selected for Tyche");
#endif

#ifdef __USE_OPENRAND_TYCHE__
#define STATE_VARIABLE a
#else // __USE_ORIGINAL_TYCHE__
#define STATE_VARIABLE b
#endif


namespace {
inline OPENRAND_DEVICE uint32_t rotl(uint32_t value, unsigned int x) {
  return (value << x) | (value >> (32 - x));
}
}  // namespace

namespace openrand {

class Tyche : public BaseRNG<Tyche> {
 public:
  OPENRAND_DEVICE Tyche(uint64_t seed, uint32_t ctr,
                        uint32_t global_seed = openrand::DEFAULT_GLOBAL_SEED) {
    // seed = seed ^ global_seed;
    a = static_cast<uint32_t>(seed >> 32);
    b = static_cast<uint32_t>(seed & 0xFFFFFFFFULL);
    d = d ^ ctr;

    for (int i = 0; i < 20; i++) {
      mix();
    }
  }

  template <typename T = uint32_t>
  OPENRAND_DEVICE T draw() {
    mix();
    if constexpr (std::is_same_v<T, uint32_t>)
      return STATE_VARIABLE;

    else {
      uint32_t tmp = STATE_VARIABLE;
      mix();
      uint64_t res = (static_cast<uint64_t>(tmp) << 32) | STATE_VARIABLE;
      return static_cast<T>(res);
    }
  }

 private:
#ifdef __USE_ORIGINAL_TYCHE__
   inline OPENRAND_DEVICE void mix() {
     a += b;
     d = rotl(d ^ a, 16);
     c += d;
     b = rotl(b ^ c, 12);
     a += b;
     d = rotl(d ^ a, 8);
     c += d;
     b = rotl(b ^ c, 7);
   }
#else // __USE_ORIGINAL_TYCHE__
 inline OPENRAND_DEVICE void mix() {
      b = rotl(b, 7) ^ c;
      c -= d;
      d = rotl(d, 8) ^ a;
      a -= b;
      b = rotl(b, 12) ^ c;
      c -= d;
      d = rotl(d, 16) ^ a;
      a -= b;
  }
#endif

  uint32_t a, b;
  uint32_t c = 2654435769;
  uint32_t d = 1367130551;
};  // class Tyche

}  // namespace openrand

#endif  // OPENRAND_TYCHE_H_
