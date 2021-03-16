// Romu Pseudorandom Number Generators
//
// Copyright 2020 Mark A. Overton
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// ------------------------------------------------------------------------------------------------
//
// Website: romu-random.org
// Paper:   https://arxiv.org/abs/2002.11331
//
// Copy and paste the generator you want from those below.
// To compile, you will need to #include <stdint.h> and use the ROTL definition
// below.

#define ROTL(d, lrot) ((d << (lrot)) | (d >> (8 * sizeof(d) - (lrot))))

//===== RomuQuad
//==================================================================================
//
// More robust than anyone could need, but uses more registers than RomuTrio.
// Est. capacity >= 2^90 bytes. Register pressure = 8 (high). State size = 256
// bits.

uint64_t wState, xState, yState, zState; // set to nonzero seed

uint64_t romuQuad_random() {
  uint64_t wp = wState, xp = xState, yp = yState, zp = zState;
  wState = 15241094284759029579u * zp; // a-mult
  xState = zp + ROTL(wp, 52);          // b-rotl, c-add
  yState = yp - xp;                    // d-sub
  zState = yp + wp;                    // e-add
  zState = ROTL(zState, 19);           // f-rotl
  return xp;
}

//===== RomuTrio
//==================================================================================
//
// Great for general purpose work, including huge jobs.
// Est. capacity = 2^75 bytes. Register pressure = 6. State size = 192 bits.

// uint64_t txState, tyState, tzState;  // set to nonzero seed

uint64_t romuTrio_random() {
  uint64_t xp = xState, yp = yState, zp = zState;
  xState = 15241094284759029579u * zp;
  yState = yp - xp;
  yState = ROTL(yState, 12);
  zState = zp - yp;
  zState = ROTL(zState, 44);
  return xp;
}

/*
//===== RomuDuo
==================================================================================
//
// Might be faster than RomuTrio due to using fewer registers, but might
struggle with massive jobs.
// Est. capacity = 2^61 bytes. Register pressure = 5. State size = 128 bits.

// uint64_t xState, yState;  // set to nonzero seed

uint64_t romuDuo_random () {
   uint64_t xp = xState;
   xState = 15241094284759029579u * yState;
   yState = ROTL(yState,36) + ROTL(yState,15) - xp;
   return xp;
}


//===== RomuDuoJr
================================================================================
//
// The fastest generator using 64-bit arith., but not suited for huge jobs.
// Est. capacity = 2^51 bytes. Register pressure = 4. State size = 128 bits.

// uint64_t xState, yState;  // set to nonzero seed

uint64_t romuDuoJr_random () {
   uint64_t xp = xState;
   xState = 15241094284759029579u * yState;
   yState = yState - xp;  yState = ROTL(yState,27);
   return xp;
}


//===== RomuQuad32
================================================================================
//
// 32-bit arithmetic: Good for general purpose use.
// Est. capacity >= 2^62 bytes. Register pressure = 7. State size = 128 bits.

uint32_t wState, xState, yState, zState;  // set to nonzero seed

uint32_t romuQuad32 () {
   uint32_t wp = wState, xp = xState, yp = yState, zp = zState;
   wState = 3323815723u * zp;  // a-mult
   xState = zp + ROTL(wp,26);  // b-rotl, c-add
   yState = yp - xp;           // d-sub
   zState = yp + wp;           // e-add
   zState = ROTL(zState,9);    // f-rotl
   return xp;
}


//===== RomuTrio32
===============================================================================
//
// 32-bit arithmetic: Good for general purpose use, except for huge jobs.
// Est. capacity >= 2^53 bytes. Register pressure = 5. State size = 96 bits.

uint32_t xState, yState, zState;  // set to nonzero seed

uint32_t romuTrio32_random () {
   uint32_t xp = xState, yp = yState, zp = zState;
   xState = 3323815723u * zp;
   yState = yp - xp; yState = ROTL(yState,6);
   zState = zp - yp; zState = ROTL(zState,22);
   return xp;
}


//===== RomuMono32
===============================================================================
//
// 32-bit arithmetic: Suitable only up to 2^26 output-values. Outputs 16-bit
numbers.
// Fixed period of (2^32)-47. Must be seeded using the romuMono32_init function.
// Capacity = 2^27 bytes. Register pressure = 2. State size = 32 bits.

uint32_t state;

void romuMono32_init (uint32_t seed) {
   state = (seed & 0x1fffffffu) + 1156979152u;  // Accepts 29 seed-bits.
}

uint16_t romuMono32_random () {
   uint16_t result = state >> 16;
   state *= 3611795771u;  state = ROTL(state,12);
   return result;
}
*/